// Run-directory outputs (units in column names):
//   manifest.json, history.csv, summary.json (last; "complete": true when finished)
//   bulk_NNNN.vtu     hex mesh: displacement, stresses, phase fraction, layer, fragment
//   surface_NNNN.vtu  boundary faces with parent-element fields (light-weight rendering)
//   cracks_NNNN.vtu   interface faces: damage, opening, zone, kind, strength, toughness
//   xplanes_NNNN.csv  per x-plane: cracked fraction of the brittle layer, depth, opening
//   outputs.csv       output index -> time
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>

#include "fe.hpp"
#include "model.hpp"

#ifndef BSF_BUILD_ID
#define BSF_BUILD_ID "unknown"
#endif

namespace bsf {

std::string build_identifier() { return BSF_BUILD_ID; }

namespace {

std::string jstr(const std::string& s) {
  std::string o = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') o += '\\';
    if (c == '\n') {
      o += "\\n";
      continue;
    }
    o += c;
  }
  return o + "\"";
}

std::string jnum(double v) {
  if (!std::isfinite(v)) return "null";
  std::ostringstream o;
  o << std::setprecision(10) << v;
  return o.str();
}

// base64 encoder for VTK binary arrays
std::string b64(const unsigned char* d, size_t n) {
  static const char* tbl = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string o;
  o.reserve(4 * ((n + 2) / 3));
  size_t i = 0;
  for (; i + 2 < n; i += 3) {
    unsigned v = (d[i] << 16) | (d[i + 1] << 8) | d[i + 2];
    o += tbl[(v >> 18) & 63];
    o += tbl[(v >> 12) & 63];
    o += tbl[(v >> 6) & 63];
    o += tbl[v & 63];
  }
  if (i < n) {
    unsigned v = d[i] << 16;
    if (i + 1 < n) v |= d[i + 1] << 8;
    o += tbl[(v >> 18) & 63];
    o += tbl[(v >> 12) & 63];
    o += (i + 1 < n) ? tbl[(v >> 6) & 63] : '=';
    o += '=';
  }
  return o;
}

struct VtuWriter {
  std::ofstream f;
  bool binary;
  VtuWriter(const std::string& path, bool bin, size_t npts, size_t ncells) : f(path), binary(bin) {
    f << "<?xml version=\"1.0\"?>\n<VTKFile type=\"UnstructuredGrid\" version=\"1.0\" byte_order=\"LittleEndian\" "
         "header_type=\"UInt64\">\n<UnstructuredGrid>\n<Piece NumberOfPoints=\""
      << npts << "\" NumberOfCells=\"" << ncells << "\">\n";
  }
  template <class Tp>
  void array(const char* name, const char* type, int ncomp, const std::vector<Tp>& v) {
    f << "<DataArray type=\"" << type << "\" Name=\"" << name << "\" NumberOfComponents=\"" << ncomp
      << "\" format=\"" << (binary ? "binary" : "ascii") << "\">\n";
    if (binary) {
      uint64_t nb = v.size() * sizeof(Tp);
      std::vector<unsigned char> buf(8 + nb);
      std::memcpy(buf.data(), &nb, 8);
      if (nb) std::memcpy(buf.data() + 8, v.data(), nb);
      f << b64(buf.data(), buf.size()) << "\n";
    } else {
      f << std::setprecision(8);
      for (size_t i = 0; i < v.size(); ++i) f << v[i] << ((i + 1) % 12 ? ' ' : '\n');
      f << "\n";
    }
    f << "</DataArray>\n";
  }
  void close() { f << "</Piece>\n</UnstructuredGrid>\n</VTKFile>\n"; }
};

std::string idx4(int i) {
  std::ostringstream o;
  o << std::setw(4) << std::setfill('0') << i;
  return o.str();
}

}  // namespace

void Model::write_manifest() const {
  std::ofstream f(op.dir + "/manifest.json");
  double lcz[3];
  for (int z = 0; z < 3; ++z) {
    double E = (mode == "bimaterial") ? layer_mat[std::min<size_t>(z == 2 ? 1 : z, layer_mat.size() - 1)].E
                                      : (z == 1 ? tr.phase.E : layer_mat[0].E);
    lcz[z] = E * zones[z].toughness.mean / (zones[z].strength.mean * zones[z].strength.mean);
  }
  double hmax = std::max(mesh.p.L / mesh.p.nx, mesh.p.W / mesh.p.ny), hzmin = 1e300;
  for (size_t k = 1; k < mesh.z_nodes.size(); ++k) hzmin = std::min(hzmin, mesh.z_nodes[k] - mesh.z_nodes[k - 1]);
  f << "{\n \"solver\": \"stripcz\",\n \"build_id\": " << jstr(build_identifier()) << ",\n \"mode\": " << jstr(mode)
    << ",\n \"input_file\": " << jstr(cfg.source()) << ",\n \"labels\": {";
  {
    std::stringstream ss(cfg.get_string("output.labels", ""));
    std::string kv;
    bool first = true;
    while (std::getline(ss, kv, ';')) {
      auto eq = kv.find('=');
      if (eq == std::string::npos) continue;
      f << (first ? "" : ", ") << jstr(trim(kv.substr(0, eq))) << ": " << jstr(trim(kv.substr(eq + 1)));
      first = false;
    }
  }
  f << "},\n \"config\": {";
  {
    std::stringstream ss(cfg.dump());
    std::string line;
    bool first = true;
    while (std::getline(ss, line)) {
      auto eq = line.find(" = ");
      if (eq == std::string::npos) continue;
      f << (first ? "\n  " : ",\n  ") << jstr(line.substr(0, eq)) << ": " << jstr(line.substr(eq + 3));
      first = false;
    }
  }
  f << "\n },\n \"derived\": {\n  \"n_nodes\": " << mesh.nnode() << ",\n  \"n_elements\": " << mesh.nelem()
    << ",\n  \"n_interface_elements\": " << mesh.ifaces.size() << ",\n  \"thickness_m\": " << jnum(mesh.z_nodes.back());
  for (size_t l = 0; l < mesh.p.layers.size(); ++l)
    f << ",\n  \"layer" << l << "_thickness_m\": " << jnum(mesh.p.layers[l].thickness);
  for (int z = 0; z < 3; ++z)
    f << ",\n  \"zone" << z << "_name\": " << jstr(zones[z].name) << ",\n  \"zone" << z
      << "_cohesive_length_m\": " << jnum(lcz[z]) << ",\n  \"zone" << z << "_zeta_min\": " << jnum(lcz[z] / hmax);
  f << ",\n  \"h_inplane_max_m\": " << jnum(hmax) << ",\n  \"h_z_min_m\": " << jnum(hzmin);
  if (tr.enabled)
    f << ",\n  \"transformation_rate\": " << jnum(tr.rate) << ",\n  \"transformation_exponent\": " << jnum(tr.exponent)
      << ",\n  \"strain_normal\": " << jnum(tr.e_normal) << ",\n  \"strain_inplane\": " << jnum(tr.e_inplane)
      << ",\n  \"volume_ratio_check\": "
      << jnum((1 + tr.e_inplane) * (1 + tr.e_inplane) * (1 + tr.e_normal));
  f << ",\n  \"Gc_clamped_points\": " << n_gc_clamped << "\n }\n}\n";
  std::ofstream(op.dir + "/input_used.ini") << cfg.dump();
}

void Model::open_history() {
  hist = std::fopen((op.dir + "/history.csv").c_str(), "w");
  std::fprintf(hist,
               "step,time_s,dt_s,temperature_K,dT_K,front_depth_um,crack_area_mm2,crack_area_x_mm2,crack_area_y_mm2,"
               "crack_area_z_mm2,crack_area_layer_mm2,crack_area_zone0_mm2,crack_area_zone1_mm2,crack_area_zone2_mm2,"
               "damaged_area_mm2,dissipated_energy_mJ,n_channel_cracks,mean_spacing_mm,max_crack_depth_um,"
               "delaminated_area_mm2,curvature_1pm,sxx_brittle_MPa,sxx_substrate_MPa,max_principal_MPa,n_fragments,"
               "pin_reaction_N,newton_iters,pcg_iters,residual,wall_s\n");
}

void Model::write_history_row(const Diagnostics& dg) {
  using namespace std::chrono;
  double wall = duration<double>(steady_clock::now().time_since_epoch()).count() - wall_start;
  double area = met.area[0] + met.area[1] + met.area[2] + met.area[3];
  double T = temperature(time);
  std::fprintf(hist,
               "%d,%.10g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%.6g,%d,%.6g,%.6g,%.6g,%.6g,"
               "%.6g,%.6g,%.6g,%d,%.6g,%d,%d,%.4g,%.4g\n",
               step, time, step ? time - last["time_s"] : 0.0, T, T - T_ref, s_front * 1e6, area * 1e6,
               met.area[0] * 1e6, met.area[1] * 1e6, met.area[2] * 1e6, met.area[3] * 1e6, met.area_zone[0] * 1e6,
               met.area_zone[1] * 1e6, met.area_zone[2] * 1e6, met.damaged_area * 1e6, met.dissipated * 1e3,
               met.n_channel, met.mean_spacing * 1e3, met.max_crack_depth * 1e6, met.delaminated_area * 1e6,
               met.curvature, met.sxx_top * 1e-6, met.sxx_bot * 1e-6, met.max_principal * 1e-6, met.n_fragments,
               met.pin_reaction, dg.newton_iters, dg.lin_iters, dg.res, wall);
  std::fflush(hist);
  last["time_s"] = time;
  last["crack_area_mm2"] = area * 1e6;
  last["dissipated_energy_mJ"] = met.dissipated * 1e3;
  last["n_channel_cracks"] = met.n_channel;
  last["mean_spacing_mm"] = met.mean_spacing * 1e3;
  last["delaminated_area_mm2"] = met.delaminated_area * 1e6;
  last["curvature_1pm"] = met.curvature;
  last["dT_K"] = T - T_ref;
  last["front_depth_um"] = s_front * 1e6;
  last["n_fragments"] = met.n_fragments;
  last["max_crack_depth_um"] = met.max_crack_depth * 1e6;
  last["wall_s"] = wall;
}

void Model::write_vtu(int idx) {
  const std::string s = idx4(idx);
  const int nn = int(mesh.nnode()), ne = int(mesh.nelem());
  std::vector<int> lab, erid;
  label_fragments(lab, erid);
  std::vector<double> sig6(size_t(ne) * 6), smax(ne), sxx(ne), szz(ne), sxz(ne), phi(ne);
  std::vector<int32_t> frag(ne), layer(ne);
#pragma omp parallel for
  for (int e = 0; e < ne; ++e) {
    double sg[6], ev[3], ph;
    element_state(e, s_front, sg, ph);
    sym_eig3(sg, ev);
    for (int k = 0; k < 6; ++k) sig6[size_t(e) * 6 + k] = sg[k] * 1e-6;
    smax[e] = ev[0] * 1e-6;
    sxx[e] = sg[0] * 1e-6;
    szz[e] = sg[2] * 1e-6;
    sxz[e] = sg[5] * 1e-6;
    phi[e] = ph;
  }
  for (int e = 0; e < ne; ++e) {
    frag[e] = lab[erid[e]];
    layer[e] = mesh.elem_layer[e];
  }
  {
    VtuWriter w(op.dir + "/bulk_" + s + ".vtu", op.binary, nn, ne);
    std::vector<double> pts(size_t(3) * nn);
    for (int n = 0; n < nn; ++n)
      for (int i = 0; i < 3; ++i) pts[3 * n + i] = mesh.X[n][i];
    w.f << "<Points>\n";
    w.array("Points", "Float64", 3, pts);
    w.f << "</Points>\n<Cells>\n";
    std::vector<int32_t> conn(size_t(8) * ne), offs(ne);
    std::vector<uint8_t> types(ne, 12);
    for (int e = 0; e < ne; ++e) {
      for (int a = 0; a < 8; ++a) conn[size_t(8) * e + a] = mesh.hex[e][a];
      offs[e] = 8 * (e + 1);
    }
    w.array("connectivity", "Int32", 1, conn);
    w.array("offsets", "Int32", 1, offs);
    w.array("types", "UInt8", 1, types);
    w.f << "</Cells>\n<PointData Vectors=\"displacement_m\">\n";
    w.array("displacement_m", "Float64", 3, u);
    w.f << "</PointData>\n<CellData Scalars=\"max_principal_MPa\">\n";
    w.array("max_principal_MPa", "Float64", 1, smax);
    w.array("sxx_MPa", "Float64", 1, sxx);
    w.array("szz_MPa", "Float64", 1, szz);
    w.array("sxz_MPa", "Float64", 1, sxz);
    w.array("stress_MPa", "Float64", 6, sig6);
    w.array("phase_fraction", "Float64", 1, phi);
    w.array("layer", "Int32", 1, layer);
    w.array("fragment", "Int32", 1, frag);
    w.f << "</CellData>\n";
    w.close();
  }
  {
    std::vector<const BFace*> faces;
    for (auto* v : {&mesh.top, &mesh.bottom, &mesh.xmin, &mesh.xmax, &mesh.ymin, &mesh.ymax})
      for (auto& b : *v) faces.push_back(&b);
    const size_t nf = faces.size();
    VtuWriter w(op.dir + "/surface_" + s + ".vtu", op.binary, 4 * nf, nf);
    std::vector<double> pts(12 * nf), disp(12 * nf), sm(nf), sx(nf), ph(nf);
    std::vector<int32_t> conn(4 * nf), offs(nf), fr(nf), ly(nf);
    std::vector<uint8_t> types(nf, 9);
    for (size_t k = 0; k < nf; ++k) {
      for (int a = 0; a < 4; ++a) {
        int n = faces[k]->n[a];
        for (int i = 0; i < 3; ++i) {
          pts[12 * k + 3 * a + i] = mesh.X[n][i];
          disp[12 * k + 3 * a + i] = u[3 * n + i];
        }
        conn[4 * k + a] = int32_t(4 * k + a);
      }
      offs[k] = int32_t(4 * (k + 1));
      int e = faces[k]->elem;
      sm[k] = smax[e];
      sx[k] = sxx[e];
      ph[k] = phi[e];
      fr[k] = frag[e];
      ly[k] = layer[e];
    }
    w.f << "<Points>\n";
    w.array("Points", "Float64", 3, pts);
    w.f << "</Points>\n<Cells>\n";
    w.array("connectivity", "Int32", 1, conn);
    w.array("offsets", "Int32", 1, offs);
    w.array("types", "UInt8", 1, types);
    w.f << "</Cells>\n<PointData>\n";
    w.array("displacement_m", "Float64", 3, disp);
    w.f << "</PointData>\n<CellData>\n";
    w.array("max_principal_MPa", "Float64", 1, sm);
    w.array("sxx_MPa", "Float64", 1, sx);
    w.array("phase_fraction", "Float64", 1, ph);
    w.array("fragment", "Int32", 1, fr);
    w.array("layer", "Int32", 1, ly);
    w.f << "</CellData>\n";
    w.close();
  }
  {
    const size_t ni = mesh.ifaces.size();
    VtuWriter w(op.dir + "/cracks_" + s + ".vtu", op.binary, 4 * ni, ni);
    std::vector<double> pts(12 * ni), disp(12 * ni), dmg(ni), open(ni), sg(ni), gc(ni), tn(ni);
    std::vector<int32_t> conn(4 * ni), offs(ni), kind(ni), zone(ni), plane(ni);
    std::vector<uint8_t> types(ni, 9);
    for (size_t f = 0; f < ni; ++f) {
      const auto& ifc = mesh.ifaces[f];
      for (int a = 0; a < 4; ++a) {
        for (int i = 0; i < 3; ++i) {
          pts[12 * f + 3 * a + i] = mesh.X[ifc.a[a]][i];
          disp[12 * f + 3 * a + i] = 0.5 * (u[3 * ifc.a[a] + i] + u[3 * ifc.b[a] + i]);
        }
        conn[4 * f + a] = int32_t(4 * f + a);
      }
      offs[f] = int32_t(4 * (f + 1));
      double d = 0, o = -1e300, s1 = 0, g1 = 0, t1 = 0;
      for (int g = 0; g < 4; ++g) {
        size_t q = 4 * f + g;
        int zn = gp_zone_now[q];
        d += 0.25 * gp_state[q].d;
        o = std::max(o, double(gp_open[q]));
        s1 += 0.25 * gp_props[q][zn].sigma_t;
        g1 += 0.25 * gp_props[q][zn].Gc;
        t1 += 0.25 * gp_tn[q];
      }
      dmg[f] = d;
      open[f] = o * 1e6;
      sg[f] = s1 * 1e-6;
      gc[f] = g1;
      tn[f] = t1 * 1e-6;
      kind[f] = ifc.kind;
      zone[f] = gp_zone_now[4 * f];
      plane[f] = ifc.plane;
    }
    w.f << "<Points>\n";
    w.array("Points", "Float64", 3, pts);
    w.f << "</Points>\n<Cells>\n";
    w.array("connectivity", "Int32", 1, conn);
    w.array("offsets", "Int32", 1, offs);
    w.array("types", "UInt8", 1, types);
    w.f << "</Cells>\n<PointData>\n";
    w.array("displacement_m", "Float64", 3, disp);
    w.f << "</PointData>\n<CellData Scalars=\"damage\">\n";
    w.array("damage", "Float64", 1, dmg);
    w.array("opening_um", "Float64", 1, open);
    w.array("strength_MPa", "Float64", 1, sg);
    w.array("fracture_energy_Jpm2", "Float64", 1, gc);
    w.array("normal_traction_MPa", "Float64", 1, tn);
    w.array("kind", "Int32", 1, kind);
    w.array("zone", "Int32", 1, zone);
    w.array("plane", "Int32", 1, plane);
    w.f << "</CellData>\n";
    w.close();
  }
}

void Model::write_pvd() const {
  {
    std::ofstream f(op.dir + "/outputs.csv");
    f << "index,time_s\n";
    for (auto& [t, i] : vtu_list) f << i << "," << std::setprecision(10) << t << "\n";
  }
  if (!op.vtu) return;
  for (std::string kind : {"bulk", "surface", "cracks"}) {
    std::ofstream f(op.dir + "/" + kind + ".pvd");
    f << "<?xml version=\"1.0\"?>\n<VTKFile type=\"Collection\" version=\"0.1\">\n<Collection>\n";
    for (auto& [t, i] : vtu_list)
      f << "<DataSet timestep=\"" << std::setprecision(10) << t << "\" file=\"" << kind << "_" << idx4(i)
        << ".vtu\"/>\n";
    f << "</Collection>\n</VTKFile>\n";
  }
}

void Model::write_xplanes(int idx) const {
  // per x-plane: cracked fraction of the brittle zone (top layer / transformed phase), crack depth
  std::map<int, std::array<double, 5>> pl;  // area_brittle, cracked_brittle, depth, x, max opening at top
  const double H = mesh.z_nodes.back();
  for (size_t f = 0; f < mesh.ifaces.size(); ++f) {
    if (mesh.ifaces[f].kind != IF_X) continue;
    for (int g = 0; g < 4; ++g) {
      size_t q = 4 * f + g;
      auto& r = pl[mesh.ifaces[f].plane];
      r[3] = gp_x[q][0];
      bool cracked = gp_state[q].d >= coh.d_crack;
      if (gp_zone_now[q] == 1) {
        r[0] += gp_dA[q];
        if (cracked) r[1] += gp_dA[q];
      }
      if (cracked) r[2] = std::max(r[2], H - gp_x[q][2]);
      r[4] = std::max(r[4], double(gp_open[q]));
    }
  }
  std::ofstream f(op.dir + "/xplanes_" + idx4(idx) + ".csv");
  f << "plane,x_mm,brittle_cracked_fraction,crack_depth_um,max_opening_um\n";
  for (auto& [p, r] : pl)
    f << p << "," << r[3] * 1e3 << "," << (r[0] > 0 ? r[1] / r[0] : 0.0) << "," << r[2] * 1e6 << "," << r[4] * 1e6
      << "\n";
}

void Model::write_summary(bool complete, const std::string& msg) const {
  std::vector<int> lab, erid;
  int nf = label_fragments(lab, erid);
  std::vector<double> vol(nf, 0.0);
  for (size_t e = 0; e < mesh.nelem(); ++e) vol[lab[erid[e]]] += elem_vol[e];
  std::sort(vol.begin(), vol.end());
  std::ofstream f(op.dir + "/summary.json");
  f << "{\n \"complete\": " << (complete ? "true" : "false") << ",\n \"message\": " << jstr(msg)
    << ",\n \"mode\": " << jstr(mode) << ",\n \"steps\": " << step << ",\n \"time_s\": " << jnum(time)
    << ",\n \"build_id\": " << jstr(build_identifier());
  for (auto& [k, v] : last) f << ",\n \"final_" << k << "\": " << jnum(v);
  f << ",\n \"onset_time_s\": " << jnum(onset_time < 0 ? NAN : onset_time) << ",\n \"onset_dT_K\": "
    << jnum(mode == "bimaterial" && onset_load >= 0 ? onset_load : NAN) << ",\n \"onset_front_depth_um\": "
    << jnum(mode == "transformation" && onset_load >= 0 ? onset_load * 1e6 : NAN) << ",\n \"crack_area_by_kind_mm2\": {";
  for (int k = 0; k < IF_NKINDS; ++k) f << (k ? ", " : "") << jstr(iface_kind_name(k)) << ": " << jnum(met.area[k] * 1e6);
  f << "},\n \"crack_area_by_zone_mm2\": {";
  for (int z = 0; z < 3; ++z) f << (z ? ", " : "") << jstr(zones[z].name) << ": " << jnum(met.area_zone[z] * 1e6);
  f << "},\n \"fragment_volumes_mm3\": [";
  for (int i = 0; i < nf; ++i) f << (i ? ", " : "") << jnum(vol[i] * 1e9);
  f << "]\n}\n";
}

}  // namespace bsf
