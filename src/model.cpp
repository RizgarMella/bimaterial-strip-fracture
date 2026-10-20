#include "model.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

#include "fe.hpp"

namespace bsf {

double Table::operator()(double x) const {
  if (t.empty()) return 0.0;
  if (x <= t.front()) return v.front();
  if (x >= t.back()) return v.back();
  auto it = std::upper_bound(t.begin(), t.end(), x);
  size_t i = size_t(it - t.begin());
  double f = (x - t[i - 1]) / (t[i] - t[i - 1]);
  return v[i - 1] + f * (v[i] - v[i - 1]);
}

Table read_table(const Config& c, const std::string& key, double def) {
  Table T;
  auto g = c.get_groups(key);
  if (g.empty()) {
    T.set_constant(def);
    return T;
  }
  for (auto& r : g) {
    if (r.size() < 2) throw std::runtime_error(key + ": expected 'time, value; time, value; ...'");
    if (!T.t.empty() && r[0] <= T.t.back()) throw std::runtime_error(key + ": times must increase");
    T.t.push_back(r[0]);
    T.v.push_back(r[1]);
  }
  return T;
}

void ZoneFracture::read(const Config& c, const std::string& s, double s_mean, double g_mean, double m, uint64_t seed0) {
  name = s.rfind("fracture.", 0) == 0 ? s.substr(9) : s;
  strength.read(c, s + ".strength", s_mean, m);
  toughness.read(c, s + ".toughness", g_mean, m);
  corr_length = c.get_double(s + ".correlation_length", corr_length);
  rho = c.get_double(s + ".toughness_strength_correlation", rho);
  reference_area = c.get_double(s + ".reference_area", reference_area);
  seed = uint64_t(c.get_int(s + ".seed", int(seed0)));
}

void Transformation::read(const Config& c, double H) {
  enabled = true;
  exponent = c.get_double("transformation.exponent", exponent);
  rate = c.get_double("transformation.rate", rate);
  if (c.has("transformation.depth_at_time")) {  // "s_ref, t_ref" -> k = s_ref^n / t_ref
    auto v = c.get_list("transformation.depth_at_time", {});
    if (v.size() != 2 || v[1] <= 0) throw std::runtime_error("transformation.depth_at_time = depth, time");
    rate = std::pow(v[0], exponent) / v[1];
  }
  t_start = c.get_double("transformation.start_time", t_start);
  max_depth = c.get_double("transformation.max_depth", 0.95 * H);
  pbr = c.get_double("transformation.pilling_bedworth_ratio", pbr);
  anisotropy = c.get_double("transformation.anisotropy", anisotropy);
  phase.E = c.get_double("phase.youngs_modulus", 200e9);
  phase.nu = c.get_double("phase.poisson_ratio", 0.3);
  phase.alpha = c.get_double("phase.thermal_expansion", 10e-6);
  if (pbr <= 0) throw std::runtime_error("transformation.pilling_bedworth_ratio must be > 0");
  if (anisotropy < 0) throw std::runtime_error("transformation.anisotropy must be >= 0");
  // solve (1 + a e)^2 (1 + e) = PBR for the normal strain e (Newton)
  double e = std::cbrt(pbr) - 1.0;
  for (int it = 0; it < 100; ++it) {
    double f = (1 + anisotropy * e) * (1 + anisotropy * e) * (1 + e) - pbr;
    double df = 2 * anisotropy * (1 + anisotropy * e) * (1 + e) + (1 + anisotropy * e) * (1 + anisotropy * e);
    double de = f / df;
    e -= de;
    if (std::abs(de) < 1e-15) break;
  }
  e_normal = e;
  e_inplane = anisotropy * e;
}

double Transformation::depth(double t) const {
  if (!enabled || t <= t_start || rate <= 0) return 0.0;
  return std::min(max_depth, std::pow(rate * (t - t_start), 1.0 / exponent));
}

Model::Model(const Config& c) : cfg(c) {
  mode = c.get_string("model.type", "bimaterial");
  if (mode != "bimaterial" && mode != "transformation")
    throw std::runtime_error("model.type must be bimaterial | transformation");
  T_ref = c.get_double("material.reference_temperature", T_ref);
  temperature_ = read_table(c, "loading.temperature_history", T_ref);
  // cohesive law
  coh.K = c.get_double("cohesive.stiffness", 1e17);
  coh.Ks_ratio = c.get_double("cohesive.shear_stiffness_ratio", 1.0);
  coh.beta = c.get_double("cohesive.mode_mixity_beta", 0.0);
  std::string soft = c.get_string("cohesive.softening", "linear");
  if (soft != "linear" && soft != "exponential") throw std::runtime_error("cohesive.softening: linear|exponential");
  coh.softening = soft == "exponential";
  coh.viscosity = c.get_double("cohesive.viscosity", 1e-4);
  coh.friction = c.get_double("cohesive.crack_face_friction", 0.0);
  coh.residual = c.get_double("cohesive.residual_stiffness", 1e-6);
  coh.d_crack = c.get_double("cohesive.cracked_damage", 0.99);
  // zones
  if (mode == "bimaterial") {
    zones[0].read(c, "fracture.bottom", 400e6, 100.0, 15.0, 1);
    zones[1].read(c, "fracture.top", 150e6, 10.0, 6.0, 2);
    zones[2].read(c, "fracture.interface", 200e6, 20.0, 8.0, 3);
  } else {
    zones[0].read(c, "fracture.substrate", 400e6, 100.0, 15.0, 1);
    zones[1].read(c, "fracture.phase", 150e6, 5.0, 6.0, 2);
    zones[2].read(c, "fracture.front", 150e6, 10.0, 8.0, 3);
  }
  flaws = c.get_groups("fracture.flaws");
  bc = c.get_string("boundary.mode", "free");
  if (bc != "free" && bc != "roller_bottom" && bc != "clamped_bottom" && bc != "clamped_end")
    throw std::runtime_error("boundary.mode must be free | roller_bottom | clamped_bottom | clamped_end");
  plane_strain_y = c.get_bool("boundary.plane_strain_y", false);
  // time / solver / output
  tp.t_end = c.get_double("time.end", tp.t_end);
  tp.dt0 = c.get_double("time.dt_initial", tp.dt0);
  tp.dt_min = c.get_double("time.dt_min", tp.dt_min);
  tp.dt_max = c.get_double("time.dt_max", tp.dt_max);
  tp.grow = c.get_double("time.grow", tp.grow);
  tp.cut = c.get_double("time.cut", tp.cut);
  tp.max_load_change = c.get_double("time.max_load_change", tp.max_load_change);
  tp.max_damage_increment = c.get_double("time.max_damage_increment", tp.max_damage_increment);
  tp.softening_dt_factor = c.get_double("time.softening_dt_factor", tp.softening_dt_factor);
  tp.initiation_target = c.get_double("time.initiation_target", tp.initiation_target);
  tp.max_steps = c.get_int("time.max_steps", tp.max_steps);
  sp.newton_max = c.get_int("solver.newton_max_iterations", sp.newton_max);
  sp.newton_tol = c.get_double("solver.newton_tolerance", sp.newton_tol);
  sp.newton_dtol = c.get_double("solver.displacement_tolerance", sp.newton_dtol);
  sp.lin_rtol = c.get_double("solver.linear_tolerance", sp.lin_rtol);
  sp.lin_maxit = c.get_int("solver.linear_max_iterations", sp.lin_maxit);
  sp.stabilization = c.get_double("solver.stabilization", sp.stabilization);
  sp.line_search = c.get_bool("solver.line_search", sp.line_search);
  sp.amg = c.get_string("solver.preconditioner", "amg") == "amg";
  sp.amg_rebuild_iters = c.get_int("solver.amg_rebuild_iterations", sp.amg_rebuild_iters);
  op.dir = c.get_string("output.directory", op.dir);
  op.times = c.get_list("output.times", {});
  op.every = c.get_int("output.every_steps", op.every);
  op.on_crack_growth = c.get_double("output.on_crack_area_growth", op.on_crack_growth);
  op.vtu = c.get_bool("output.vtu", op.vtu);
  op.binary = c.get_bool("output.binary", op.binary);
  op.max_vtu = c.get_int("output.max_vtu", op.max_vtu);
  tp.stops = op.times;
  for (double t : temperature_.t) tp.stops.push_back(t);
  tp.stops.push_back(tp.t_end);
  std::sort(tp.stops.begin(), tp.stops.end());
  tp.stops.erase(std::remove_if(tp.stops.begin(), tp.stops.end(), [&](double t) { return t <= 0 || t > tp.t_end; }),
                 tp.stops.end());
  tp.stops.erase(std::unique(tp.stops.begin(), tp.stops.end()), tp.stops.end());
}

double Model::row_phase(int k, double s) const {
  if (!tr.enabled || s <= 0) return 0.0;
  const double zf = mesh.z_nodes.back() - s;  // front height
  const double z0 = mesh.z_nodes[k], z1 = mesh.z_nodes[k + 1];
  return std::min(1.0, std::max(0.0, (z1 - zf) / (z1 - z0)));
}

void Model::setup() {
  StripParams spm;
  spm.read(cfg);
  if (mode == "transformation" && cfg.get_int("geometry.n_layers", 2) != 1)
    throw std::runtime_error("model.type = transformation uses a single layer: set geometry.n_layers = 1");
  mesh.build(spm);
  std::cout << mesh.summary() << std::endl;
  const int nn = int(mesh.nnode()), ne = int(mesh.nelem());
  // layer materials
  layer_mat.assign(mesh.p.layers.size(), Elastic{});
  for (size_t l = 0; l < mesh.p.layers.size(); ++l) {
    std::string s = "layer" + std::to_string(l);
    Elastic def;
    if (mode == "bimaterial" && l == 1) def = Elastic{380e9, 0.22, 8e-6};
    if (mode == "bimaterial" && l == 0) def = Elastic{200e9, 0.30, 17e-6};
    if (mode == "transformation") def = Elastic{95e9, 0.34, 6e-6};
    layer_mat[l].E = cfg.get_double(s + ".youngs_modulus", def.E);
    layer_mat[l].nu = cfg.get_double(s + ".poisson_ratio", def.nu);
    layer_mat[l].alpha = cfg.get_double(s + ".thermal_expansion", def.alpha);
  }
  if (mode == "transformation") tr.read(cfg, mesh.z_nodes.back());

  elem_vol.assign(ne, 0);
  elem_hz.assign(ne, 0);
  for (int e = 0; e < ne; ++e) {
    double X[8][3], N[8], B[8][3], dV;
    for (int a = 0; a < 8; ++a)
      for (int i = 0; i < 3; ++i) X[a][i] = mesh.X[mesh.hex[e][a]][i];
    for (int g = 0; g < 8; ++g) {
      if (!hex_gp(X, g, N, B, dV)) throw std::runtime_error("bad element " + std::to_string(e));
      elem_vol[e] += dV;
    }
    int k = mesh.elem_ijk[e][2];
    elem_hz[e] = mesh.z_nodes[k + 1] - mesh.z_nodes[k];
  }

  // interface points
  const size_t ni = mesh.ifaces.size();
  ngp = 4 * ni;
  gp_dA.assign(ngp, 0);
  gp_R.assign(ngp, {});
  gp_x.assign(ngp, {});
  gp_row.assign(ngp, 0);
  gp_props.assign(ngp, {});
  gp_static_zone.assign(ngp, 0);
  gp_state.assign(ngp, GPState{});
  gp_trial = gp_state;
  gp_open.assign(ngp, 0.f);
  gp_tn.assign(ngp, 0.f);
  gp_ts.assign(ngp, 0.f);
  gp_zone_now.assign(ngp, 0);
  const int top_layer = int(mesh.p.layers.size()) - 1;
  for (size_t f = 0; f < ni; ++f) {
    const auto& ifc = mesh.ifaces[f];
    double X[4][3];
    for (int a = 0; a < 4; ++a)
      for (int i = 0; i < 3; ++i) X[a][i] = mesh.X[ifc.a[a]][i];
    for (int g = 0; g < 4; ++g) {
      double N[4], dA, R[3][3], x[3];
      quad_gp(X, g, N, dA, R, x);
      size_t q = 4 * f + g;
      gp_dA[q] = dA;
      for (int i = 0; i < 3; ++i) {
        gp_x[q][i] = x[i];
        for (int j = 0; j < 3; ++j) gp_R[q][3 * i + j] = R[i][j];
      }
      // element row of the point: for a horizontal plane the row below it (plane index - 1), for a
      // vertical plane the row containing the point
      if (ifc.kind == IF_Z || ifc.kind == IF_LAYER) {
        gp_row[q] = ifc.plane - 1;
      } else {
        auto it = std::upper_bound(mesh.z_nodes.begin(), mesh.z_nodes.end(), x[2]);
        gp_row[q] = std::min(int(mesh.z_nodes.size()) - 2, std::max(0, int(it - mesh.z_nodes.begin()) - 1));
      }
      if (mode == "bimaterial") {
        if (ifc.kind == IF_LAYER)
          gp_static_zone[q] = 2;
        else
          gp_static_zone[q] = (ifc.layer == top_layer && mesh.p.layers.size() > 1) ? 1 : 0;
      } else {
        gp_static_zone[q] = -1;
      }
    }
  }
  // Weibull properties of every zone at every point
  double lo[3] = {0, 0, 0}, hi[3] = {mesh.p.L, mesh.p.W, mesh.z_nodes.back()};
  n_gc_clamped = 0;
  for (int z = 0; z < 3; ++z) {
    const auto& Z = zones[z];
    GaussianField f1, f2;
    const bool corr = Z.corr_length > 0;
    const bool need2 = corr && Z.toughness.dist == "weibull" && Z.rho < 1.0;
    if (corr) {
      f1.generate(lo, hi, Z.corr_length, Z.seed * 7907 + 11);
      if (need2) f2.generate(lo, hi, Z.corr_length, Z.seed * 104729 + 23);
    }
    std::mt19937_64 rng(Z.seed * 7907 + 11);
    std::normal_distribution<double> nd(0, 1);
    const double rho = std::max(-1.0, std::min(1.0, Z.rho)), rr = std::sqrt(std::max(0.0, 1 - rho * rho));
    for (size_t f = 0; f < ni; ++f) {
      double z1 = 0, z2 = 0, Ael = 0;
      if (!corr) {
        z1 = nd(rng);
        z2 = nd(rng);
        for (int g = 0; g < 4; ++g) Ael += gp_dA[4 * f + g];
      }
      for (int g = 0; g < 4; ++g) {
        size_t q = 4 * f + g;
        const auto& x = gp_x[q];
        double xi1 = corr ? f1(x[0], x[1], x[2]) : z1;
        double xi2 = corr ? (need2 ? f2(x[0], x[1], x[2]) : 0.0) : z2;
        double sig = Z.strength.quantile(normal_cdf(xi1));
        if (!corr && Z.reference_area > 0 && Z.strength.dist == "weibull")
          sig = Z.strength.threshold +
                (sig - Z.strength.threshold) * std::pow(Z.reference_area / Ael, 1.0 / Z.strength.modulus);
        double gc = Z.toughness.quantile(normal_cdf(rho * xi1 + rr * xi2));
        for (auto& fl : flaws) {
          if (fl.size() < 5) throw std::runtime_error("fracture.flaws: x,y,z,radius,strength_factor[,toughness_factor]");
          double dx = x[0] - fl[0], dy = x[1] - fl[1], dz = x[2] - fl[2];
          if (dx * dx + dy * dy + dz * dz <= fl[3] * fl[3]) {
            sig *= fl[4];
            if (fl.size() > 5) gc *= fl[5];
          }
        }
        double gmin = 1.05 * min_fracture_energy(coh, sig);
        if (gc < gmin) {
          gc = gmin;
          ++n_gc_clamped;
        }
        gp_props[q][z].sigma_t = sig;
        gp_props[q][z].Gc = gc;
      }
    }
  }
  if (n_gc_clamped) std::cout << "warning: G_c raised at " << n_gc_clamped << " points (softening branch)\n";

  // boundary conditions
  mask_u.assign(size_t(3) * nn, 1.0);
  auto fix_face = [&](const std::vector<BFace>& faces, int comp) {
    for (auto& bf : faces)
      for (int a = 0; a < 4; ++a) mask_u[3 * bf.n[a] + comp] = 0.0;
  };
  const int nx = mesh.p.nx, ny = mesh.p.ny;
  auto elem_at = [&](int i, int j, int k) { return (k * ny + j) * nx + i; };
  int n000 = mesh.hex[elem_at(0, 0, 0)][0], n100 = mesh.hex[elem_at(nx - 1, 0, 0)][1];
  int n010 = mesh.hex[elem_at(0, ny - 1, 0)][3];
  if (bc == "free") {
    // 3-2-1 statically determinate support (no reactions for self-equilibrated loading)
    pin_dofs = {3 * n000, 3 * n000 + 1, 3 * n000 + 2, 3 * n100 + 1, 3 * n100 + 2, 3 * n010 + 2};
    if (plane_strain_y) pin_dofs = {3 * n000, 3 * n000 + 2, 3 * n100 + 2};
  } else if (bc == "roller_bottom") {
    fix_face(mesh.bottom, 2);
    pin_dofs = {3 * n000, 3 * n000 + 1, 3 * n100 + 1};
    if (plane_strain_y) pin_dofs = {3 * n000};
  } else if (bc == "clamped_bottom") {
    for (int c = 0; c < 3; ++c) fix_face(mesh.bottom, c);
  } else if (bc == "clamped_end") {
    for (int c = 0; c < 3; ++c) fix_face(mesh.xmin, c);
  }
  for (int d : pin_dofs) mask_u[d] = 0.0;
  if (plane_strain_y) {
    fix_face(mesh.ymin, 1);
    fix_face(mesh.ymax, 1);
  }
  {
    std::vector<char> isb(nn, 0);
    for (auto& bf : mesh.bottom)
      for (int a = 0; a < 4; ++a) isb[bf.n[a]] = 1;
    for (int n = 0; n < nn; ++n)
      if (isb[n]) bottom_nodes.push_back(n);
    // bottom centre line (y = W/2 or y = 0 for a single element) for the curvature
    double yc = (ny % 2 == 0) ? 0.5 * mesh.p.W : 0.0;
    for (int n : bottom_nodes)
      if (std::abs(mesh.X[n][1] - yc) < 1e-12) centre_line.push_back(n);
  }

  // matrices
  Kb.init(3, mesh.adj_ptr, mesh.adj);
  Kmat.init(3, mesh.adj_ptr, mesh.adj);
  hex_blk.resize(ne);
  for (int e = 0; e < ne; ++e)
    for (int a = 0; a < 8; ++a)
      for (int b = 0; b < 8; ++b) hex_blk[e][a * 8 + b] = Kb.find(mesh.hex[e][a], mesh.hex[e][b]);
  if_blk.resize(ni);
  for (size_t f = 0; f < ni; ++f) {
    int ids[8];
    for (int a = 0; a < 4; ++a) {
      ids[a] = mesh.ifaces[f].a[a];
      ids[a + 4] = mesh.ifaces[f].b[a];
    }
    for (int a = 0; a < 8; ++a)
      for (int b = 0; b < 8; ++b) if_blk[f][a * 8 + b] = Kb.find(ids[a], ids[b]);
  }
  u.assign(size_t(3) * nn, 0.0);
  u_old = u;
  assemble_bulk(0.0);
}

int Model::gp_zone(size_t q, double s) const {
  if (gp_static_zone[q] >= 0) return gp_static_zone[q];
  const int k = gp_row[q];
  const int kind = mesh.ifaces[q / 4].kind;
  if (kind == IF_X || kind == IF_Y) return row_phase(k, s) >= 0.5 ? 1 : 0;
  // horizontal plane between row k (below) and row k + 1 (above)
  const int nrow = int(mesh.z_nodes.size()) - 1;
  double above = (k + 1 < nrow) ? row_phase(k + 1, s) : 1.0, below = row_phase(k, s);
  if (above >= 0.5 && below >= 0.5) return 1;
  if (above >= 0.5) return 2;  // the transformation front (phase above, substrate below)
  return 0;
}

}  // namespace bsf
