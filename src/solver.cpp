// Time stepping (quasi-static with viscous regularisation, as in pelletcz):
// adaptive steps, rejection on Newton failure or large damage jumps, steps on
// the viscous time scale while cracks run, failure-index driven approach of
// the next initiation, and limited load change per step (temperature change
// or front advance).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>

#include "fe.hpp"
#include "model.hpp"

namespace bsf {

static double wall_now() {
  using namespace std::chrono;
  return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int Model::label_fragments(std::vector<int>& lab, std::vector<int>& erid) const {
  const int ne = int(mesh.nelem());
  std::map<std::array<int, 3>, int> rid;
  erid.resize(ne);
  for (int e = 0; e < ne; ++e) {
    auto it = rid.find(mesh.elem_region[e]);
    if (it == rid.end()) it = rid.emplace(mesh.elem_region[e], int(rid.size())).first;
    erid[e] = it->second;
  }
  std::vector<int> node_rid(mesh.nnode(), -1);
  for (int e = 0; e < ne; ++e)
    for (int a = 0; a < 8; ++a) node_rid[mesh.hex[e][a]] = erid[e];
  std::vector<int> parent(rid.size());
  std::iota(parent.begin(), parent.end(), 0);
  std::function<int(int)> find = [&](int x) { return parent[x] == x ? x : parent[x] = find(parent[x]); };
  for (size_t f = 0; f < mesh.ifaces.size(); ++f) {
    bool intact = false;
    for (int g = 0; g < 4; ++g)
      if (gp_state[4 * f + g].d < coh.d_crack) intact = true;
    if (!intact) continue;
    int a = find(node_rid[mesh.ifaces[f].a[0]]), b = find(node_rid[mesh.ifaces[f].b[0]]);
    if (a != b) parent[a] = b;
  }
  std::map<int, int> roots;
  lab.resize(parent.size());
  for (size_t r = 0; r < parent.size(); ++r) {
    int root = find(int(r));
    auto it = roots.find(root);
    if (it == roots.end()) it = roots.emplace(root, int(roots.size())).first;
    lab[r] = it->second;
  }
  return int(roots.size());
}

void Model::compute_metrics() {
  Metrics m;
  const double H = mesh.z_nodes.back();
  std::map<int, std::array<double, 2>> xpl;  // brittle area, cracked brittle area
  for (size_t f = 0; f < mesh.ifaces.size(); ++f) {
    const auto& ifc = mesh.ifaces[f];
    for (int g = 0; g < 4; ++g) {
      size_t q = 4 * f + g;
      const auto& st = gp_state[q];
      const int zn = gp_zone_now[q];
      const double dA = gp_dA[q];
      bool cracked = st.d >= coh.d_crack;
      if (cracked) {
        m.area[ifc.kind] += dA;
        m.area_zone[zn] += dA;
        if (ifc.kind == IF_Z || ifc.kind == IF_LAYER) m.delaminated_area += dA;
        if (ifc.kind == IF_X || ifc.kind == IF_Y) m.max_crack_depth = std::max(m.max_crack_depth, H - gp_x[q][2]);
      }
      if (st.d > 0) m.damaged_area += dA;
      m.dissipated += dA * dissipated_energy(coh, gp_props[q][zn], st.dmax);
      if (ifc.kind == IF_X && zn == 1) {
        auto& r = xpl[ifc.plane];
        r[0] += dA;
        if (cracked) r[1] += dA;
      }
    }
  }
  for (auto& [p, r] : xpl)
    if (r[0] > 0 && r[1] >= 0.9 * r[0]) ++m.n_channel;
  m.mean_spacing = mesh.p.L / (m.n_channel + 1);
  // curvature of the bottom centre line: least-squares uz = a x^2 + b x + c
  {
    double S[5] = {0, 0, 0, 0, 0}, T3[3] = {0, 0, 0};
    for (int n : centre_line) {
      double x = mesh.X[n][0], w = u[3 * n + 2], xp = 1;  // small strain: reference x
      for (int k = 0; k < 5; ++k) {
        S[k] += xp;
        if (k < 3) T3[k] += xp * w;
        xp *= x;
      }
    }
    // normal equations [S4 S3 S2; S3 S2 S1; S2 S1 S0] [a b c] = [T2 T1 T0]
    double A[3][3] = {{S[4], S[3], S[2]}, {S[3], S[2], S[1]}, {S[2], S[1], S[0]}}, b[3] = {T3[2], T3[1], T3[0]};
    double det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0]) +
                 A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
    if (std::abs(det) > 0 && centre_line.size() >= 3) {
      double da = b[0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1]) - A[0][1] * (b[1] * A[2][2] - A[1][2] * b[2]) +
                  A[0][2] * (b[1] * A[2][1] - A[1][1] * b[2]);
      m.curvature = 2.0 * da / det;
    }
  }
  // mean sigma_xx in the middle third, brittle layer vs substrate; max principal
  const int ne = int(mesh.nelem());
  const int top_layer = int(mesh.p.layers.size()) - 1;
  double sb = 0, vb = 0, ss = 0, vs = 0, mp = -1e300;
#pragma omp parallel for reduction(+ : sb, vb, ss, vs) reduction(max : mp)
  for (int e = 0; e < ne; ++e) {
    double sg[6], ev[3], ph;
    element_state(e, s_front, sg, ph);
    sym_eig3(sg, ev);
    mp = std::max(mp, ev[0]);
    double xc = 0;
    for (int a = 0; a < 8; ++a) xc += mesh.X[mesh.hex[e][a]][0] / 8;
    if (xc < mesh.p.L / 3 || xc > 2 * mesh.p.L / 3) continue;
    bool brittle = (mode == "bimaterial") ? (mesh.elem_layer[e] == top_layer && top_layer > 0) : (ph >= 0.5);
    if (brittle) {
      sb += sg[0] * elem_vol[e];
      vb += elem_vol[e];
    } else {
      ss += sg[0] * elem_vol[e];
      vs += elem_vol[e];
    }
  }
  m.sxx_top = vb > 0 ? sb / vb : 0;
  m.sxx_bot = vs > 0 ? ss / vs : 0;
  m.max_principal = mp;
  if (!Rres.empty() && !pin_dofs.empty()) {
    double r[3] = {0, 0, 0};
    for (int d : pin_dofs) r[d % 3] += -Rres[d];
    m.pin_reaction = std::sqrt(r[0] * r[0] + r[1] * r[1] + r[2] * r[2]);
  }
  std::vector<int> lab, erid;
  m.n_fragments = label_fragments(lab, erid);
  met = m;
}

void Model::run() {
  namespace fs = std::filesystem;
  wall_start = wall_now();
  fs::create_directories(op.dir);
  setup();
  write_manifest();
  open_history();
  Diagnostics dg;
  s_front = tr.depth(0.0);
  solve_mechanics(0.0, 0.0, dg);
  u_old = u;
  gp_state = gp_trial;
  compute_metrics();
  write_history_row(dg);
  if (op.vtu) write_vtu(0);
  vtu_list.push_back({0.0, 0});
  write_xplanes(0);

  double dTmax = 0;
  for (double v : temperature_.v) dTmax = std::max(dTmax, std::abs(v - T_ref));
  double hzmin = 1e300;
  for (size_t k = 1; k < mesh.z_nodes.size(); ++k) hzmin = std::min(hzmin, mesh.z_nodes[k] - mesh.z_nodes[k - 1]);
  const double H = mesh.z_nodes.back();
  const double ds_max = std::min(tp.max_load_change * H, 0.5 * hzmin);
  const double eps_t = 1e-9 * std::max(1.0, tp.t_end);
  double dt = tp.dt0, fi_prev = 0, t_prev = 0, last_print = wall_now();
  size_t stop_i = 0;
  int rejections = 0;
  bool complete = true;
  std::string end_msg = "completed";
  while (time < tp.t_end - eps_t) {
    if (step >= tp.max_steps) {
      complete = false;
      end_msg = "time.max_steps reached";
      break;
    }
    while (stop_i < tp.stops.size() && tp.stops[stop_i] <= time + eps_t) ++stop_i;
    double t_stop = stop_i < tp.stops.size() ? tp.stops[stop_i] : tp.t_end;
    dt = std::min({dt, tp.dt_max, t_stop - time});
    for (int k = 0; k < 60; ++k) {  // limit the load change per step
      bool okT = dTmax <= 0 || std::abs(temperature(time + dt) - temperature(time)) <= tp.max_load_change * dTmax;
      bool okS = !tr.enabled || tr.depth(time + dt) - tr.depth(time) <= ds_max;
      if ((okT && okS) || dt <= tp.dt_min) break;
      dt *= 0.5;
    }
    const double t1 = (t_stop - (time + dt) < 1e-6 * dt) ? t_stop : time + dt;
    dt = t1 - time;
    const double s_old = s_front;
    s_front = tr.depth(t1);
    gp_trial = gp_state;
    bool ok = solve_mechanics(t1, dt, dg);
    std::string why = ok ? "" : "newton";
    double maxdd = 0;
    if (ok) {
#pragma omp parallel for reduction(max : maxdd)
      for (long q = 0; q < long(ngp); ++q) maxdd = std::max(maxdd, gp_trial[q].dv - gp_state[q].dv);
      if (maxdd > tp.max_damage_increment && dt > 4 * tp.dt_min) {
        ok = false;
        why = "damage";
      }
    }
    if (!ok) {
      u = u_old;
      gp_trial = gp_state;
      s_front = s_old;
      dt *= tp.cut;
      if (n_softening > 0 && coh.viscosity > 0) dt = std::min(dt, 4.0 * tp.softening_dt_factor * coh.viscosity);
      ++rejections;
      if (dt < tp.dt_min) {
        complete = false;
        end_msg = "time step below time.dt_min at t = " + std::to_string(time) + " s (" + why + ")";
        std::cerr << end_msg << std::endl;
        break;
      }
      continue;
    }
    u_old = u;
    gp_state = gp_trial;
    time = t1;
    ++step;
    compute_metrics();
    double area = met.area[0] + met.area[1] + met.area[2] + met.area[3];
    if (onset_time < 0 && area > 0) {
      onset_time = time;
      onset_load = (mode == "bimaterial") ? temperature(time) - T_ref : s_front;
      std::cout << "  first crack at t = " << time << " s"
                << (mode == "bimaterial" ? ", dT = " + std::to_string(onset_load) + " K"
                                         : ", front depth = " + std::to_string(onset_load * 1e6) + " um")
                << std::endl;
    }
    write_history_row(dg);
    bool out = false;
    for (double to : op.times)
      if (std::abs(to - time) <= eps_t) out = true;
    if (op.every > 0 && step % op.every == 0) out = true;
    if (op.on_crack_growth > 0 && area - last_vtu_area >= op.on_crack_growth) out = true;
    if (time >= tp.t_end - eps_t) out = true;
    if (out && n_vtu < op.max_vtu) {
      ++n_vtu;
      if (op.vtu) write_vtu(n_vtu);
      vtu_list.push_back({time, n_vtu});
      write_pvd();
      write_xplanes(n_vtu);
      last_vtu_area = area;
    }
    if (wall_now() - last_print > 5.0 || out) {
      last_print = wall_now();
      std::cout << std::setprecision(4) << "step " << step << "  t=" << time << " s  dt=" << dt
                << "  dT=" << temperature(time) - T_ref << " K  front=" << s_front * 1e6 << " um  cracks="
                << area * 1e6 << " mm2  channels=" << met.n_channel << "  delam=" << met.delaminated_area * 1e6
                << " mm2  FI=" << failure_index << " soft=" << n_softening << "  newton=" << dg.newton_iters
                << " pcg=" << dg.lin_iters << std::endl;
    }
    if (dg.newton_iters <= 4 && maxdd < 0.5 * tp.max_damage_increment)
      dt *= tp.grow;
    else if (dg.newton_iters > sp.newton_max / 2)
      dt *= 0.7;
    const double dt_visc = tp.softening_dt_factor * std::max(coh.viscosity, tp.dt_min);
    if (n_softening > 0) {
      dt = std::min(dt, dt_visc);
    } else if (failure_index > 0.3 && failure_index > fi_prev && fi_prev > 0) {
      double rate = (failure_index - fi_prev) / std::max(time - t_prev, 1e-30);
      double dt_pred = (tp.initiation_target - failure_index) / rate;
      dt = std::min(dt, std::max({dt_pred, 0.25 * (time - t_prev), dt_visc}));
    }
    fi_prev = failure_index;
    t_prev = time;
  }
  std::cout << "finished: " << end_msg << " (" << step << " steps, " << rejections << " rejected, "
            << wall_now() - wall_start << " s wall, " << amg_builds << " AMG builds)" << std::endl;
  write_pvd();
  write_summary(complete, end_msg);
  if (hist) std::fclose(hist);
  hist = nullptr;
}

}  // namespace bsf
