// Small-strain elasticity with eigenstrains (thermal mismatch, transformation
// swelling), cohesive interfaces, modified Newton with the symmetric secant
// interface matrix, backtracking line search and AMG-preconditioned CG.
#include <algorithm>
#include <cmath>
#include <iostream>

#include "fe.hpp"
#include "model.hpp"

namespace bsf {

namespace {
inline void atomic_add(double& x, double v) {
#pragma omp atomic
  x += v;
}
}  // namespace

// material state at a Gauss point: elastic constants, thermal expansion,
// phase fraction and the eigenstrain (diagonal)
struct GPMat {
  double lam, mu, phi, eig[3];
};

static GPMat gp_material(const Model& m, int e, double z, double s, double dT) {
  const Elastic& S = m.layer_mat[m.mesh.elem_layer[e]];
  GPMat g;
  (void)z;
  double phi = m.row_phase(m.mesh.elem_ijk[e][2], s);
  g.phi = phi;
  double E = S.E, nu = S.nu, al = S.alpha;
  if (phi > 0) {
    E = (1 - phi) * S.E + phi * m.tr.phase.E;
    nu = (1 - phi) * S.nu + phi * m.tr.phase.nu;
    al = (1 - phi) * S.alpha + phi * m.tr.phase.alpha;
  }
  g.lam = E * nu / ((1 + nu) * (1 - 2 * nu));
  g.mu = E / (2 * (1 + nu));
  g.eig[0] = g.eig[1] = al * dT + phi * m.tr.e_inplane;
  g.eig[2] = al * dT + phi * m.tr.e_normal;
  return g;
}

void Model::assemble_bulk(double s) {
  Kb.zero();
  const int ne = int(mesh.nelem());
#pragma omp parallel for schedule(static)
  for (int e = 0; e < ne; ++e) {
    double X[8][3];
    for (int a = 0; a < 8; ++a)
      for (int i = 0; i < 3; ++i) X[a][i] = mesh.X[mesh.hex[e][a]][i];
    double Ke[8][8][9] = {};
    for (int g = 0; g < 8; ++g) {
      double N[8], B[8][3], dV;
      hex_gp(X, g, N, B, dV);
      double z = 0;
      for (int a = 0; a < 8; ++a) z += N[a] * X[a][2];
      GPMat gm = gp_material(*this, e, z, s, 0.0);
      for (int a = 0; a < 8; ++a)
        for (int b = 0; b < 8; ++b) {
          double dd = B[a][0] * B[b][0] + B[a][1] * B[b][1] + B[a][2] * B[b][2];
          for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
              Ke[a][b][3 * i + j] +=
                  (gm.lam * B[a][i] * B[b][j] + gm.mu * B[a][j] * B[b][i] + (i == j ? gm.mu * dd : 0.0)) * dV;
        }
    }
    for (int a = 0; a < 8; ++a)
      for (int b = 0; b < 8; ++b) {
        double* blk = Kb.block(hex_blk[e][a * 8 + b]);
        for (int k = 0; k < 9; ++k) atomic_add(blk[k], Ke[a][b][k]);
      }
  }
  s_built = s;
}

void Model::compute_eigen_forces(double t, double s) {
  const int nn = int(mesh.nnode()), ne = int(mesh.nelem());
  const double dT = temperature(t) - T_ref;
  F_eig.assign(size_t(3) * nn, 0.0);
  std::vector<double> F_dev(size_t(3) * nn, 0.0);
  // volume-mean eigenstrain removed for the force reference (free expansion carries no stress)
  double em[3] = {0, 0, 0}, vt = 0;
  for (int e = 0; e < ne; ++e) {
    double zc = 0;
    for (int a = 0; a < 8; ++a) zc += mesh.X[mesh.hex[e][a]][2] / 8;
    GPMat gm = gp_material(*this, e, zc, s, dT);
    for (int i = 0; i < 3; ++i) em[i] += gm.eig[i] * elem_vol[e];
    vt += elem_vol[e];
  }
  for (double& v : em) v /= vt;
#pragma omp parallel for schedule(static)
  for (int e = 0; e < ne; ++e) {
    double X[8][3];
    for (int a = 0; a < 8; ++a)
      for (int i = 0; i < 3; ++i) X[a][i] = mesh.X[mesh.hex[e][a]][i];
    double fe[8][3] = {}, fd[8][3] = {};
    for (int g = 0; g < 8; ++g) {
      double N[8], B[8][3], dV;
      hex_gp(X, g, N, B, dV);
      double z = 0;
      for (int a = 0; a < 8; ++a) z += N[a] * X[a][2];
      GPMat gm = gp_material(*this, e, z, s, dT);
      double tr = gm.eig[0] + gm.eig[1] + gm.eig[2];
      double trd = tr - (em[0] + em[1] + em[2]);
      double sig[3], sigd[3];
      for (int i = 0; i < 3; ++i) {
        sig[i] = gm.lam * tr + 2 * gm.mu * gm.eig[i];
        sigd[i] = gm.lam * trd + 2 * gm.mu * (gm.eig[i] - em[i]);
      }
      for (int a = 0; a < 8; ++a)
        for (int i = 0; i < 3; ++i) {
          fe[a][i] += B[a][i] * sig[i] * dV;
          fd[a][i] += B[a][i] * sigd[i] * dV;
        }
    }
    for (int a = 0; a < 8; ++a)
      for (int i = 0; i < 3; ++i) {
        atomic_add(F_eig[3 * mesh.hex[e][a] + i], fe[a][i]);
        atomic_add(F_dev[3 * mesh.hex[e][a] + i], fd[a][i]);
      }
  }
  Fref_dev = norm2(F_dev);
}

void Model::interface_forces(double dt, std::vector<double>* F, bool assemble) {
  const long ni = long(mesh.ifaces.size());
  double fimax = 0.0;
  int nsoft = 0;
  const double s = s_front;
#pragma omp parallel for schedule(dynamic, 64) reduction(max : fimax) reduction(+ : nsoft)
  for (long f = 0; f < ni; ++f) {
    const auto& ifc = mesh.ifaces[f];
    double fA[4][3] = {}, Kloc[4][4][9] = {};
    for (int g = 0; g < 4; ++g) {
      size_t q = size_t(f) * 4 + g;
      double xi = QUAD_SIGN[g][0] * GP1, eta = QUAD_SIGN[g][1] * GP1;
      double N[4];
      for (int a = 0; a < 4; ++a) N[a] = 0.25 * (1 + QUAD_SIGN[a][0] * xi) * (1 + QUAD_SIGN[a][1] * eta);
      double jump[3] = {0, 0, 0};
      for (int a = 0; a < 4; ++a)
        for (int i = 0; i < 3; ++i) jump[i] += N[a] * (u[3 * ifc.b[a] + i] - u[3 * ifc.a[a] + i]);
      const double* R = gp_R[q].data();
      double dl[3];
      for (int k = 0; k < 3; ++k) dl[k] = R[3 * k] * jump[0] + R[3 * k + 1] * jump[1] + R[3 * k + 2] * jump[2];
      const int zn = gp_zone(q, s);
      gp_zone_now[q] = int8_t(zn);
      const GPProps& P = gp_props[q][zn];
      LawOut o;
      cohesive_law(coh, P, gp_state[q], dl, dt, gp_trial[q], o);
      gp_open[q] = float(dl[0]);
      gp_tn[q] = float(o.t[0]);
      gp_ts[q] = float(std::hypot(o.t[1], o.t[2]));
      const auto& trq = gp_trial[q];
      if (trq.d == 0.0) {
        double dnp = std::max(dl[0], 0.0);
        double de = std::sqrt(dnp * dnp + coh.beta * coh.beta * (dl[1] * dl[1] + dl[2] * dl[2]));
        fimax = std::max(fimax, de / delta0(coh, P));
      } else if (trq.d - trq.dv > 0.02 || trq.d - gp_state[q].d > 0.05) {
        ++nsoft;
      }
      const double dA = gp_dA[q];
      double tg[3];
      for (int i = 0; i < 3; ++i) tg[i] = R[i] * o.t[0] + R[3 + i] * o.t[1] + R[6 + i] * o.t[2];
      for (int a = 0; a < 4; ++a)
        for (int i = 0; i < 3; ++i) fA[a][i] += N[a] * tg[i] * dA;
      if (assemble) {
        double Kg[9];
        for (int i = 0; i < 3; ++i)
          for (int j = 0; j < 3; ++j) {
            double v = 0;
            for (int k = 0; k < 3; ++k)
              for (int l = 0; l < 3; ++l) v += R[3 * k + i] * o.D[k][l] * R[3 * l + j];
            Kg[3 * i + j] = v;
          }
        for (int a = 0; a < 4; ++a)
          for (int b = 0; b < 4; ++b) {
            double w = N[a] * N[b] * dA;
            for (int k = 0; k < 9; ++k) Kloc[a][b][k] += w * Kg[k];
          }
      }
    }
    if (F)
      for (int a = 0; a < 4; ++a)
        for (int i = 0; i < 3; ++i) {
          atomic_add((*F)[3 * ifc.b[a] + i], fA[a][i]);
          atomic_add((*F)[3 * ifc.a[a] + i], -fA[a][i]);
        }
    if (assemble)
      for (int a = 0; a < 4; ++a)
        for (int b = 0; b < 4; ++b) {
          double* AA = Kmat.block(if_blk[f][a * 8 + b]);
          double* BB = Kmat.block(if_blk[f][(a + 4) * 8 + b + 4]);
          double* AB = Kmat.block(if_blk[f][a * 8 + b + 4]);
          double* BA = Kmat.block(if_blk[f][(a + 4) * 8 + b]);
          for (int k = 0; k < 9; ++k) {
            double v = Kloc[a][b][k];
            atomic_add(AA[k], v);
            atomic_add(BB[k], v);
            atomic_add(AB[k], -v);
            atomic_add(BA[k], -v);
          }
        }
  }
  failure_index = fimax;
  n_softening = nsoft;
}

bool Model::solve_mechanics(double t, double dt, Diagnostics& dg) {
  const int nn = int(mesh.nnode());
  const long N = 3L * nn;
  for (long i = 0; i < N; ++i)
    if (mask_u[i] == 0.0) u[i] = 0.0;
  if (tr.enabled && std::abs(s_front - s_built) > 1e-9 * mesh.z_nodes.back()) assemble_bulk(s_front);
  compute_eigen_forces(t, s_front);
  // force scale: the stress-producing part of the eigenstrain loads, but never below a small
  // fraction of the full eigen-force (uniform swelling produces no stress at all)
  Fref = std::max(Fref_dev, 1e-3 * norm2(F_eig));
  if (Fref <= 0) Fref = 1.0;
  double diag_mean = 0;
  for (int n = 0; n < nn; ++n) {
    const double* b = &Kb.val[size_t(Kb.diag[n]) * 9];
    diag_mean += b[0] + b[4] + b[8];
  }
  diag_mean /= 3.0 * nn;
  std::vector<double> R(N), Fi(N), du(N), u0(N);
  auto residual = [&](bool assemble) {
    Kb.matvec(u, R);
    std::fill(Fi.begin(), Fi.end(), 0.0);
    if (assemble) Kmat.val = Kb.val;
    interface_forces(dt, &Fi, assemble);
    double s2 = 0;
#pragma omp parallel for reduction(+ : s2) schedule(static)
    for (long i = 0; i < N; ++i) {
      R[i] = F_eig[i] - R[i] - Fi[i];
      s2 += R[i] * R[i] * mask_u[i];
    }
    return std::sqrt(s2);
  };
  dg.newton_iters = dg.lin_iters = 0;
  double rn = residual(true);
  std::vector<double> rhist;
  for (int it = 0; it < sp.newton_max; ++it) {
    dg.res = rn / Fref;
    if (!std::isfinite(rn)) return false;
    rhist.push_back(rn);
    if (it >= 12 && rn > 0.5 * rhist[it - 8]) return false;
    if (rn <= 1e-3 * sp.newton_tol * Fref) {
      Rres = R;
      return true;
    }
    if (sp.stabilization > 0) Kmat.add_diagonal(sp.stabilization * diag_mean);
    std::fill(du.begin(), du.end(), 0.0);
    const bool rebuild = !amg.ready() || (last_pcg > sp.amg_rebuild_iters && last_pcg > 2 * base_pcg);
    if (sp.amg && rebuild) {
      amg.setup(Kmat, mask_u, &mesh.X);
      ++amg_builds;
    }
    auto lr = pcg(Kmat, R, du, mask_u, sp.lin_rtol, sp.lin_maxit, sp.amg ? &amg : nullptr);
    last_pcg = lr.iters;
    if (rebuild) base_pcg = std::max(lr.iters, 1);
    dg.lin_iters += lr.iters;
    ++dg.newton_iters;
    if (!lr.converged && lr.rel_res > 1e-2) return false;
    u0 = u;
#pragma omp parallel for schedule(static)
    for (long i = 0; i < N; ++i) u[i] = u0[i] + du[i];
    double rfull = residual(true), abest = 1.0;
    if (sp.line_search && !(rfull < rn)) {
      double rbest = rfull, alpha = 1.0;
      for (int ls = 0; ls < 3; ++ls) {
        alpha *= 0.5;
#pragma omp parallel for schedule(static)
        for (long i = 0; i < N; ++i) u[i] = u0[i] + alpha * du[i];
        double r = residual(false);
        if (r < rbest) {
          rbest = r;
          abest = alpha;
        }
        if (r < rn) break;
      }
#pragma omp parallel for schedule(static)
      for (long i = 0; i < N; ++i) u[i] = u0[i] + abest * du[i];
      rfull = residual(true);
    }
    double dun = norm_inf(du) * abest, umax = norm_inf(u);
    rn = rfull;
    if (rn <= sp.newton_tol * Fref || (dun <= sp.newton_dtol * std::max(umax, 1e-15) && rn <= 10 * sp.newton_tol * Fref)) {
      dg.res = rn / Fref;
      Rres = R;
      return true;
    }
  }
  dg.res = rn / Fref;
  return false;
}

void Model::element_state(int e, double s, double sig[6], double& phi) const {
  double X[8][3], ue[8][3];
  for (int a = 0; a < 8; ++a) {
    int n = mesh.hex[e][a];
    for (int i = 0; i < 3; ++i) {
      X[a][i] = mesh.X[n][i];
      ue[a][i] = u[3 * n + i];
    }
  }
  const double dT = temperature(time) - T_ref;
  for (int k = 0; k < 6; ++k) sig[k] = 0;
  phi = 0;
  double vol = 0;
  for (int g = 0; g < 8; ++g) {
    double N[8], B[8][3], dV;
    hex_gp(X, g, N, B, dV);
    double z = 0, H[3][3] = {};
    for (int a = 0; a < 8; ++a) {
      z += N[a] * X[a][2];
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) H[i][j] += ue[a][i] * B[a][j];
    }
    GPMat gm = gp_material(*this, e, z, s, dT);
    double ex = H[0][0] - gm.eig[0], ey = H[1][1] - gm.eig[1], ez = H[2][2] - gm.eig[2];
    double trc = ex + ey + ez;
    sig[0] += (gm.lam * trc + 2 * gm.mu * ex) * dV;
    sig[1] += (gm.lam * trc + 2 * gm.mu * ey) * dV;
    sig[2] += (gm.lam * trc + 2 * gm.mu * ez) * dV;
    sig[3] += gm.mu * (H[0][1] + H[1][0]) * dV;
    sig[4] += gm.mu * (H[1][2] + H[2][1]) * dV;
    sig[5] += gm.mu * (H[0][2] + H[2][0]) * dV;
    phi += gm.phi * dV;
    vol += dV;
  }
  for (int k = 0; k < 6; ++k) sig[k] /= vol;
  phi /= vol;
}

}  // namespace bsf
