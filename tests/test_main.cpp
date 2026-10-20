// Verification tests (ctest). Each check prints PASS/FAIL.
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <string>

#include "config.hpp"
#include "interface_law.hpp"
#include "model.hpp"

using namespace bsf;

static int failures = 0;
static void check(bool ok, const std::string& what, double got, double want) {
  std::printf("%s  %-66s got %-12.6g expected %-12.6g\n", ok ? "PASS" : "FAIL", what.c_str(), got, want);
  if (!ok) ++failures;
}
static bool rel(double a, double b, double tol) { return std::abs(a - b) <= tol * std::abs(b); }

static void test_cohesive_law() {
  CohesiveParams P;
  P.K = 1e17;
  P.viscosity = 0;
  GPProps g{150e6, 10.0, 0.0};
  GPState s, tr;
  LawOut o;
  double work = 0, dprev = 0, tprev = 0, peak = 0;
  const double df = deltaf(P, g);
  for (int i = 1; i <= 4000; ++i) {
    double d = 1.2 * df * i / 4000.0, dl[3] = {d, 0, 0};
    cohesive_law(P, g, s, dl, 1.0, tr, o);
    s = tr;
    peak = std::max(peak, o.t[0]);
    work += 0.5 * (o.t[0] + tprev) * (d - dprev);
    dprev = d;
    tprev = o.t[0];
  }
  check(rel(peak, 150e6, 0.002), "cohesive law: peak traction = sigma_t", peak, 150e6);
  check(rel(work, 10.0, 0.005), "cohesive law: work of separation = Gc", work, 10.0);
}

static void test_transformation_strains() {
  Config c;
  c.parse_text("[transformation]\npilling_bedworth_ratio = 1.56\nanisotropy = 1\n");
  Transformation t;
  t.read(c, 1e-3);
  check(rel(t.e_normal, std::cbrt(1.56) - 1, 1e-12), "PBR isotropic: e = PBR^(1/3) - 1", t.e_normal, std::cbrt(1.56) - 1);
  Config c2;
  c2.parse_text("[transformation]\npilling_bedworth_ratio = 1.56\nanisotropy = 0.02\ndepth_at_time = 100e-6, 1e4\n");
  Transformation t2;
  t2.read(c2, 1e-3);
  double vr = (1 + t2.e_inplane) * (1 + t2.e_inplane) * (1 + t2.e_normal);
  check(rel(vr, 1.56, 1e-12) && rel(t2.e_inplane, 0.02 * t2.e_normal, 1e-12), "PBR anisotropic: volume ratio and split",
        vr, 1.56);
  check(rel(t2.depth(1e4), 100e-6, 1e-12) && rel(t2.depth(8e4), 200e-6, 1e-12), "cubic law s = (k t)^(1/3)",
        t2.depth(8e4), 200e-6);
}

static std::string strip_base(const std::string& extra) {
  return std::string(R"(
[cohesive]
stiffness = 1e18
[output]
vtu = false
[time]
end = 1
dt_initial = 1
dt_max = 1
)") + extra;
}

// Timoshenko (1925) bimetal curvature, plane strain across the width:
// kappa = 6 (1+m)^2 de / (h [3 (1+m)^2 + (1 + m n)(m^2 + 1/(m n))]), with
// E' = E/(1 - nu^2) and alpha' = (1 + nu) alpha.
static void test_bimetal_curvature() {
  namespace fs = std::filesystem;
  std::string dir = (fs::temp_directory_path() / "stripcz_test_bimetal").string();
  Config c;
  c.parse_text(strip_base(R"(
[model]
type = bimaterial
[geometry]
length = 20e-3
width = 0.2e-3
n_layers = 2
[mesh]
nx = 200
ny = 1
[layer0]
thickness = 0.5e-3
nz = 8
youngs_modulus = 200e9
poisson_ratio = 0.3
thermal_expansion = 17e-6
x_cracks = none
[layer1]
thickness = 0.3e-3
nz = 6
youngs_modulus = 380e9
poisson_ratio = 0.3
thermal_expansion = 8e-6
x_cracks = none
[fracture.interface]
strength.distribution = constant
strength.mean = 1e12
toughness.distribution = constant
toughness.mean = 1e6
[loading]
temperature_history = 0, 293.15; 1, 393.15
[boundary]
mode = free
plane_strain_y = true
[output]
directory = )") + dir + "\n");
  Model m(c);
  m.run();
  const double nu = 0.3, dT = 100;
  double t1 = 0.3e-3, t2 = 0.5e-3, E1 = 380e9 / (1 - nu * nu), E2 = 200e9 / (1 - nu * nu);
  double de = (1 + nu) * (17e-6 - 8e-6) * dT, mm = t1 / t2, n = E1 / E2, h = t1 + t2;
  double kap = 6 * (1 + mm) * (1 + mm) * de / (h * (3 * (1 + mm) * (1 + mm) + (1 + mm * n) * (mm * mm + 1 / (mm * n))));
  check(rel(std::abs(m.met.curvature), kap, 0.03), "bimetal curvature vs Timoshenko (plane strain)",
        std::abs(m.met.curvature), kap);
  check(m.met.curvature > 0, "strip bends towards the low-expansion top layer", m.met.curvature, kap);
  check(m.met.pin_reaction < 1e-2, "free strip: no pin reactions (< 0.01 N)", m.met.pin_reaction, 0.0);
  fs::remove_all(dir);
}

// A fully transformed free block expands by the transformation strains
// without stress: checks the eigenstrain path and the phase elastic constants.
static void test_free_swelling() {
  namespace fs = std::filesystem;
  std::string dir = (fs::temp_directory_path() / "stripcz_test_swell").string();
  Config c;
  c.parse_text(strip_base(R"(
[model]
type = transformation
[geometry]
length = 1e-3
width = 0.2e-3
n_layers = 1
[mesh]
nx = 10
ny = 2
[layer0]
thickness = 0.5e-3
nz = 5
x_cracks = none
[transformation]
pilling_bedworth_ratio = 1.3
anisotropy = 0.5
rate = 1e-9
max_depth = 0.6e-3
[boundary]
mode = free
[output]
directory = )") + dir + "\n");
  Model m(c);
  m.run();
  // displacement of the top far corner relative to the pinned origin
  double ux = 0, uz = 0;
  for (size_t n = 0; n < m.mesh.nnode(); ++n)
    if (std::abs(m.mesh.X[n][0] - 1e-3) < 1e-12 && std::abs(m.mesh.X[n][2] - 0.5e-3) < 1e-12) {
      ux = m.u[3 * n];
      uz = m.u[3 * n + 2];
    }
  check(rel(ux, m.tr.e_inplane * 1e-3, 1e-3), "free swelling: in-plane extension = e_p L", ux, m.tr.e_inplane * 1e-3);
  check(rel(uz, m.tr.e_normal * 0.5e-3, 1e-3), "free swelling: normal extension = e_n H", uz, m.tr.e_normal * 0.5e-3);
  check(std::abs(m.met.max_principal) < 1e3, "free swelling: stress free", m.met.max_principal, 0.0);
  fs::remove_all(dir);
}

static void test_zones() {
  Config c;
  c.parse_text(strip_base(R"(
[model]
type = transformation
[geometry]
length = 1e-3
width = 0.1e-3
n_layers = 1
[mesh]
nx = 4
ny = 1
[layer0]
thickness = 1e-3
nz = 10
x_cracks = all
z_cracks = all
[transformation]
rate = 1
)"));
  Model m(c);
  m.setup();
  const double s = 0.37e-3;  // front at z = 0.63 mm, inside the element row 0.6 .. 0.7 mm
  int bad = 0, n2 = 0;
  for (size_t q = 0; q < m.ngp; ++q) {
    int z = m.gp_zone(q, s);
    double zz = m.gp_x[q][2];
    bool horiz = std::abs(m.gp_R[q][2]) > 0.9;
    // row 0.6-0.7 is 70 % transformed (>= 0.5): all of it counts as phase; the front plane is z = 0.6
    int expect = horiz ? (zz > 0.65e-3 ? 1 : (std::abs(zz - 0.6e-3) < 1e-9 ? 2 : 0)) : (zz > 0.6e-3 ? 1 : 0);
    if (horiz && std::abs(zz - 0.6e-3) < 1e-9) ++n2;
    if (z != expect) ++bad;
  }
  check(bad == 0 && n2 > 0, "zones: phase above the front, front plane, substrate below", bad, 0);
  int counts[3] = {0, 0, 0};
  for (size_t q = 0; q < m.ngp; ++q) counts[m.gp_zone(q, 0.0)]++;
  check(counts[1] == 0 && counts[2] == 0, "zones: everything is substrate before transformation", counts[1] + counts[2], 0);
}

static void test_weibull_zones() {
  Config c;
  c.parse_text(strip_base(R"(
[model]
type = bimaterial
[geometry]
length = 4e-3
width = 0.4e-3
n_layers = 2
[mesh]
nx = 80
ny = 8
[layer0]
thickness = 0.4e-3
nz = 8
x_cracks = all
[layer1]
thickness = 0.4e-3
nz = 8
x_cracks = all
[fracture.top]
strength.mean = 150e6
strength.weibull_modulus = 6
correlation_length = 0
[fracture.bottom]
strength.mean = 400e6
strength.weibull_modulus = 15
correlation_length = 0
)"));
  Model m(c);
  m.setup();
  double s[3] = {0, 0, 0};
  int n[3] = {0, 0, 0};
  for (size_t q = 0; q < m.ngp; ++q) {
    int z = m.gp_static_zone[q];
    s[z] += m.gp_props[q][z].sigma_t;
    n[z]++;
  }
  check(rel(s[1] / n[1], 150e6, 0.03), "Weibull per zone: top-layer mean strength", s[1] / n[1], 150e6);
  check(rel(s[0] / n[0], 400e6, 0.03), "Weibull per zone: substrate mean strength", s[0] / n[0], 400e6);
  check(n[2] == 4 * 80 * 8, "interface points = layer interface faces x 4", n[2], 4 * 80 * 8);
}

int main() {
  test_cohesive_law();
  test_transformation_strains();
  test_zones();
  test_weibull_zones();
  test_free_swelling();
  test_bimetal_curvature();
  std::printf("\n%d failure(s)\n", failures);
  return failures ? 1 : 0;
}
