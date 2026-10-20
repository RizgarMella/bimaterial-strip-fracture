// Layered-strip cohesive-zone fracture model.
//
// Two loadings (combinable):
//  * differential thermal expansion: uniform temperature history T(t), each
//    layer with its own CTE (strain-free at material.reference_temperature);
//  * surface transformation: the top of the strip converts into a new phase
//    behind a front at depth s(t) = (k t)^(1/n) (n = 3: cubic law). The new
//    phase has its own elastic constants and swells by the Pilling-Bedworth
//    ratio PBR = V_new / V_parent, split between the surface-normal and the
//    in-plane directions ((1 + e_p)^2 (1 + e_n) = PBR, e_p = a e_n).
// Fracture properties come from Weibull distributions on mesh-independent
// random fields, with separate distributions per zone:
//   bimaterial      : zone 0 bottom layer, zone 1 top layer, zone 2 interface
//   transformation  : zone 0 substrate, zone 1 transformed phase, zone 2 front
#pragma once
#include <array>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include "amg.hpp"
#include "config.hpp"
#include "interface_law.hpp"
#include "mesh.hpp"
#include "random_field.hpp"
#include "sparse.hpp"

namespace bsf {

struct Table {
  std::vector<double> t, v;
  double operator()(double x) const;
  void set_constant(double c) { t = {0.0}; v = {c}; }
};
Table read_table(const Config& c, const std::string& key, double def);

struct Elastic {
  double E = 200e9, nu = 0.3, alpha = 10e-6;
  double lambda() const { return E * nu / ((1 + nu) * (1 - 2 * nu)); }
  double mu() const { return E / (2 * (1 + nu)); }
};

// Weibull strength + toughness of one zone
struct ZoneFracture {
  std::string name;
  WeibullSpec strength, toughness;
  double corr_length = 50e-6;   // m; 0 = independent per interface element
  double rho = 0.5;             // toughness-strength correlation
  double reference_area = 0.0;  // weakest-link scaling for corr_length = 0
  uint64_t seed = 1;
  void read(const Config& c, const std::string& section, double s_mean, double g_mean, double m, uint64_t seed0);
};

struct Transformation {
  bool enabled = false;
  double exponent = 3.0;        // s^n = k t   (3 = cubic law, 2 = parabolic, 1 = linear)
  double rate = 0.0;            // k [m^n/s]
  double t_start = 0.0;         // s
  double max_depth = 0.0;       // m (0 = 95 % of the strip thickness)
  double pbr = 1.56;            // Pilling-Bedworth ratio V_new / V_parent
  double anisotropy = 1.0;      // a = e_inplane / e_normal (1 = isotropic swelling)
  Elastic phase;                // elastic constants of the new phase
  double e_normal = 0.0, e_inplane = 0.0;  // derived transformation strains
  void read(const Config& c, double H);
  double depth(double t) const;
};

struct TimeParams {
  double t_end = 1.0, dt0 = 1e-2, dt_min = 1e-10, dt_max = 1e9;
  double grow = 1.5, cut = 0.25;
  double max_load_change = 0.02;      // fraction of the max |T - T_ref| or of the strip thickness per step
  double max_damage_increment = 0.5;
  double softening_dt_factor = 2.0;
  double initiation_target = 1.02;
  int max_steps = 2000000;
  std::vector<double> stops;
};

struct SolverParams {
  int newton_max = 40;
  double newton_tol = 5e-4, newton_dtol = 1e-6, lin_rtol = 1e-3;
  int lin_maxit = 20000;
  double stabilization = 1e-9;
  bool line_search = true, amg = true;
  int amg_rebuild_iters = 40;
};

struct OutputParams {
  std::string dir = "out";
  std::vector<double> times;
  int every = 0;
  double on_crack_growth = 0;
  bool vtu = true, binary = true;
  int max_vtu = 400;
};

struct Diagnostics {
  int newton_iters = 0, lin_iters = 0;
  double res = 0;
};

class Model {
public:
  explicit Model(const Config& cfg);
  void setup();
  void run();

  // loading
  double temperature(double t) const { return temperature_(t); }
  // transformed volume fraction of element row k for front depth s. The fraction is uniform within an
  // element: a swelling strain that varied inside a trilinear element would create spurious stresses
  // (the element cannot follow a strain gradient), while row-wise constant strains are compatible.
  double row_phase(int k, double s) const;
  // mechanics
  void assemble_bulk(double s);
  void compute_eigen_forces(double t, double s);
  bool solve_mechanics(double t, double dt, Diagnostics& dg);
  void interface_forces(double dt, std::vector<double>* F, bool assemble);
  void add_tie_stiffness(BlockCSR& K) const;
  void add_tie_forces(std::vector<double>& F) const;
  void element_state(int e, double s, double sig[6], double& phi) const;  // stress + phase fraction at centre
  int gp_zone(size_t q, double s) const;
  // metrics / outputs
  void compute_metrics();
  int label_fragments(std::vector<int>& lab, std::vector<int>& erid) const;
  void write_manifest() const;
  void open_history();
  void write_history_row(const Diagnostics& dg);
  void write_vtu(int idx);
  void write_pvd() const;
  void write_xplanes(int idx) const;
  void write_summary(bool complete, const std::string& msg) const;

  const Config& cfg;
  std::string mode = "bimaterial";  // bimaterial | transformation
  Mesh mesh;
  std::vector<Elastic> layer_mat;
  double T_ref = 293.15;
  Table temperature_;
  Transformation tr;
  CohesiveParams coh;
  std::array<ZoneFracture, 3> zones;
  std::vector<std::vector<double>> flaws;  // x,y,z,r,strength_factor[,toughness_factor]
  std::string bc = "free";
  // width (y) faces: free (narrow strip, sigma_yy = 0), constrained (u_y = 0: laterally clamped plane
  // strain), generalized (wide free plate: u_y = 0 on y = 0 and u_y = a + b (z - zc)/H on y = W with a, b
  // free, i.e. uniform in-plane strain and curvature across the width)
  std::string y_faces = "free";
  bool plane_strain_y = false;  // true when y_faces != free
  int aux_node = -1;            // generalized: carries a (x dof) and b (y dof)
  std::vector<int> ymax_nodes;
  double tie_penalty = 0, z_centre = 0;
  TimeParams tp;
  SolverParams sp;
  OutputParams op;

  // interface integration points
  size_t ngp = 0;
  std::vector<double> gp_dA;
  std::vector<std::array<double, 9>> gp_R;
  std::vector<std::array<double, 3>> gp_x;
  std::vector<int> gp_row;                   // element row below/at the point (front test)
  std::vector<std::array<GPProps, 3>> gp_props;  // per zone
  std::vector<int8_t> gp_static_zone;        // bimaterial: fixed zone; -1 = decided by the front
  std::vector<GPState> gp_state, gp_trial;
  std::vector<float> gp_open, gp_tn, gp_ts;
  std::vector<int8_t> gp_zone_now;
  int n_gc_clamped = 0;
  double failure_index = 0;
  int n_softening = 0;

  // fields
  std::vector<double> u, u_old, mask_u, elem_vol, elem_hz;
  std::vector<int> pin_dofs, bottom_nodes, centre_line;
  double time = 0, s_front = 0, s_built = -1;
  BlockCSR Kb, Kmat, Kstr;
  AMG amg;
  int last_pcg = 1 << 30, base_pcg = 1, amg_builds = 0;
  std::vector<std::array<int, 64>> hex_blk, if_blk;
  std::vector<double> F_eig, Rres;
  double Fref = 1, Fref_dev = 1;

  // bookkeeping
  int step = 0, n_vtu = 0;
  std::vector<std::pair<double, int>> vtu_list;
  double last_vtu_area = 0, wall_start = 0;
  double onset_time = -1, onset_load = -1;  // load = dT [K] or front depth [m]
  std::FILE* hist = nullptr;
  std::map<std::string, double> last;
  struct Metrics {
    double area[IF_NKINDS] = {0, 0, 0, 0};
    double area_zone[3] = {0, 0, 0};
    double damaged_area = 0, dissipated = 0;
    int n_channel = 0;                // x-planes cracked through the brittle layer
    double mean_spacing = 0;          // m
    double max_crack_depth = 0;       // m below the top surface
    double delaminated_area = 0;      // m2 on the layer interface / front
    double curvature = 0;             // 1/m (bottom centre line, + = concave up)
    double sxx_top = 0, sxx_bot = 0;  // mean sigma_xx in the brittle layer / substrate (mid third)
    double max_principal = 0;
    double pin_reaction = 0;
    int n_fragments = 1;
  } met;
};

std::string build_identifier();

}  // namespace bsf
