// Structured 3D hexahedral mesh of a layered strip (length L along x, width
// W along y, thickness along z) with zero-thickness interface elements on the
// potential crack planes:
//   * x-planes  (normal x): channel cracks across the width,
//   * y-planes  (normal y): longitudinal cracks (mud-crack networks),
//   * z-planes  (normal z): delamination / spallation planes inside a layer,
//   * the material interface between layers (always cohesive).
// Crack planes are enabled per layer. As in pelletcz, every element gets a
// region tuple, nodes are duplicated once per region and every face shared by
// two regions becomes a 4+4 node interface element.
#pragma once
#include <array>
#include <string>
#include <vector>

namespace bsf {

class Config;

enum IfaceKind { IF_X = 0, IF_Y = 1, IF_Z = 2, IF_LAYER = 3, IF_NKINDS = 4 };
const char* iface_kind_name(int k);

struct LayerSpec {
  double thickness = 0.5e-3;
  int nz = 10;
  int x_every = 1;   // 0: no x-planes; k: a plane every k elements
  int y_every = 0;
  int z_every = 0;   // horizontal planes inside the layer every k element rows
  double grading = 1.0;  // >1 clusters elements towards the top of the layer
};

struct StripParams {
  double L = 4e-3, W = 0.25e-3;
  int nx = 80, ny = 1;
  std::vector<LayerSpec> layers;  // bottom -> top
  void read(const Config& c);
  double total_thickness() const;
};

struct Iface {
  std::array<int, 4> a, b;  // side A (lower coordinate) and side B node copies
  int kind;                 // IfaceKind
  int plane;                // node index of the plane along its normal
  int layer;                // layer of side A
};

struct BFace {
  std::array<int, 4> n;
  int elem;
};

struct Mesh {
  StripParams p;
  std::vector<std::array<double, 3>> X;
  std::vector<std::array<int, 8>> hex;
  std::vector<int> elem_layer;
  std::vector<std::array<int, 3>> elem_ijk;
  std::vector<std::array<int, 3>> elem_region;  // zseg, xseg, yseg
  std::vector<Iface> ifaces;
  std::vector<BFace> bottom, top, xmin, xmax, ymin, ymax;
  std::vector<double> x_nodes, y_nodes, z_nodes;
  std::vector<int> layer_k0;  // first element row of each layer (+ total at the end)
  int n_orig_nodes = 0;
  std::vector<int> adj_ptr, adj;

  size_t nnode() const { return X.size(); }
  size_t nelem() const { return hex.size(); }
  void build(const StripParams& sp);
  void build_graph();
  std::string summary() const;
};

extern const int HEX_FACES[6][4];

}  // namespace bsf
