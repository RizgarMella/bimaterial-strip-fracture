#include "mesh.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "config.hpp"

namespace bsf {

const int HEX_FACES[6][4] = {
    {0, 3, 2, 1},  // z-
    {4, 5, 6, 7},  // z+
    {0, 1, 5, 4},  // y-
    {1, 2, 6, 5},  // x+
    {2, 3, 7, 6},  // y+
    {3, 0, 4, 7},  // x-
};

const char* iface_kind_name(int k) {
  switch (k) {
    case IF_X: return "x_plane";
    case IF_Y: return "y_plane";
    case IF_Z: return "z_plane";
    case IF_LAYER: return "layer_interface";
  }
  return "?";
}

static int parse_every(const std::string& s) {
  if (s == "none" || s == "0" || s.empty()) return 0;
  if (s == "all") return 1;
  int v = std::stoi(s);
  if (v < 0) throw std::runtime_error("crack plane spacing must be >= 0");
  return v;
}

double StripParams::total_thickness() const {
  double h = 0;
  for (auto& l : layers) h += l.thickness;
  return h;
}

void StripParams::read(const Config& c) {
  L = c.get_double("geometry.length", L);
  W = c.get_double("geometry.width", W);
  nx = c.get_int("mesh.nx", nx);
  ny = c.get_int("mesh.ny", ny);
  int nl = c.get_int("geometry.n_layers", 2);
  if (nl < 1 || nl > 8) throw std::runtime_error("geometry.n_layers must be 1..8");
  layers.assign(nl, LayerSpec{});
  for (int l = 0; l < nl; ++l) {
    std::string s = "layer" + std::to_string(l);
    auto& L_ = layers[l];
    L_.thickness = c.get_double(s + ".thickness", L_.thickness);
    L_.nz = c.get_int(s + ".nz", L_.nz);
    L_.x_every = parse_every(c.get_string(s + ".x_cracks", "all"));
    L_.y_every = parse_every(c.get_string(s + ".y_cracks", "none"));
    L_.z_every = parse_every(c.get_string(s + ".z_cracks", "none"));
    L_.grading = c.get_double(s + ".grading", 1.0);
  }
  // convenience for the thickness-ratio series: fixed total thickness and element count
  if (nl == 2 && c.has("geometry.top_fraction")) {
    double h = c.get_double("geometry.total_thickness", total_thickness());
    double f = c.get_double("geometry.top_fraction", 0.2);
    int nzt = c.get_int("mesh.nz_total", layers[0].nz + layers[1].nz);
    if (f <= 0 || f >= 1) throw std::runtime_error("geometry.top_fraction must be in (0, 1)");
    layers[1].thickness = f * h;
    layers[0].thickness = (1 - f) * h;
    layers[1].nz = std::max(1, int(std::lround(f * nzt)));
    layers[0].nz = std::max(1, nzt - layers[1].nz);
  }
}

namespace {
struct ArrHash {
  size_t operator()(const std::array<int, 4>& a) const {
    size_t h = 1469598103934665603ull;
    for (int v : a) h = (h ^ size_t(v)) * 1099511628211ull;
    return h;
  }
};
}  // namespace

void Mesh::build(const StripParams& sp) {
  p = sp;
  const int nx = p.nx, ny = p.ny;
  if (nx < 1 || ny < 1 || p.layers.empty()) throw std::runtime_error("mesh: bad resolution");
  x_nodes.resize(nx + 1);
  y_nodes.resize(ny + 1);
  for (int i = 0; i <= nx; ++i) x_nodes[i] = p.L * i / nx;
  for (int j = 0; j <= ny; ++j) y_nodes[j] = p.W * j / ny;
  z_nodes = {0.0};
  layer_k0.clear();
  for (auto& L_ : p.layers) {
    layer_k0.push_back(int(z_nodes.size()) - 1);
    double z0 = z_nodes.back();
    for (int k = 1; k <= L_.nz; ++k) {
      double s = double(k) / L_.nz;
      // grading > 1 clusters nodes towards the top of the layer
      double g = 1.0 - std::pow(1.0 - s, L_.grading);
      z_nodes.push_back(z0 + L_.thickness * g);
    }
  }
  layer_k0.push_back(int(z_nodes.size()) - 1);
  const int nz = int(z_nodes.size()) - 1;

  // regions
  const int ne = nx * ny * nz;
  hex.assign(ne, {});
  elem_layer.assign(ne, 0);
  elem_ijk.assign(ne, {});
  elem_region.assign(ne, {});
  std::vector<std::array<int, 8>> hex_o(ne);
  auto onode = [&](int i, int j, int k) { return (k * (ny + 1) + j) * (nx + 1) + i; };
  n_orig_nodes = (nx + 1) * (ny + 1) * (nz + 1);
  std::vector<std::array<double, 3>> Xo(n_orig_nodes);
  for (int k = 0; k <= nz; ++k)
    for (int j = 0; j <= ny; ++j)
      for (int i = 0; i <= nx; ++i) Xo[onode(i, j, k)] = {x_nodes[i], y_nodes[j], z_nodes[k]};
  // z segment index of every element row
  std::vector<int> zseg(nz), layer_of(nz);
  int seg = 0;
  for (size_t l = 0; l < p.layers.size(); ++l) {
    const auto& L_ = p.layers[l];
    for (int kl = 0; kl < L_.nz; ++kl) {
      int k = layer_k0[l] + kl;
      if (kl > 0 && L_.z_every > 0 && kl % L_.z_every == 0) ++seg;
      zseg[k] = seg;
      layer_of[k] = int(l);
    }
    ++seg;  // layer boundary is always a plane
  }
  for (int k = 0; k < nz; ++k)
    for (int j = 0; j < ny; ++j)
      for (int i = 0; i < nx; ++i) {
        int e = (k * ny + j) * nx + i;
        hex_o[e] = {onode(i, j, k),     onode(i + 1, j, k),     onode(i + 1, j + 1, k),     onode(i, j + 1, k),
                    onode(i, j, k + 1), onode(i + 1, j, k + 1), onode(i + 1, j + 1, k + 1), onode(i, j + 1, k + 1)};
        const auto& L_ = p.layers[layer_of[k]];
        elem_layer[e] = layer_of[k];
        elem_ijk[e] = {i, j, k};
        elem_region[e] = {zseg[k], L_.x_every > 0 ? i / L_.x_every : 0, L_.y_every > 0 ? j / L_.y_every : 0};
      }
  // region ids and node copies
  auto rid = [&](int e) {
    auto& r = elem_region[e];
    return (r[0] * (nx + 1) + r[1]) * (ny + 1) + r[2];
  };
  std::vector<std::vector<int>> n2e(n_orig_nodes);
  for (int e = 0; e < ne; ++e)
    for (int a = 0; a < 8; ++a) n2e[hex_o[e][a]].push_back(e);
  std::vector<std::vector<std::pair<int, int>>> copies(n_orig_nodes);
  X.clear();
  for (int n = 0; n < n_orig_nodes; ++n)
    for (int e : n2e[n]) {
      int r = rid(e);
      bool found = false;
      for (auto& pr : copies[n])
        if (pr.first == r) found = true;
      if (!found) {
        copies[n].push_back({r, int(X.size())});
        X.push_back(Xo[n]);
      }
    }
  auto copy_of = [&](int n, int r) {
    for (auto& pr : copies[n])
      if (pr.first == r) return pr.second;
    throw std::runtime_error("mesh: missing node copy");
  };
  for (int e = 0; e < ne; ++e)
    for (int a = 0; a < 8; ++a) hex[e][a] = copy_of(hex_o[e][a], rid(e));

  // faces
  std::unordered_map<std::array<int, 4>, std::pair<int, int>, ArrHash> fm;
  fm.reserve(size_t(ne) * 3);
  ifaces.clear();
  bottom.clear(); top.clear(); xmin.clear(); xmax.clear(); ymin.clear(); ymax.clear();
  for (int e = 0; e < ne; ++e) {
    auto [i, j, k] = elem_ijk[e];
    for (int f = 0; f < 6; ++f) {
      std::array<int, 4> q;
      for (int a = 0; a < 4; ++a) q[a] = hex_o[e][HEX_FACES[f][a]];
      BFace bf;
      for (int a = 0; a < 4; ++a) bf.n[a] = hex[e][HEX_FACES[f][a]];
      bf.elem = e;
      if (f == 0 && k == 0) { bottom.push_back(bf); continue; }
      if (f == 1 && k == nz - 1) { top.push_back(bf); continue; }
      if (f == 5 && i == 0) { xmin.push_back(bf); continue; }
      if (f == 3 && i == nx - 1) { xmax.push_back(bf); continue; }
      if (f == 2 && j == 0) { ymin.push_back(bf); continue; }
      if (f == 4 && j == ny - 1) { ymax.push_back(bf); continue; }
      std::array<int, 4> key = q;
      std::sort(key.begin(), key.end());
      auto it = fm.find(key);
      if (it == fm.end()) {
        fm.emplace(key, std::make_pair(e, f));
        continue;
      }
      int eA = it->second.first, fA = it->second.second;
      fm.erase(it);
      int rA = rid(eA), rB = rid(e);
      if (rA == rB) continue;
      Iface ifc;
      for (int a = 0; a < 4; ++a) {
        int on = hex_o[eA][HEX_FACES[fA][a]];
        ifc.a[a] = copy_of(on, rA);
        ifc.b[a] = copy_of(on, rB);
      }
      auto &RA = elem_region[eA], &RB = elem_region[e];
      auto [iA, jA, kA] = elem_ijk[eA];
      ifc.layer = std::min(elem_layer[eA], elem_layer[e]);
      if (RA[0] != RB[0]) {
        ifc.kind = (elem_layer[eA] != elem_layer[e]) ? IF_LAYER : IF_Z;
        ifc.plane = std::max(kA, k);
      } else if (RA[1] != RB[1]) {
        ifc.kind = IF_X;
        ifc.plane = std::max(iA, i);
      } else {
        ifc.kind = IF_Y;
        ifc.plane = std::max(jA, j);
      }
      ifaces.push_back(ifc);
    }
  }
  build_graph();
}

void Mesh::build_graph() {
  const int nn = int(X.size());
  std::vector<std::vector<int>> nb(nn);
  auto link = [&](const int* ids, int m) {
    for (int a = 0; a < m; ++a)
      for (int b = 0; b < m; ++b) nb[ids[a]].push_back(ids[b]);
  };
  for (auto& h : hex) link(h.data(), 8);
  for (auto& f : ifaces) {
    int ids[8] = {f.a[0], f.a[1], f.a[2], f.a[3], f.b[0], f.b[1], f.b[2], f.b[3]};
    link(ids, 8);
  }
  adj_ptr.assign(nn + 1, 0);
  adj.clear();
  for (int n = 0; n < nn; ++n) {
    auto& v = nb[n];
    v.push_back(n);
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    adj_ptr[n + 1] = adj_ptr[n] + int(v.size());
    adj.insert(adj.end(), v.begin(), v.end());
  }
}

std::string Mesh::summary() const {
  int cnt[IF_NKINDS] = {0, 0, 0, 0};
  for (auto& f : ifaces) cnt[f.kind]++;
  std::ostringstream o;
  o << "mesh: " << nelem() << " hex, " << nnode() << " nodes (" << n_orig_nodes << " before duplication), "
    << ifaces.size() << " interface elements [x " << cnt[0] << ", y " << cnt[1] << ", z " << cnt[2] << ", layer "
    << cnt[3] << "], layers:";
  for (auto& l : p.layers) o << " " << l.thickness * 1e3 << " mm/" << l.nz;
  return o.str();
}

}  // namespace bsf
