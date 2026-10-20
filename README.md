# stripcz — cohesive-zone fracture of bi-material and transforming strips

A parallel C++17 finite-element code for cracking of layered strips, built on the same stack and
approach as [cohesive-pellet-fracture](https://github.com/RizgarMella/cohesive-pellet-fracture).
That stack is cohesive-zone crack planes, Weibull-distributed strength and toughness on mesh-independent
random fields, AMG-preconditioned CG with OpenMP, viscous-regularised adaptive time stepping and MPI
ensembles. The study tooling (run directories, crack-area convergence with GCI, series planners, PySR
fit) follows [peridynamic-pellet-fracture](https://github.com/RizgarMella/peridynamic-pellet-fracture).

Two problems:

* **Series A — bi-material strip, differential expansion.**
  * A brittle, low-toughness top layer is bonded to a tougher substrate with a different thermal
    expansion coefficient.
  * Heating stretches the top layer. It channel-cracks, the cracks multiply until their spacing saturates,
    and they may kink into the interface (delamination) or penetrate the substrate.
  * The **top layer, bottom layer and interface each have their own Weibull strength and toughness
    distributions.**
  * The planned study keeps the total thickness fixed and varies the fraction of top material.
* **Series B — transforming strip, swelling.**
  * A single-material strip whose top surface converts into a more brittle phase.
  * The front advances as **s(t) = (k t)^{1/3}** (cubic law; exponent configurable).
  * The new phase swells by a chosen **Pilling–Bedworth ratio**, split between the surface-normal and
    in-plane directions. **The swelling, not a thermal mismatch, drives the stresses.**
  * The **substrate, growing phase and transformation front have separate distributions.**

> **Status.** Solver, verification tests and study planners are complete and the tests pass. The series
> themselves have **not** been run. The images below come from six small demonstration runs and one 3D
> smoke test (see [Demos](#demos)). Two of the runs were stopped early, which the Demos section notes.
>
> **Documentation.**
> * [docs/DESIGN.md](docs/DESIGN.md): questions, model decisions, run order and gates, data layout,
>   compute plan, limitations.
> * [docs/INPUT_REFERENCE.md](docs/INPUT_REFERENCE.md): every input key.
> * [CHANGELOG.md](CHANGELOG.md): what changed, including the width-condition fix that affects the
>   earlier demos.

---

## Model

| Ingredient | Implementation |
|---|---|
| Geometry | strip L × W × h (x, y, z), one or more bonded layers, structured hexahedra (2×2×2 Gauss), optional grading towards the top of a layer |
| Potential cracks | zero-thickness cohesive interfaces, enabled per layer: x-planes (channel cracks), y-planes (longitudinal cracks; with x-planes they form mud-crack networks when ny > 1), z-planes (delamination / spallation at any depth), and the layer interface (always cohesive) |
| Cohesive law | penalty K, linear or exponential softening, mixed-mode effective separation (β), irreversible damage with viscous regularisation, full-penalty contact in compression, optional friction (shared with pelletcz) |
| Statistics | per zone: Weibull strength and fracture energy (own mean, modulus, threshold), on a Gaussian random field of the zone's own correlation length and seed (or per element with weakest-link scaling), strength–toughness correlation ρ; local flaws |
| Series A load | uniform temperature history; each layer has its own E, ν, α; eigenstrain α(T − T_ref) |
| Series B load | front depth s(t) = (k(t − t₀))^{1/n}. The transformed fraction is uniform within each element (the fraction of the element row behind the front), with mixed elastic constants. The eigenstrain is φ·diag(e_p, e_p, e_n), with (1 + e_p)²(1 + e_n) = PBR and e_p = a·e_n (a = 1 isotropic, a ≪ 1 for growth mostly normal to the surface) |
| Zones in B | a vertical crack-plane point uses the substrate or phase distribution; a horizontal plane with phase above and substrate below is the *front*. Points switch zone as the front passes; damage is irreversible |
| Boundary conditions | `free` (statically determinate 3-2-1 pins; the strip bends freely, pin reactions are reported), `roller_bottom`, `clamped_bottom` (rigid substrate), `clamped_end` (cantilever). Width faces `y_faces`: `free` (narrow strip), `generalized` (wide free plate: uniform in-plane strain and curvature across the width, set by two global dofs through a penalty tie), `constrained` (u_y = 0, a laterally clamped strip) |
| Numerics | modified Newton with symmetric secant interface matrix and line search; PCG + smoothed-aggregation AMG (rigid-body near-nullspace); adaptive steps rejected on large damage jumps; dt ~ 2μ while cracks run; initiation approached via the failure index; load change limited per step (ΔT, or front advance ≤ half the thinnest element) |
| Parallelism | OpenMP throughout; `mpirun -n R stripcz --ensemble cases.txt` runs cases over ranks; Slurm array scripts from the planners |

### Why the transformed fraction is element-wise

A swelling strain that varies *inside* a trilinear element cannot be followed by its linear
displacement field. With a 50 % normal swelling strain this produces spurious stresses of order E·e_n,
and in testing those cracked the strip at sub-micron front depths.

The transformed volume fraction is therefore applied uniformly within each element. Row-wise constant
strains are compatible across horizontal element faces, so only the real in-plane mismatch between
phase and substrate generates stress. Grade the mesh (`layer0.grading`) so the phase is resolved by
several element rows.

---

## Build and run

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
cd build && ctest --output-on-failure && cd ..
OMP_NUM_THREADS=4 ./build/stripcz inputs/bimaterial_strip.ini          # series A base case
OMP_NUM_THREADS=4 ./build/stripcz inputs/transforming_strip.ini        # series B base case
./build/stripcz inputs/bimaterial_strip.ini geometry.top_fraction=0.3 fracture.top.strength.weibull_modulus=10
python tools/render.py runs/bimaterial_strip --frames 3,6,9,-1         # figures (matplotlib)
```

Every input key is listed in [docs/INPUT_REFERENCE.md](docs/INPUT_REFERENCE.md). ParaView reads
`bulk.pvd`, `surface.pvd` and `cracks.pvd`.

### Outputs (run directory)

| file | content |
|---|---|
| `manifest.json` | inputs, build id, labels, derived quantities: cohesive lengths and ζ = l_cz/h per zone, transformation strains, volume-ratio check |
| `history.csv` | one row per step (units in the names), see below |
| `bulk_/surface_/cracks_NNNN.vtu` | stresses, phase fraction, layer, fragments; crack faces with damage, opening, zone, strength, toughness |
| `xplanes_NNNN.csv` | per x-plane: cracked fraction of the brittle layer or phase, crack depth, opening (crack-spacing statistics) |
| `outputs.csv`, `summary.json` | output times; final state, onset, crack area by plane type and by zone, fragment volumes (`"complete": true` only for finished runs) |

`history.csv` columns:

* **Load:** T, ΔT, front depth.
* **Crack area:** by plane type (x / y / z / layer interface) and by zone; damaged area; dissipated energy.
* **Channel cracks:** x-planes cracked through ≥ 90 % of the brittle layer, and their mean spacing;
  brittle-layer cells bounded by through-cracks and their mean plan area (the mud-crack network in 3D).
* **Damage extent:** deepest crack below the surface; delaminated area (cracked horizontal planes).
* **Strip state:** curvature of the bottom centre line; mean σ_xx of the brittle layer and the substrate
  over the middle third; max principal stress; fragments.
* **Checks and solver effort:** pin reaction, Newton and PCG iterations.

---

## Verification

`ctest` runs `tests/test_main.cpp`:

| test | result |
|---|---|
| cohesive law: peak traction = σ_t, work of separation = G_c | < 0.1 % |
| PBR: isotropic e = PBR^{1/3} − 1; anisotropic split reproduces the volume ratio exactly | exact |
| growth law s = (k t)^{1/3} from `depth_at_time` | exact |
| zones: phase above the front, front plane, substrate below; all substrate before transformation | exact |
| Weibull per zone: mean strength of top layer and substrate; interface point count | < 0.6 % |
| fully transformed free block: extensions e_p·L and e_n·H, stress free | 0.0 % |
| **bimetal curvature** vs Timoshenko (1925), laterally constrained plane strain, free strip; no pin reactions | 0.7 % |
| **wide plate** (`generalized`): biaxial bimetal curvature with E/(1 − ν); cooling with α_top > α_bot gives exactly the stresses and curvature of heating with α_top < α_bot; uniform expansion of one material is stress free | 0.07 % / exact / < 25 Pa |

---

## Demos

Small runs. All use 80 × 1 elements along the 4 mm strip (50 µm in-plane), `free` supports, and
Weibull fields with correlation length 30–50 µm in every zone. At 50 µm the in-plane element size is
close to the crack spacings found, so spacings here are **mesh-limited**. They show the behaviour; the
convergence study is what turns them into numbers.

> **Width condition.** Demos A, A′, B and C were run with the earlier *laterally constrained* width
> condition (u_y = 0 on both faces, now `boundary.y_faces = constrained`). That condition adds a σ_yy
> proportional to each layer's absolute thermal or swelling strain, and plane-strain (1 + ν)α factors to
> the mismatch.
>
> Demos A″ use the **wide-plate** condition (`generalized`), which is now the default in the inputs. To
> reproduce the earlier demos exactly, add `boundary.y_faces=constrained`. In the wide-plate condition,
> case A′ is exactly the mirror of A″-cool (verified by the tests); with the constrained condition its
> curvature was ≈ 35 % larger.

### A — bi-material strip heated by 500 K (`inputs/bimaterial_strip.ini`)

**Setup.**
* Layers: 0.1 mm top layer (E = 380 GPa, α = 8×10⁻⁶ /K, σ̄_t = 150 MPa, G_c = 10 J/m², m = 6) on a
  0.4 mm substrate (E = 200 GPa, α = 17×10⁻⁶ /K, 400 MPa, 100 J/m², m = 15).
* Interface: 200 MPa, 20 J/m², m = 8.
* Mesh: 20 element rows.
* Cost: 3,718 steps, 20 min on 1 core.

![Bi-material strip side views](docs/images/bimaterial_elevation.png)

![Bi-material strip histories](docs/images/bimaterial_history.png)

**Before cracking.** The substrate expands more, so the strip bends towards the top layer. The top layer
goes into tension, while the substrate just below the interface is in compression.

**Initiation.** The first crack forms at **ΔT = 78 K** in a weak region of the top-layer field.

**Multiplication.**
* Channel cracks then multiply: 4 at 112 K, 11 at 187 K, 18 at 300 K, and **31 at 500 K**.
* The final mean spacing is 125 µm, or 1.25 h_top.
* Each new crack relaxes the stress in the segments next to it, so the curvature peaks at 2.75 /m
  (ΔT ≈ 110 K) and then falls to 0.19 /m.

**Delamination and arrest.**
* From ≈ 130 K, cracks kink into the interface. **2.05 mm of the 4 mm** interface has delaminated by
  500 K.
* No crack entered the substrate: substrate crack area is 0, and the deepest crack is 95 µm (the top
  layer's Gauss points).
* The tougher substrate arrests the channels.

**Check.** The 3-2-1 pins carried < 0.4 mN throughout, so they held the strip without loading it.

### A′ — the same strip split 50/50

Same materials, fields and heating as A, but `geometry.top_fraction=0.5`: 0.25 mm of brittle top on
0.25 mm of substrate, 10 element rows each. It ran in 8.5 min on 2 cores.

![50/50 bi-material strip side views](docs/images/bimaterial_50_50_elevation.png)

![50/50 bi-material strip histories](docs/images/bimaterial_50_50_history.png)

**Bending.** With half the thickness brittle, the strip bends much more, peaking at **7.8 /m** at
ΔT ≈ 250 K against 2.75 /m for A. The bending puts a strong stress gradient through the top layer:
* at the interface it is in tension;
* at its free surface it is in **compression**.

**Cracks start at the interface.** They are not surface cracks:
* The first crack forms at **ΔT = 110 K**, later than A's 78 K.
* At 200 K, 17 planes have cracked, 13 of them more than half way, but **none reaches the top
  surface**. They arrest in the compressed upper part of the layer.

**Through-cracks follow delamination.**
* Delamination starts at ≈ 216 K and relieves the bending. Only then do the cracks run through:
  3 at 300 K, and **8 through the full layer** at 500 K (12 through ≥ 90 % of it).
* The final spacing is ≈ 440 µm, or 1.8 h_top (A: 1.25 h_top).
* By 500 K **2.82 mm** of the 4 mm interface has delaminated (A: 2.05 mm).
* The substrate again stays uncracked.

```bash
./build/stripcz inputs/bimaterial_strip.ini geometry.top_fraction=0.5 boundary.y_faces=constrained \
    output.directory=runs/bimaterial_strip_50_50
python tools/render.py runs/bimaterial_strip_50_50 --frames 4,13,27,46 --warp 20
```

### A″ — 50/50 strip with the top layer expanding more (wide plate)

Same 50/50 strip and Weibull fields as A′, but the expansion coefficients are swapped. The brittle top
now has α = 17×10⁻⁶ /K and the substrate 8×10⁻⁶ /K; E and ν are unchanged. The wide-plate width
condition is used. Two loadings:

* **Heated by 500 K** from 293 K. The top wants to grow more than the substrate, so the strip bends with
  the top convex.
* **Cooled by 500 K** from a stress-free state at 793 K, as after bonding at temperature. The top
  shrinks more and goes into tension. Because the width condition is now right, this is exactly the
  mirror of heating with the original coefficients.

**Heated:**

![Top expands more, heated](docs/images/top_expands_heat_elevation.png)

* Bending puts the brittle layer's **free surface in tension** and its lower part in compression, about
  −80 MPa on average.
* **Surface cracks** start at ΔT = 140 K. They grow down to ≈ 105 µm, 40 % of the layer, and stop in the
  compressed part. **None runs through the layer.**
* Curvature reaches −7.3 /m at 230 K. Then the **interface fails**: delamination starts at 221 K and is
  almost complete by 240 K (3.79 of 4 mm, 3.95 mm by 500 K).
* The top layer comes off as one cracked sheet: **spallation**, not channel cracking.
* Run time: 83 min on 1 core (shared).

![Top expands more, heated - histories](docs/images/top_expands_heat_history.png)

**Cooled:**

![Top expands more, cooled](docs/images/top_expands_cool_elevation.png)

* The top layer is in tension, highest at the interface. Cracks start there at |ΔT| = 122 K and grow
  upwards, stopping below the compressed free surface.
* Delamination starts at |ΔT| = 239 K. The first through-cracks follow at 276 K: 2 at 300 K and **9 at
  500 K**, ≈ 400 µm apart.
* 2.65 mm of the interface has delaminated by the end. The substrate is uncracked.
* This is the sequence of A′, which was the mirrored problem, now without the constrained-width
  artefacts: curvature 2.7 /m against 3.6 /m at |ΔT| = 100 K.
* Run time: 102 min on 1 core (shared).

```bash
./build/stripcz inputs/bimaterial_strip.ini geometry.top_fraction=0.5 layer1.thermal_expansion=17e-6 \
    layer0.thermal_expansion=8e-6 output.directory=runs/bimaterial_50_50_top_expands_heat
./build/stripcz inputs/bimaterial_strip.ini geometry.top_fraction=0.5 layer1.thermal_expansion=17e-6 \
    layer0.thermal_expansion=8e-6 material.reference_temperature=793.15 \
    "loading.temperature_history=0,793.15;500,293.15" output.directory=runs/bimaterial_50_50_top_expands_cool
python tools/render.py runs/bimaterial_50_50_top_expands_heat --frames 2,6,24,27 --warp 20
python tools/render.py runs/bimaterial_50_50_top_expands_cool --frames 4,10,24,44 --warp 20
```

### B — transforming strip, PBR = 1.56, swelling mostly normal to the surface (`inputs/transforming_strip.ini`)

**Setup.**
* Substrate: 0.5 mm, Zr-like (E = 95 GPa).
* Transformed phase: ZrO₂-like (E = 200 GPa, σ̄ = 150 MPa, G_c = 5 J/m², m = 6).
* Front: 150 MPa, 20 J/m².
* Growth: cubic law reaching 100 µm at 10⁴ s.
* Swelling: a = 0.005, giving e_n = 0.551 and e_p = 0.0028, with (1 + e_p)²(1 + e_n) = 1.56.
* Mesh: 24 rows graded towards the surface.
* **Run length:** stopped at t = 1,074 s (front at 47.5 µm), so not the full 10⁴ s. It was 35 min on
  1 core; the bursts below need 10⁻⁴ s steps.
* The figures use true displacements and zoom onto the top 120 µm of the left half.

![Transforming strip, PBR 1.56](docs/images/transform_pbr156_elevation.png)

![Transforming strip histories, PBR 1.56](docs/images/transform_pbr156_history.png)

**Stress and bending.**
* The in-plane swelling is constrained, so the phase sits at **−400 to −700 MPa** compression.
* Neither channel cracks nor the tensile mode of series A appear.
* The strip bends away from the phase, with curvature −3 /m before the first spall.

**First spallation.**
* At a front depth of **23 µm** (t ≈ 125 s) the whole phase layer **spalls** along the plane just above
  the front.
* The spall starts at a free end (x ≈ 3.9 mm, edge peel) and runs the full length within ≈ 0.2 s.
* The released layer becomes stress free.

**Second spallation.**
* The front keeps advancing under the spalled layer, so a second compressed phase layer builds up.
* At **47 µm** it too starts to spall from the ends: delaminated length 7.25 mm, i.e. two layers across
  most of the strip.
* This repeated spalling produces a layered scale, as seen in breakaway oxidation.

### C — transforming strip, PBR = 0.995, isotropic (a tensile phase)

**Setup.** The same strip with PBR < 1: the phase *shrinks* (e = −0.17 % in every direction). Run with
`transformation.pilling_bedworth_ratio=0.995 transformation.anisotropy=1`. **Run length:** stopped at
t = 210 s (front at 27.6 µm).

![Transforming strip, PBR 0.995](docs/images/transform_pbr0995_elevation.png)

**Behaviour.**
* The constrained shrinkage puts the phase in **tension** (≈ 300 MPa before cracking), so it behaves
  like series A.
* The first crack forms at a front depth of only **6.2 µm**.
* By 21.5 µm there are 31 channel cracks, about one per element, which is the mesh-limited spacing.
* As the phase thickens, some older cracks no longer reach through it, so the count of through-cracks
  drops to 25–27. They are slowly re-initiated deeper.
* Only 0.12 mm of the front has delaminated.

Reproduce (render options as used above):

```bash
./build/stripcz inputs/bimaterial_strip.ini boundary.y_faces=constrained
python tools/render.py runs/bimaterial_strip --frames 3,9,23,37 --warp 20
./build/stripcz inputs/transforming_strip.ini boundary.y_faces=constrained   # stopped at t = 1,074 s
python tools/render.py runs/transforming_strip --frames 1,3,20,-1 --warp 1 --xlim 0 2 --zlim 0.42 0.54
./build/stripcz inputs/transforming_strip.ini boundary.y_faces=constrained transformation.pilling_bedworth_ratio=0.995 \
    transformation.anisotropy=1 output.directory=runs/transforming_strip_pbr0995   # stopped at t = 210 s
python tools/render.py runs/transforming_strip_pbr0995 --frames 1,5,-1 --warp 20 --xlim 1 2 --zlim 0.40 0.52
```

---

## Studies

All planners write a self-contained study directory, using the same layout as the pellet projects:
`study.json`, `cases.csv`, `cases.txt` for MPI, `jobs/jobs.tsv`, a Slurm array script and a dataset card.
Each case is the base input plus overrides. Runs are resumable, and every derived table has a
`.meta.json` sidecar.

```bash
python tools/thickness_ratio_study.py plan --seeds 3 --interface-toughness 5 20 80
python tools/study_common.py run studies/ratio --parallel 2 --threads 2       # or: mpirun ... --ensemble
python tools/thickness_ratio_study.py analyze studies/ratio && python tools/thickness_ratio_study.py plot studies/ratio
```

### Series A — fixed total thickness, varying top-material fraction (`tools/thickness_ratio_study.py`)

**Design.**
* Fixed geometry: h = 0.5 mm and the same number of elements through the thickness. The interface
  always lies on a node plane, with ≥ 2 elements in the top layer.
* Top fraction f = h_top/h ∈ {0.05, 0.1, 0.2, 0.3, 0.5, 0.7}.
* Optional interface-toughness levels.
* Independent seeds for all three zone fields.
* Load: heating to ΔT = 500 K.

**Metrics.**
* Onset ΔT.
* Channel cracks vs ΔT.
* Saturated spacing, also as spacing/h_top.
* Delaminated fraction.
* Crack penetration into the substrate.
* Curvature.

**Analysis and figures.**
* Seed means, sd and bootstrap CI per level.
* Figures: onset, spacing/h_top, delamination and curvature against f; crack-count curves.

### Series B — growing brittle phase with Pilling–Bedworth swelling (`tools/transformation_study.py`)

**Design.** One factor at a time (default) or full factorial over:
* PBR ∈ {0.95, 1.28, 1.56, 1.79};
* anisotropy a ∈ {0.002, 0.005, 0.02, 0.1};
* phase toughness ∈ {2, 5, 20} J/m²;
* front toughness ∈ {5, 20, 80} J/m².

Each runs with the cubic law; `--exponents 2 3` adds parabolic growth. Phase, substrate and front each
have their own seeds.

**Metrics.**
* Critical depth at first crack.
* Channel cracks and their spacing vs depth.
* Spallation (delaminated area) vs depth.
* Crack depth into the substrate.
* Curvature and in-plane stresses.

### Mesh convergence on crack surface area (`tools/mesh_convergence.py --series A|B`)

Refines nx and the element rows together; levels nx ∈ {40, 60, 80, 120, 160}. Three families, as in
pelletcz:
* `field`: correlated fields of fixed physical length in every zone, so refinement samples the same
  material;
* `element`: per-element draws with weakest-link scaling;
* `constant`: no scatter.

Analysis:
* Crack area and channel count at load levels.
* Celik GCI on the three finest levels, with Richardson extrapolation and a seed-noise test.
* The recommended level is written to `derived/convergence.json`.

### Optional 3D studies (planned, not run)

Every planner takes `--dim 3d`. This swaps the one-element-wide strip for a **2 × 2 mm plate patch**:
* `inputs/bimaterial_strip_3d.ini` and `inputs/transforming_strip_3d.ini`;
* 50 µm elements in-plane, the same thickness and materials as the 2D bases, free edges;
* crack planes on every element boundary in **x and y**, so channel cracks can join into a
  **mud-crack network**;
* interfaces and fronts that delaminate or spall in 2D patches rather than lines.

The factors, metrics and analysis are the same as in 2D, plus two network metrics:
* `n_brittle_cells`: cells of the brittle layer bounded by cracked x/y planes;
* `mean_cell_area_mm2`.

`tools/render.py` adds a top-surface plan view of the crack network (`plan.png`) whenever `ny > 1`.
The example below is only a **10 × 10-element smoke test** of the 3D deck (200 µm elements, 5 min on
2 cores), run to check the network metric and the plan view. It is not a study result:

![3D smoke test, plan view](docs/images/smoke3d_plan.png)

Fields are kept (`output.vtu = true`), because a 3D run is too expensive to repeat just for pictures.

```bash
python tools/thickness_ratio_study.py plan --dim 3d            # -> studies/ratio-3d
python tools/transformation_study.py plan --dim 3d             # -> studies/transform-3d
python tools/mesh_convergence.py plan --series A --dim 3d      # nx = ny = 20, 30, 40, 60
python tools/study_common.py estimate studies/ratio-3d --cores 32   # dry-run sizes + cost, no simulation
cd studies/ratio-3d && mpirun -n 4 ../../build/stripcz --ensemble cases.txt --threads 32   # on a cluster
```

`study_common.py estimate` sizes every distinct mesh with `stripcz --dry-run`, which builds the mesh and
the Weibull fields but no matrices. It then gives an order-of-magnitude cost, scaled from demo A's
measured 21 k dofs and 20 min on 1 core:
* the cost per step grows as dofs^1.15;
* the number of crack-burst steps grows as (interface points)^0.5;
* the OpenMP speed-up depends on mesh size.

Current estimates on 32 cores, accurate to about ±3×:

| study (`--dim 3d`) | cases | largest mesh | memory | longest case | total |
|---|---|---|---|---|---|
| thickness ratio (6 fractions × 3 seeds) | 18 | 0.32 M dofs | 1.3 GB | ~3 h | ~750 core-h |
| transformation (one factor at a time × 2 seeds) | 22 | 0.92 M dofs | 3.2 GB | ~12 h | ~8,500 core-h |
| mesh convergence, series A, `field` family (3 seeds) | 12 | 0.58 M dofs (nx = 60) | 2.4 GB | ~5 h | ~600 core-h |
| mesh convergence, series B, `field` family (1 seed) | 4 | 3.1 M dofs (nx = 60) | 10.8 GB | ~3.5 days | ~3,200 core-h |

**Choices that keep the 3D decks affordable.**
* The bi-material substrate has no crack planes. The 2D demos never cracked it; re-enable with
  `layer0.x_cracks=all layer0.y_cracks=all`.
* The transformation deck needs x, y and z planes everywhere, because the front can spall at any depth.
  That is why it is roughly 5× larger.

**Before launching a series:**
* Run a single coarse case (e.g. `mesh.nx=20 mesh.ny=20`) and replace the estimate with its measured
  time.
* Remember that MPI distributes cases, not the domain: each case must fit in one node's memory.

### Metamodel (`collect` → `tools/fit_pysr.py`)

Each series' `collect` writes `derived/dataset.csv`. It holds the dimensionless groups of
`tools/features.py` and a train/validation split.

Series A groups:
* Π_mis = E′|Δα|ΔT/σ_t;
* h_top/l_ch, E_t/E_b, G_c,i/G_c,t, σ_i/σ_t, m.

Series B groups:
* Π_swell = E_p′|e_p|/σ_p;
* PBR, a, s/l_ch, s/H, E_p/E_s, G_c,f/G_c,p, m.

Targets:
* spacing / thickness;
* onset stress ratio (A) or critical depth / l_ch (B);
* delaminated fraction.

`fit_pysr.py` fits PySR, scores the Pareto front on the validation split, keeps the simplest equation
within 5 % of the best, and compares with gradient boosting. Without PySR it reports the gradient-boosting
baseline only.

---

## Limitations

* **Predefined crack planes.** Cracks follow element boundaries on enabled planes; use `all` spacing.
  Inclined crack paths, such as kinking at angles other than 90°, are not represented.
* **Small strain, linear elasticity, no creep or plasticity.**
  * Growth stresses in real oxides relax by creep. The anisotropy parameter is the knob for the
    effective in-plane mismatch.
  * Large normal swelling strains (e_n ≈ 0.5 for PBR = 1.56, a = 0.005) are treated as small strains.
    They are stress-free in the normal direction, so this mainly affects the geometry.
* **No buckling.** Compressed thin phases spall here by edge peel and shear, never by buckling.
* **Element-wise transformation.** The front is resolved to the element-row size; the onset depth
  converges with the grading.
* **Width condition.** `generalized` assumes the in-plane strain and curvature across the width are
  uniform, as in a wide plate. The tie costs about 1.2× the PCG iterations. Demos A, A′, B and C used
  the older `constrained` condition (see [CHANGELOG.md](CHANGELOG.md)).
* **Through-crack count.** The ≥ 90 % threshold is evaluated on floating-point areas. A plane cracked
  through exactly 90 % of the layer may or may not count; `xplanes_*.csv` gives the fractions.
* **MPI parallelises cases, not one solve.**

## Repository

| path | content |
|---|---|
| `src/` | solver. `mesh.*` (layered mesh, crack-plane regions, node duplication); `model.*` (zones, Weibull fields, transformation kinematics, boundary conditions); `mechanics.cpp` (eigenstrains, interfaces, width tie, Newton); `solver.cpp` (time stepping, metrics, fragments, cells); `output.cpp` (VTU/CSV/JSON); `interface_law.*`, `random_field.*`, `sparse.*`, `amg.*`, `config.*`, `fe.hpp` shared with pelletcz; `main.cpp` (CLI, dry run, MPI ensemble) |
| `tests/` | verification tests (`ctest`) |
| `inputs/` | 2D bases `bimaterial_strip.ini`, `transforming_strip.ini`; optional 3D decks `*_3d.ini` |
| `tools/` | `stripio.py` (readers), `render.py` (side, plan and history figures), `study_common.py` (planning, running, catalogue, cost estimate), `thickness_ratio_study.py`, `transformation_study.py`, `mesh_convergence.py`, `features.py`, `fit_pysr.py` |
| `docs/` | `DESIGN.md`, `INPUT_REFERENCE.md`, demo images |

## References

* Timoshenko S. (1925) Analysis of bi-metal thermostats. J. Opt. Soc. Am. 11, 233–255.
* Pilling N.B., Bedworth R.E. (1923) The oxidation of metals at high temperatures. J. Inst. Metals 29, 529–591.
* Hutchinson J.W., Suo Z. (1992) Mixed mode cracking in layered materials. Adv. Appl. Mech. 29, 63–191.
* Evans A.G., Hutchinson J.W. (1995) The thermomechanical integrity of thin films and multilayers. Acta Metall. Mater. 43, 2507–2530.
* Thouless M.D. (1991) Cracking and delamination of coatings. J. Vac. Sci. Technol. A 9, 2510–2515.
* Williamson R.L., Knoll D.A. (2009) Simulating dynamic fracture in oxide fuel pellets using cohesive zone models. SMiRT-20, paper 1775 (the cohesive-zone approach and viscous regularisation).
* Turon A. et al. (2007) Eng. Fract. Mech. 74 (cohesive-zone mesh size); Celik I.B. et al. (2008) J. Fluids Eng. 130 (GCI); Vaněk P. et al. (1996) Computing 56 (smoothed-aggregation AMG); Cranmer M. (2023) arXiv:2305.01582 (PySR).
