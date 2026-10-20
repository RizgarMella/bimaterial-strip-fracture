# stripcz — study design

This document defines the computational studies of fracture in layered strips: the questions, the
model decisions behind them, the run order and gates, the data layout, the compute plan and the known
limitations. It follows the structure of the pellet projects
([peridynamic-pellet-fracture](https://github.com/RizgarMella/peridynamic-pellet-fracture),
[cohesive-pellet-fracture](https://github.com/RizgarMella/cohesive-pellet-fracture)).

**None of the studies below has been run.** The solver and planners are verified (`ctest`) and
dry-run. The README demos are single illustrative runs, not study results.

---

## 1. Questions

| # | Question | Study | Main output |
|---|----------|-------|-------------|
| Q1 | At what resolution do crack area, onset and channel count converge, for each Weibull treatment? | `mesh_convergence.py --series A` and `--series B` | GCI tables, `derived/convergence.json` (mesh used downstream) |
| Q2 | At fixed total thickness, how does the fraction of brittle top material set onset, channel-crack spacing, delamination, crack penetration and bending? | `thickness_ratio_study.py` | spacing/h_top, onset and delaminated fraction vs f, with seed bands |
| Q3 | For a growing brittle phase, how do the Pilling–Bedworth ratio, the normal/in-plane split of the swelling, the toughness of the phase and of the front, and the growth law set the critical depth, channel cracking and spallation? | `transformation_study.py` | critical depth, crack counts and spalled area vs each factor |
| Q4 | Can these be condensed into closed-form relations in dimensionless groups? | `collect` → `fit_pysr.py` | model cards (expression, validation error, training domain) |
| Q5 (optional) | In 3D plate patches, what mud-crack networks and 2D delamination/spallation patterns form, and does the 2D strip predict their scale? | `--dim 3d` versions of Q1–Q3 | cell counts and areas, plan views; comparison with 2D spacing |

## 2. Model and the decisions behind it

The model is a 3D small-strain finite-element solid (trilinear hexahedra). Zero-thickness cohesive
interfaces sit on predefined crack planes. The cohesive law, solver and time stepping are shared with
cohesive-pellet-fracture: triangular softening, viscous regularisation μ = 10⁻⁴ s, modified Newton with
the secant interface matrix, AMG-preconditioned CG, and steps on the viscous time scale during crack
bursts. The README *Model* table lists the ingredients. The decisions that shape the studies are these.

### 2.1 Weibull properties per zone, on mesh-independent fields

Each of the three zones has its own Weibull strength and fracture-energy distribution:
* bi-material: top layer, bottom layer, interface;
* transforming strip: substrate, growing phase, front.

Each zone also has its own correlation length, seed and strength–toughness correlation.

Values come from Gaussian random fields mapped through the Weibull quantile. The fields are generated
on a grid that does not depend on the FE mesh, so refining the mesh samples the same material. This is
what makes the convergence study (Q1) meaningful, as in the pellet projects.

The per-element alternative (`correlation_length = 0`, optional weakest-link area scaling) is kept as a
convergence family, because per-element scatter changes the weakest-link statistics under refinement.

In the transforming strip, a crack-plane point takes the distribution of the zone it is currently in,
and switches as the front passes. Damage is irreversible across the switch.

### 2.2 Transformation kinematics: element-wise fraction

The front depth follows s(t) = (k(t − t₀))^{1/n}, with n = 3 (cubic) by default. The phase swells by
the Pilling–Bedworth ratio, split into a normal strain e_n and an in-plane strain e_p:
* (1 + e_p)²(1 + e_n) = PBR;
* e_p = a·e_n, where a is `anisotropy`.

The transformed fraction is applied **uniformly within each element**, as the fraction of the element
row behind the front.

A smooth fraction inside an element is not compatible with the trilinear displacement field. With
e_n ≈ 0.5 it produced spurious stresses of order E·e_n that cracked the strip at sub-micron depths.
Row-wise constant strains are compatible across horizontal faces, so only the physical in-plane mismatch
loads the strip.

The cost is that the front is resolved only to the element-row height. Grade the mesh towards the
surface (`grading`); the critical depth converges with the grading.

### 2.3 Width condition: a wide free plate (`boundary.y_faces = generalized`)

A one-element-wide strip has to stand for something across its width.

**`generalized` (default).** This represents a wide free plate:
* u_y = 0 on y = 0;
* u_y = a + b(z − z_c)/H on y = W, where a and b are two global unknowns tied in by a penalty
  (`tie_penalty_factor`, default 30).

So the in-plane strain and curvature across the width are uniform and free. The mismatch is then
equi-biaxial with E/(1 − ν), and a uniform expansion is stress free. The tests check the biaxial
Timoshenko curvature (0.07 %) and the exact mirror of heating vs cooling with swapped CTEs.

**`constrained` (u_y = 0 on both faces).** This is a laterally clamped strip, and was the setting used
by the first demos. It adds a σ_yy proportional to each layer's *absolute* thermal or swelling strain
(≈ 680 MPa in the cooling case where this was found), and plane-strain (1 + ν)α factors. It is kept for
reproducibility and for genuinely constrained geometries.

**`free`.** A narrow strip (σ_yy ≈ 0).

The tie keeps two dense rows out of the AMG aggregation and uses the tie-free operator for the strength
of connection. Even so, it costs about 1.2× the PCG iterations of the constrained case.

### 2.4 Crack planes and what is counted

**Planes.** Crack planes are enabled per layer:
* x-planes: channel cracks;
* y-planes: longitudinal cracks; with x-planes they form networks in 3D;
* z-planes: delamination or spallation at any depth.

The layer interface is always cohesive.

**Bi-material decks.** These enable x-planes in both layers, so channel cracks may penetrate the
substrate.

**Transforming decks.** These enable x- and z-planes everywhere, because the front can spall at any
depth.

**Metrics.** These are written per step:
* **Through-crack:** an x-plane cracked through ≥ 90 % of the brittle zone's area on that plane.
* **Cells:** brittle-zone regions joined by intact x/y ligaments.
* **Delaminated area:** cracked horizontal planes, including the layer interface and the front.
* **Curvature:** a quadratic fit to the bottom centre line, on reference coordinates (small strain).

### 2.5 Loading paths

**Series A.** A uniform temperature history; each layer is strain-free at `reference_temperature`.
* The base case heats from 293 K.
* *Cooling from a stress-free state at temperature* (bonding or deposition) is set with
  `reference_temperature` and a falling history.
* With the wide-plate condition, heating with α_top < α_bot is exactly the mirror of cooling with
  α_top > α_bot, so one of the pair suffices.

**Series B.** Isothermal; only the front moves.

## 3. Studies, run order and gates

```
build + ctest ─► Q1 meshconv-A ─► Q2 ratio ─────┐
              └► Q1 meshconv-B ─► Q3 transform ─┴─► Q4 collect + fit_pysr
                                   (optional) Q5: --dim 3d versions after the 2D gates
```

### Q1 Mesh convergence on crack surface area (`mesh_convergence.py`)

**Design.**
* nx ∈ {40, 60, 80, 120, 160} on the 4 mm strip (100 → 25 µm), with the through-thickness rows scaled
  alike.
* Families `field` (main), `element` and `constant`, each with 3 seeds (constant with 1).

**Metrics.**
* Crack area and channel count at ΔT* = 100…500 K (A) or s* = 10…100 µm (B).
* Onset, delaminated area.
* ζ = l_cz/h, the cohesive length over the element size.

**Analysis.**
* Celik et al. (2008) GCI on the three finest levels, with Richardson extrapolation.
* Seed-noise test: |φ₁ − φ₂| ≤ 2 combined SEM.

**Gate.**
* The coarsest `field` level within 10 % of the extrapolated final crack area is written to
  `convergence.json`.
* If no level qualifies, report the finest level. Treat Q2/Q3 spacings as upper bounds and extend the
  levels.

**Context.** The README demos used nx = 80 (50 µm). There the spacings are close to the element size,
so Q1 is required before any spacing is quoted.

### Q2 Thickness ratio (`thickness_ratio_study.py`)

**Design.**
* h = 0.5 mm, with f = h_top/h ∈ {0.05, 0.1, 0.2, 0.3, 0.5, 0.7}.
* `--nz-total 40` keeps ≥ 2 rows in the thinnest top layer.
* Optional interface toughness {5, 20, 80} J/m², 3 seeds.
* Heating to ΔT = 500 K.

**Metrics.**
* Onset ΔT, channel cracks vs ΔT, saturated spacing/h_top.
* Delaminated fraction, penetration depth, curvature.

**Analysis.** Seed means, sd and bootstrap CI per level, plotted against f.

**What to look for.** The demos (f = 0.2 and 0.5) suggest two regimes:
* **Thin films (small f):** surface-to-interface channel cracks at roughly uniform tension. Spacing is
  about h_top.
* **Thick films (f ≳ 0.5):** strong bending puts the free surface of the brittle layer in compression.
  Cracks then start at the interface, arrest below the surface, and only run through after delamination
  relieves the bending.

Locating the transition in f, and how interface toughness shifts it, is the central result of Q2.

**Swapped-CTE (top expands more) variants.** Use `layer0/1.thermal_expansion` overrides:
* heating: compressed top layer, surface cracks and spallation;
* cooling: the mirror of the base case.

### Q3 Transformation (`transformation_study.py`)

**Design.** One factor at a time around (PBR 1.56, a = 0.005, G_c,phase = 5, G_c,front = 20 J/m², n = 3);
`--design full` gives the factorial.
* PBR ∈ {0.95, 1.28, 1.56, 1.79};
* a ∈ {0.002, 0.005, 0.02, 0.1};
* G_c,phase ∈ {2, 5, 20} J/m²;
* G_c,front ∈ {5, 20, 80} J/m²;
* `--exponents 2 3` adds parabolic growth;
* 2 seeds.
* The growth constant is set by `depth_at_time` (100 µm at 10⁴ s) for every exponent, so the laws differ
  in timing, not in final depth.

**Metrics.**
* Critical depth at the first crack.
* Channel cracks and spacing vs depth.
* Spalled area vs depth, and the number of spallation generations (read from the delaminated-area
  history).
* Crack depth into the substrate, curvature, and mean in-plane stresses.

**What to look for.** The demos suggest two modes:
* **PBR > 1 with mostly normal growth:** a compressed phase that spalls in generations.
* **PBR < 1:** a tensile phase that channel-cracks from a few µm.

The sign of e_p decides the mode, so the boundary in (PBR, a) is the first map to produce.

### Q4 Metamodel (`collect` → `fit_pysr.py`)

**Groups and targets.** `tools/features.py` defines the dimensionless groups and targets (README
*Metamodel*). Every 4th case is held out for validation.

**Fit.**
* PySR, with each Pareto equation scored on the validation split.
* The selected equation is the simplest within 5 % of the best validation RMSE.
* A gradient-boosting baseline shows what the closed form gives up.
* `--learning-curve` decides whether more cases are worth running.

**Caveat.** The one-factor-at-a-time designs are thin for a symbolic fit. A Sobol design over the same
ranges, as in the pellet campaign, is the natural extension once Q2/Q3 show which factors matter.

### Q5 Optional 3D plate patches (`--dim 3d`; planned, not run)

**Geometry.** 2 × 2 mm patches at 50 µm in-plane with x and y planes (`inputs/*_3d.ini`):
* Bi-material: the substrate has no crack planes, as the 2D demos never cracked it.
* Transforming: x, y and z planes everywhere.

**Additional metrics.** `n_brittle_cells` and `mean_cell_area_mm2`.

**Figures.** A plan view of the network (`render.py` → `plan.png`). Fields are always kept.

**Analysis.** Compare the mean cell size with the 2D spacing at the same f or PBR.

**Gate.**
* Run only after the 2D Q1 gate, at the recommended resolution.
* Time one coarse 3D case first (`mesh.nx=20 mesh.ny=20`).

## 4. Data layout

As in the pellet projects (`tools/study_common.py`):

```
studies/<study>/
  study.json  DATASET_CARD.md  cases.csv  cases.txt (MPI ensemble)  inputs/base.ini
  jobs/jobs.tsv  jobs/slurm_array.sh  jobs/cost_estimate.csv
  runs/<case_id>/   manifest.json  history.csv  summary.json  outputs.csv  xplanes_*.csv  (*.vtu, *.pvd)
  derived/*.csv + *.meta.json      figures/*.png       catalog.csv
```

**Case ids and provenance.**
* Case ids are `<prefix>-<index>-<hash of the overrides>`, so re-planning is idempotent.
* Every manifest records the build id: a hash of the sources plus the git revision.
* `summary.json` is written last, with `"complete": true` only for finished runs.

**What is stored.**
* 2D studies keep CSV/JSON only, unless `output.vtu=true` is passed.
* 3D studies keep the fields.

## 5. Compute plan

`python tools/study_common.py estimate <study> --cores N` builds every distinct mesh with
`stripcz --dry-run`, without matrices or a solve. It scales from demo A, which measured 21 k dofs,
6.6 k interface points and 1,220 s on 1 core:
* the cost per step grows as dofs^1.15;
* the number of crack-burst steps grows as (interface points)^0.5;
* the OpenMP speed-up depends on mesh size.

| study | cases | largest mesh | estimate (8 cores) | estimate (32 cores) |
|---|---|---|---|---|
| Q1 meshconv-A (2D) | 35 | 81 k dofs | ~110 core-h, longest ~1 h | – |
| Q1 meshconv-B (2D) | 35 | 184 k dofs | ~320 core-h, longest ~3 h | – |
| Q2 ratio (2D) | 18 | 40 k dofs | ~65 core-h | – |
| Q3 transform (2D) | 22 | 46 k dofs | ~95 core-h (see note) | – |
| Q5 ratio-3d | 18 | 0.32 M dofs | – | ~750 core-h, longest ~3 h |
| Q5 transform-3d | 22 | 0.92 M dofs | – | ~8,500 core-h, longest ~12 h |
| Q5 meshconv-A-3d (field) | 12 | 0.58 M dofs | – | ~600 core-h |
| Q5 meshconv-B-3d (field) | 4 | 3.1 M dofs, 11 GB | – | ~3,200 core-h, longest ~3.5 days |

**Accuracy.** These are order-of-magnitude figures, ±3×.

**Note on series B.** Transformation runs are dominated by repeated crack bursts. Demo B reached only
1,074 s of 10⁴ s in 35 min on 1 core, so the series-B estimates are probably low by several times.

**Before every series.** Time one case and rescale `REF` in `study_common.py`.

**Parallelism.** MPI distributes cases, not the domain, so every case must fit in one node.

## 6. Limitations and open issues

* **Predefined crack planes.** Cracks follow element boundaries; inclined kinks are not represented.
* **Small strain, linear elasticity.** There is no creep, plasticity or buckling. Large normal swelling
  strains are treated as small, though they are stress-free normal to the surface. Growth-stress
  relaxation is only represented through the anisotropy parameter a.
* **Element-wise front.** It is resolved to one element row.
* **Mesh-limited demo spacings.** Q1 must precede any quoted spacing.
* **Demo provenance.** README demos A, A′, B and C used the constrained width condition (§2.3). Demo B
  and C runs were stopped early. A″ uses the wide-plate condition.
* **Through-crack count.** The ≥ 90 % criterion is evaluated with floating-point areas. With 10 rows
  in the layer, a plane cracked through exactly 9 rows may or may not count. The `xplanes_*.csv`
  fractions are the unambiguous record.
* **Curvature in 3D.** On free 3D patches the curvature is measured along one centre line only.

## 7. References

See the README. Key sources for the questions: Hutchinson & Suo (1992), Thouless (1991), Evans &
Hutchinson (1995) for thin-film cracking and delamination; Timoshenko (1925) for bimetal bending;
Pilling & Bedworth (1923) for oxide volume ratios.
