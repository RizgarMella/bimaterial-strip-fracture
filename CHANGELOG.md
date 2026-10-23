# Changelog

## 2026-10-23

### Higher-resolution cooled 50/50 demo; AMG fix for the wide-plate tie
* README demo A‴: the cooled A″ strip at 25 µm (160 × 40), with an interface 20 % weaker
  (160 MPa, 16 J/m²).
* **AMG fix.** Unaggregated nodes (the wide-plate tie node) are now dropped from the smoothed
  prolongator.
  * Before, the tie node's row picked up entries from every aggregate along y = W. That made the first
    coarse level fully dense, and it failed with "coarsest level too large" on fine meshes.
  * The separate strength matrix is now used on the fine level only.
* `STRIPCZ_AMG_DEBUG=1` prints aggregation statistics per level.

## 2026-10-20

### Optional 3D studies; top-layer-expands-more demos (`1214723`)
* 3D plate-patch decks (`inputs/*_3d.ini`) and `--dim 3d` in all three planners.
* `study_common.py estimate`: dry-run sizing and order-of-magnitude cost; no simulation.
* Brittle-cell count and mean cell area (mud-crack network metric); plan-view render (`plan.png`).
* `--dry-run` now builds only the mesh, zones and properties, and reports dofs and memory.
* Tie penalty factor 30 (`boundary.tie_penalty_factor`): < 0.02 % change in curvature, about half the
  PCG iterations of 1000.
* `summary.json` reports the onset ΔT for cooling histories (negative ΔT).
* README demos A″: 50/50 strip with α_top > α_bot, heated and cooled, using the wide-plate condition.

### Wide-plate width condition (`dc05f5c`)
* `boundary.y_faces = free | generalized | constrained`.
  * `generalized` is a wide free plate: uniform in-plane strain and curvature across the width.
  * It is now the default in the input decks.
* **Behaviour change.** The previous `plane_strain_y = true` (u_y = 0 on both faces, now `constrained`)
  made σ_yy depend on each layer's absolute strain. The problem showed up as a cooling run that should
  have mirrored a heating run but did not. Demos A, A′, B and C in the README were run with that
  condition and say so.
* Curvature is fitted on reference coordinates. The old fit on deformed coordinates had a 0.5 % bias.
* AMG accepts a separate strength-of-connection matrix and keeps dense tie rows out of the aggregation.
* New tests: biaxial Timoshenko curvature, exact heat/cool mirror, stress-free uniform expansion.

### 50/50 demo (`9bffda5`)
* README demo A′: `geometry.top_fraction = 0.5`. The cracks start at the interface and run through only
  after delamination.

### Demos and fixes (`b20b430`, `7197c7e`)
* README demos A (bi-material), B (PBR 1.56) and C (PBR 0.995).
* Renderer zoom window.

### First release (`ca720a3`)
* Solver for bi-material strips (differential expansion) and transforming strips. Transforming strips
  have cubic growth and Pilling–Bedworth swelling with a normal/in-plane split.
* Weibull strength and toughness per zone on mesh-independent fields.
* Verification tests; study planners (thickness ratio, transformation, mesh convergence); PySR fit.
