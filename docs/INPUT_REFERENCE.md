# Input reference

INI files: `[section]`, `key = value`, `#` comments. Any key can be overridden on the command line as
`section.key=value` (this is how the study tools build their cases). Unused keys are reported at the
end of a run. Tables are `time, value; time, value; ...`, piecewise linear and constant outside their range.

## [model]
| key | default | meaning |
|---|---|---|
| `type` | bimaterial | `bimaterial` (layers + temperature history) or `transformation` (one layer, growing phase) |

## [geometry] and [mesh]
| key | default | meaning |
|---|---|---|
| `geometry.length`, `geometry.width` | 4e-3, 0.25e-3 | strip length (x) and width (y) [m] |
| `geometry.n_layers` | 2 | layers, bottom to top (`[layer0]`, `[layer1]`, ...); 1 for `transformation` |
| `geometry.total_thickness`, `geometry.top_fraction` | - | two layers: fix the total thickness and set h_top/h (thickness-ratio series) |
| `mesh.nx`, `mesh.ny` | 80, 1 | elements along x and y |
| `mesh.nz_total` | - | with `top_fraction`: elements through the thickness, split between the layers |

## [layerN]
| key | default | meaning |
|---|---|---|
| `thickness`, `nz` | 0.5e-3, 10 | layer thickness [m] and element rows |
| `grading` | 1 | > 1 clusters rows towards the top of the layer (resolves a thin transformed phase) |
| `youngs_modulus`, `poisson_ratio`, `thermal_expansion` | mode dependent | elastic constants, CTE [1/K] |
| `x_cracks` | all | potential channel cracks (planes normal to x): `none`, `all`, or every N elements |
| `y_cracks` | none | longitudinal cracks (normal to y); with `x_cracks` gives mud-crack networks when ny > 1 |
| `z_cracks` | none | delamination / spallation planes inside the layer (normal to z) |

The interface between two layers is always a cohesive plane (zone `interface`).

## [material], [loading]
| key | default | meaning |
|---|---|---|
| `material.reference_temperature` | 293.15 | strain-free temperature [K] |
| `loading.temperature_history` | constant T_ref | uniform temperature table [s, K] (differential expansion) |

## [transformation] and [phase] (model.type = transformation)
| key | default | meaning |
|---|---|---|
| `exponent` | 3 | growth law s^n = k t (3 = cubic, 2 = parabolic, 1 = linear) |
| `rate` | 0 | k [m^n/s] |
| `depth_at_time` | - | `s_ref, t_ref`: sets k = s_ref^n / t_ref |
| `start_time`, `max_depth` | 0, 0.95 H | |
| `pilling_bedworth_ratio` | 1.56 | V_phase / V_parent |
| `anisotropy` | 1 | a = e_inplane / e_normal; (1 + a e)^2 (1 + e) = PBR fixes e_normal |
| `phase.youngs_modulus`, `phase.poisson_ratio`, `phase.thermal_expansion` | 200e9, 0.3, 10e-6 | elastic constants of the new phase |

The transformed fraction is uniform within an element (the fraction of the element row behind the
front); the eigenstrain is (e_inplane, e_inplane, e_normal) times that fraction.

## [fracture.ZONE] - Weibull strength and toughness per zone
Zones: `bottom`, `top`, `interface` (bimaterial) or `substrate`, `phase`, `front` (transformation).

| key | default | meaning |
|---|---|---|
| `strength.distribution`, `toughness.distribution` | weibull | `weibull` or `constant` |
| `strength.mean`, `toughness.mean` | zone dependent | mean strength [Pa] and fracture energy [J/m2] |
| `strength.weibull_modulus`, `toughness.weibull_modulus` | zone dependent | m |
| `strength.threshold`, `toughness.threshold` | 0 | location parameter |
| `correlation_length` | 50e-6 | Gaussian random field length scale [m]; 0 = independent per interface element |
| `toughness_strength_correlation` | 0.5 | rho between the two fields |
| `reference_area` | 0 | weakest-link scaling for `correlation_length = 0` |
| `seed` | 1, 2, 3 | realisation |
| `fracture.flaws` | - | `x,y,z,radius,strength_factor[,toughness_factor]; ...` |

A point uses the zone it is in at the time: in the transformation model a plane switches from substrate
to front to phase properties as the front passes; damage is irreversible across the switch.

## [cohesive]
| key | default | meaning |
|---|---|---|
| `stiffness` | 1e17 | penalty K [N/m3] (~10-20 x E/h for 20-50 um elements) |
| `softening` | linear | `linear` or `exponential` |
| `mode_mixity_beta` | 0 | shear weight in the effective separation (inputs use 1: mixed-mode delamination) |
| `viscosity` | 1e-4 | viscous regularisation [s] |
| `crack_face_friction`, `residual_stiffness`, `cracked_damage` | 0, 1e-6, 0.99 | |

## [boundary]
| key | default | meaning |
|---|---|---|
| `mode` | free | `free` (statically determinate 3-2-1 pins: the strip bends freely), `roller_bottom` (u_z = 0 on the bottom), `clamped_bottom` (rigid substrate), `clamped_end` (cantilever from x = 0) |
| `y_faces` | free | width faces: `free` (narrow strip, σ_yy = 0), `generalized` (wide free plate: u_y = 0 on y = 0 and u_y = a + b(z − z_c)/H on y = W with a, b solved for, i.e. uniform in-plane strain and curvature across the width; equi-biaxial mismatch E/(1 − ν)·Δα·ΔT), `constrained` (u_y = 0 on both faces: a laterally clamped strip; σ_yy then grows with the *absolute* thermal or swelling strain of each layer) |
| `plane_strain_y` | - | deprecated; `true` means `y_faces = constrained` |

## [time], [solver], [output]
As in pelletcz: `time.end, dt_initial, dt_min, dt_max, grow, cut, max_damage_increment (0.5),
softening_dt_factor (2), initiation_target (1.02), max_steps`, plus `time.max_load_change` (0.02: the
largest change per step as a fraction of the maximum temperature excursion, or of the strip thickness for
the front). `solver.newton_tolerance (5e-4), linear_tolerance (1e-3), preconditioner (amg|jacobi),
amg_rebuild_iterations, stabilization, line_search`. `output.directory, times, every_steps,
on_crack_area_growth [m2], vtu, binary, max_vtu, labels`.
