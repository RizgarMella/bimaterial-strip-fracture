"""features.py - dimensionless groups and targets for the two strip series (shared by the series
`collect` steps, fit_pysr.py and any predictor), following peridynamic-pellet-fracture/features.py.

Series A - bi-material strip, differential thermal expansion (thickness_ratio_study.py)
  f          = h_top / h                              top-layer fraction at fixed total thickness h
  Pi_mis     = E_t' |a_b - a_t| dT_max / sigma_t      mismatch stress of a thin film / mean top strength,
               E' = E/(1 - nu) (biaxial modulus)
  Pi_h       = h_top / l_ch,t,   l_ch = E G_c / sigma^2    film thickness / Irwin length
  Pi_E       = E_t / E_b                               stiffness ratio
  Pi_Gi      = G_c,interface / G_c,top                 interface / film toughness ratio
  Pi_Si      = sigma_interface / sigma_top
  m_top                                                Weibull modulus of the top layer
  targets: log10_spacing_over_htop (saturated channel-crack spacing / film thickness),
           Pi_onset (Pi_mis at the first crack), delam_fraction (delaminated / strip area)

Series B - transforming strip (transformation_study.py)
  Pi_swell   = E_p' |e_inplane| / sigma_p             in-plane swelling stress of the phase / its strength
  pbr, anisotropy                                     Pilling-Bedworth ratio and normal/in-plane split
  Pi_s       = s_end / l_ch,p                          final phase thickness / Irwin length of the phase
  Pi_sub     = s_end / H                               phase fraction of the strip at the end
  Pi_E       = E_p / E_s
  Pi_Gf      = G_c,front / G_c,phase
  m_phase
  targets: log10_spacing_over_s (channel spacing / phase thickness), log10_sc_over_lch (critical
           depth at first crack / l_ch,p), delam_fraction
"""
from __future__ import annotations

import math

FEATURES_A = ["f", "Pi_mis", "Pi_h", "Pi_E", "Pi_Gi", "Pi_Si", "m_top"]
TARGETS_A = ["log10_spacing_over_htop", "Pi_onset", "delam_fraction"]
FEATURES_B = ["Pi_swell", "pbr", "anisotropy", "Pi_s", "Pi_sub", "Pi_E", "Pi_Gf", "m_phase"]
TARGETS_B = ["log10_spacing_over_s", "log10_sc_over_lch", "delam_fraction"]


def for_study(sdef: dict):
    if sdef.get("series") == "B":
        return list(FEATURES_B), list(TARGETS_B)
    return list(FEATURES_A), list(TARGETS_A)


def _log10(v):
    return math.log10(v) if v and v > 0 and math.isfinite(v) else float("nan")


def groups_A(p: dict) -> dict:
    """p: E_t, nu_t, a_t, E_b, a_b, sig_t, Gc_t, sig_i, Gc_i, m_t, h, f, dT_max"""
    Ep = p["E_t"] / (1 - p["nu_t"])
    lch = p["E_t"] * p["Gc_t"] / p["sig_t"] ** 2
    return dict(f=p["f"], Pi_mis=Ep * abs(p["a_b"] - p["a_t"]) * p["dT_max"] / p["sig_t"],
                Pi_h=p["f"] * p["h"] / lch, Pi_E=p["E_t"] / p["E_b"], Pi_Gi=p["Gc_i"] / p["Gc_t"],
                Pi_Si=p["sig_i"] / p["sig_t"], m_top=p["m_t"], l_ch_top_m=lch)


def targets_A(summary: dict, history, p: dict) -> dict:
    h_top = p["f"] * p["h"]
    sp = summary.get("final_mean_spacing_mm")
    n = summary.get("final_n_channel_cracks") or 0
    dT_on = summary.get("onset_dT_K")
    Ep = p["E_t"] / (1 - p["nu_t"])
    pi_on = Ep * abs(p["a_b"] - p["a_t"]) * dT_on / p["sig_t"] if dT_on and math.isfinite(dT_on) else float("nan")
    area = p["L"] * p["W"]
    return dict(log10_spacing_over_htop=_log10(sp * 1e-3 / h_top) if n else float("nan"), Pi_onset=pi_on,
                delam_fraction=(summary.get("final_delaminated_area_mm2") or 0) * 1e-6 / area,
                n_channel=n, spacing_mm=sp)


def groups_B(p: dict) -> dict:
    """p: E_p, nu_p, E_s, sig_p, Gc_p, Gc_f, m_p, pbr, anis, e_inplane, s_end, H"""
    Ep = p["E_p"] / (1 - p["nu_p"])
    lch = p["E_p"] * p["Gc_p"] / p["sig_p"] ** 2
    return dict(Pi_swell=Ep * abs(p["e_inplane"]) / p["sig_p"], pbr=p["pbr"], anisotropy=p["anis"],
                Pi_s=p["s_end"] / lch, Pi_sub=p["s_end"] / p["H"], Pi_E=p["E_p"] / p["E_s"],
                Pi_Gf=p["Gc_f"] / p["Gc_p"], m_phase=p["m_p"], l_ch_phase_m=lch)


def targets_B(summary: dict, history, p: dict) -> dict:
    lch = p["E_p"] * p["Gc_p"] / p["sig_p"] ** 2
    n = summary.get("final_n_channel_cracks") or 0
    sp = summary.get("final_mean_spacing_mm")
    s_end = (summary.get("final_front_depth_um") or 0) * 1e-6
    sc_um = summary.get("onset_front_depth_um")
    area = p["L"] * p["W"]
    return dict(log10_spacing_over_s=_log10(sp * 1e-3 / s_end) if n and s_end > 0 else float("nan"),
                log10_sc_over_lch=_log10(sc_um * 1e-6 / lch) if sc_um and math.isfinite(sc_um) else float("nan"),
                delam_fraction=(summary.get("final_delaminated_area_mm2") or 0) * 1e-6 / area,
                n_channel=n, spacing_mm=sp, onset_depth_um=sc_um)
