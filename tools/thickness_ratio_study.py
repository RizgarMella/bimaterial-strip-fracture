#!/usr/bin/env python3
"""thickness_ratio_study.py - Series A: bi-material strips of FIXED total thickness with different
fractions of the brittle top material, loaded by differential thermal expansion.

Design (inputs/bimaterial_strip.ini is the base case)
  top_fraction f = h_top / h      : 0.05, 0.1, 0.2, 0.3, 0.5, 0.7   (h = 0.5 mm fixed)
  interface toughness G_c,i       : base value (20 J/m2); --interface-toughness adds levels
  seeds                           : independent Weibull realisations of all three zones
The mesh keeps the same number of elements through the thickness (mesh.nz_total) and puts the material
interface on a node plane, so every f is resolved alike (>= 2 elements in the top layer by default).

Optional 3D (--dim 3d, inputs/bimaterial_strip_3d.ini): a 2 x 2 mm plate patch with crack planes on
every element boundary in x AND y in the top layer, so channel cracks form a mud-crack network and the
interface can delaminate in 2D patches. Same factors; planned but intended for a cluster - check the cost
with `study_common.py estimate` before running.

Metrics per run (history.csv / summary.json)
  onset dT (first fully cracked cohesive point), channel cracks through the top layer vs dT, saturated
  crack spacing (and spacing / h_top), delaminated interface fraction, maximum crack depth (penetration
  into the substrate), curvature, dissipated energy.
Per (f, G_c,i): seed mean / sd / bootstrap CI. `collect` writes derived/dataset.csv with the
dimensionless groups of features.py for fit_pysr.py (every 4th case is held out for validation).

  python tools/thickness_ratio_study.py plan [--seeds 3 --interface-toughness 5 20 80]
  python tools/study_common.py run studies/ratio --parallel 1 --threads 2      (or MPI / Slurm)
  python tools/thickness_ratio_study.py analyze studies/ratio
  python tools/thickness_ratio_study.py plot    studies/ratio
  python tools/thickness_ratio_study.py collect studies/ratio && python tools/fit_pysr.py studies/ratio
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import features as ft  # noqa: E402
import stripio  # noqa: E402
import study_common as sc  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(REPO, "inputs", "bimaterial_strip.ini")
BASE_3D = os.path.join(REPO, "inputs", "bimaterial_strip_3d.ini")  # optional 3D plate patch (expensive)
FRACTIONS = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7]
DT_STAR = [50, 100, 200, 300, 400, 500]
# base-case material data (must match inputs/bimaterial_strip.ini unless overridden)
BASE_P = dict(E_t=380e9, nu_t=0.22, a_t=8e-6, E_b=200e9, a_b=17e-6, sig_t=150e6, Gc_t=10.0, sig_i=200e6, Gc_i=20.0,
              m_t=6.0, h=0.5e-3, L=4e-3, W=0.1e-3)


def plan(a):
    three_d = a.dim == "3d"
    sdir = os.path.join(a.root, a.name or ("ratio-3d" if three_d else "ratio"))
    if a.nz_total is None:
        a.nz_total = 20 if three_d else 40
    cases, k = [], 0
    gis = a.interface_toughness or [None]
    for f, gi, seed in itertools.product(a.fractions, gis, range(1, a.seeds + 1)):
        ov = {"geometry.top_fraction": f, "geometry.total_thickness": BASE_P["h"], "mesh.nz_total": a.nz_total,
              "loading.temperature_history": sc.table([(0, 293.15), (a.t_end, 293.15 + a.dt_max)]),
              "time.end": a.t_end, "output.times": "", "output.on_crack_area_growth": 0,
              "fracture.top.seed": 100 + seed, "fracture.bottom.seed": 200 + seed, "fracture.interface.seed": 300 + seed}
        if three_d:
            ov["output.vtu"] = True  # keep the fields: a 3D run is too expensive to repeat for pictures
        if gi is not None:
            ov["fracture.interface.toughness.mean"] = gi
        cid = sc.case_id("ra", k, ov)
        cases.append(sc.Case(cid, ov, factors={"top_fraction": f, "Gc_interface_Jpm2": gi or BASE_P["Gc_i"],
                                               "seed": seed, "h_top_um": f * BASE_P["h"] * 1e6},
                             split="val" if k % 4 == 3 else "train"))
        k += 1
    base_p = dict(BASE_P, L=2e-3, W=2e-3) if three_d else BASE_P
    sc.plan_study(sdir, os.path.basename(sdir), cases,
                  {"series": "A", "dim": a.dim, "purpose": "top-layer fraction at fixed total thickness",
                   "fractions": a.fractions, "interface_toughness": gis, "seeds": a.seeds,
                   "dT_max_K": a.dt_max, "base": base_p},
                  BASE_3D if three_d else BASE, threads=a.threads, card=__doc__.split("Metrics")[0],
                  vtu=three_d)
    if three_d:
        print("3D study planned, NOT run. Size and cost: python tools/study_common.py estimate", sdir)


def run_row(d: str) -> dict:
    r = stripio.read_run(d)
    h, s = r["history"], r["summary"]
    row = dict(onset_dT_K=s.get("onset_dT_K"), n_channel_final=s.get("final_n_channel_cracks"),
               spacing_final_mm=s.get("final_mean_spacing_mm"),
               delaminated_final_mm2=s.get("final_delaminated_area_mm2"),
               max_crack_depth_um=s.get("final_max_crack_depth_um"), curvature_final_1pm=s.get("final_curvature_1pm"),
               dissipated_final_mJ=s.get("final_dissipated_energy_mJ"), wall_s=s.get("final_wall_s"))
    for v, n, ar in zip(DT_STAR, stripio.value_at(h, "dT_K", "n_channel_cracks", DT_STAR),
                        stripio.value_at(h, "dT_K", "crack_area_mm2", DT_STAR)):
        row[f"n_channel_dT{v}"] = n
        row[f"A_dT{v}_mm2"] = ar
    return row


def _boot(v, n=2000, rng=np.random.default_rng(0)):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return np.nan, np.nan
    b = rng.choice(v, (n, len(v))).mean(1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def analyze(a):
    import pandas as pd
    sdir = a.study_dir
    cases = pd.read_csv(os.path.join(sdir, "cases.csv"))
    rows = []
    for _, c in cases.iterrows():
        d = os.path.join(sdir, "runs", c.case_id)
        if stripio.run_complete(d):
            rows.append(dict(case_id=c.case_id, top_fraction=c.top_fraction, Gc_interface_Jpm2=c.Gc_interface_Jpm2,
                             seed=c.seed, h_top_um=c.h_top_um, **run_row(d)))
    df = pd.DataFrame(rows)
    dd = os.path.join(sdir, "derived")
    sc.save_table(df, os.path.join(dd, "ratio_runs.csv"), "One row per completed run", sources=list(df.get("case_id", [])))
    if df.empty:
        print("no completed runs yet")
        return
    df["spacing_over_htop"] = df.spacing_final_mm * 1e3 / df.h_top_um
    agg = []
    for (f, gi), g in df.groupby(["top_fraction", "Gc_interface_Jpm2"]):
        row = dict(top_fraction=f, Gc_interface_Jpm2=gi, h_top_um=g.h_top_um.iloc[0], n=len(g))
        for col in ["onset_dT_K", "n_channel_final", "spacing_final_mm", "spacing_over_htop", "delaminated_final_mm2",
                    "max_crack_depth_um", "curvature_final_1pm"] + [f"n_channel_dT{v}" for v in DT_STAR]:
            v = g[col].astype(float)
            row[f"{col}_mean"], row[f"{col}_sd"] = v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0
            row[f"{col}_ci_lo"], row[f"{col}_ci_hi"] = _boot(v)
        agg.append(row)
    agg = pd.DataFrame(agg)
    sc.save_table(agg, os.path.join(dd, "ratio_levels.csv"), "Seed statistics per (top fraction, interface toughness)")
    print(agg[["top_fraction", "Gc_interface_Jpm2", "n", "onset_dT_K_mean", "n_channel_final_mean",
               "spacing_over_htop_mean", "delaminated_final_mm2_mean"]].to_string(index=False))


def plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    dd, fd = os.path.join(a.study_dir, "derived"), os.path.join(a.study_dir, "figures")
    os.makedirs(fd, exist_ok=True)
    agg = pd.read_csv(os.path.join(dd, "ratio_levels.csv"))
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.8), constrained_layout=True)
    for gi, g in agg.groupby("Gc_interface_Jpm2"):
        g = g.sort_values("top_fraction")
        lab = f"G_c,i = {gi:g} J/m²"
        axs[0].errorbar(g.top_fraction, g.onset_dT_K_mean, yerr=g.onset_dT_K_sd, marker="o", capsize=2, label=lab)
        axs[1].errorbar(g.top_fraction, g.spacing_over_htop_mean, yerr=g.spacing_over_htop_sd, marker="o", capsize=2,
                        label=lab)
        axs[2].errorbar(g.top_fraction, g.delaminated_final_mm2_mean, yerr=g.delaminated_final_mm2_sd, marker="o",
                        capsize=2, label=lab)
        axs[3].plot(g.top_fraction, g.curvature_final_1pm_mean, marker="o", label=lab)
    for ax, yl in zip(axs, ["onset ΔT [K]", "crack spacing / h_top", "delaminated area [mm²]", "curvature [1/m]"]):
        ax.set_xlabel("top-layer fraction h_top / h")
        ax.set_ylabel(yl)
        ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(fd, "ratio_summary.png"), dpi=150)
    plt.close(fig)
    runs = pd.read_csv(os.path.join(dd, "ratio_runs.csv"))
    fig, ax = plt.subplots(figsize=(6, 4.2), constrained_layout=True)
    cm = plt.get_cmap("viridis")
    fs = sorted(runs.top_fraction.unique())
    for i, f in enumerate(fs):
        g = runs[runs.top_fraction == f]
        m = [g[f"n_channel_dT{v}"].mean() for v in DT_STAR]
        ax.plot(DT_STAR, m, marker="o", color=cm(i / max(1, len(fs) - 1)), label=f"f = {f:g}")
    ax.set_xlabel("ΔT [K]")
    ax.set_ylabel("channel cracks through the top layer")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(fd, "ratio_cracks_vs_dT.png"), dpi=150)
    plt.close(fig)
    print("figures ->", fd)


def collect(a):
    import pandas as pd
    sdir = a.study_dir
    sdef = stripio.read_json(os.path.join(sdir, "study.json"))
    cases = pd.read_csv(os.path.join(sdir, "cases.csv"))
    rows = []
    for _, c in cases.iterrows():
        p = dict(sdef.get("base", BASE_P), f=c.top_fraction, Gc_i=c.Gc_interface_Jpm2, dT_max=sdef["dT_max_K"])
        row = dict(case_id=c.case_id, split=c.split, **ft.groups_A(p))
        d = os.path.join(sdir, "runs", c.case_id)
        if stripio.run_complete(d):
            r = stripio.read_run(d)
            row.update(status="complete", **ft.targets_A(r["summary"], r["history"], p))
        else:
            row["status"] = "missing"
        rows.append(row)
    df = pd.DataFrame(rows)
    sc.save_table(df, os.path.join(sdir, "derived", "dataset.csv"), "Series A features and targets",
                  sources=list(df.case_id))
    print(df.status.value_counts().to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("plan")
    p.add_argument("--root", default="studies")
    p.add_argument("--name", default=None)
    p.add_argument("--dim", choices=["2d", "3d"], default="2d",
                   help="3d: square plate patch with x and y crack planes (mud-crack networks); expensive")
    p.add_argument("--fractions", type=float, nargs="*", default=FRACTIONS)
    p.add_argument("--interface-toughness", type=float, nargs="*")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--nz-total", type=int, default=None, help="elements through the thickness (2d: 40, 3d: 20)")
    p.add_argument("--dt-max", type=float, default=500.0, help="final temperature rise [K]")
    p.add_argument("--t-end", type=float, default=500.0)
    p.add_argument("--threads", type=int, default=0)
    for name in ("analyze", "plot", "collect"):
        q = sp.add_parser(name)
        q.add_argument("study_dir")
    a = ap.parse_args()
    {"plan": plan, "analyze": analyze, "plot": plot, "collect": collect}[a.cmd](a)


if __name__ == "__main__":
    main()
