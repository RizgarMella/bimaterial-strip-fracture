#!/usr/bin/env python3
"""transformation_study.py - Series B: a single-material strip whose top transforms, front depth
s(t) = (k t)^(1/n) (cubic by default), into a more brittle phase that swells by the Pilling-Bedworth
ratio. The swelling - not a thermal mismatch - drives the stresses.

Factors (inputs/transforming_strip.ini is the base case)
  pbr           Pilling-Bedworth ratio V_phase/V_parent      : 0.95, 1.28, 1.56, 1.79
  anisotropy    e_inplane / e_normal of the swelling          : 0.002, 0.005, 0.02, 0.1
  Gc_phase      mean fracture energy of the phase [J/m2]      : 2, 5, 20
  Gc_front      mean fracture energy of the front [J/m2]      : 5, 20, 80   (spallation resistance)
  exponent      growth-law exponent n (s^n = k t)             : 3 (cubic); --exponents 2 3 adds parabolic
  seed          Weibull realisations of substrate / phase / front (separate distributions)
Design 'oat' (default): one factor at a time around (1.56, 0.005, 5, 20, n = 3); 'full': full factorial.
The growth constant is set by depth_at_time (s = 100 um at t = 1e4 s by default) for every n.

Metrics per run: critical front depth at the first crack, channel cracks through the phase and their
spacing vs front depth, spallation/delamination area (cracked horizontal planes, incl. the front), crack
depth into the substrate, strip curvature, mean in-plane stress of phase and substrate.

  python tools/transformation_study.py plan [--design oat --seeds 2]
  python tools/study_common.py run studies/transform --parallel 1 --threads 2
  python tools/transformation_study.py analyze studies/transform
  python tools/transformation_study.py plot    studies/transform
  python tools/transformation_study.py collect studies/transform && python tools/fit_pysr.py studies/transform
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
BASE = os.path.join(REPO, "inputs", "transforming_strip.ini")
PBR = [0.95, 1.28, 1.56, 1.79]
ANIS = [0.002, 0.005, 0.02, 0.1]
GC_PHASE = [2.0, 5.0, 20.0]
GC_FRONT = [5.0, 20.0, 80.0]
BASE_POINT = (1.56, 0.005, 5.0, 20.0)
DEPTH_STAR_UM = [5, 10, 20, 50, 100]
BASE_P = dict(E_p=200e9, nu_p=0.30, E_s=95e9, sig_p=150e6, Gc_f=20.0, m_p=6.0, H=0.5e-3, L=4e-3, W=0.1e-3)


def strains(pbr, a):
    e = pbr ** (1 / 3) - 1
    for _ in range(100):
        f = (1 + a * e) ** 2 * (1 + e) - pbr
        df = 2 * a * (1 + a * e) * (1 + e) + (1 + a * e) ** 2
        e -= f / df
    return e, a * e


def design(kind, exponents):
    if kind == "full":
        pts = list(itertools.product(PBR, ANIS, GC_PHASE, GC_FRONT))
    else:
        b = BASE_POINT
        pts = sorted({b} | {(p, b[1], b[2], b[3]) for p in PBR} | {(b[0], x, b[2], b[3]) for x in ANIS} |
                     {(b[0], b[1], g, b[3]) for g in GC_PHASE} | {(b[0], b[1], b[2], g) for g in GC_FRONT})
    return [pt + (n,) for pt in pts for n in exponents]


def plan(a):
    sdir = os.path.join(a.root, a.name)
    cases, k = [], 0
    for (pbr, anis, gc, gf, n), seed in itertools.product(design(a.design, a.exponents), range(1, a.seeds + 1)):
        ov = {"transformation.pilling_bedworth_ratio": pbr, "transformation.anisotropy": anis,
              "transformation.exponent": n, "transformation.depth_at_time": f"{a.depth:g},{a.t_end:g}",
              "fracture.phase.toughness.mean": gc, "fracture.front.toughness.mean": gf, "time.end": a.t_end,
              "output.times": "",
              "output.on_crack_area_growth": 0, "fracture.substrate.seed": 100 + seed,
              "fracture.phase.seed": 200 + seed, "fracture.front.seed": 300 + seed}
        cid = sc.case_id("tr", k, ov)
        cases.append(sc.Case(cid, ov, factors={"pbr": pbr, "anisotropy": anis, "Gc_phase_Jpm2": gc, "Gc_front_Jpm2": gf,
                                               "exponent": n, "seed": seed}, split="val" if k % 4 == 3 else "train"))
        k += 1
    sc.plan_study(sdir, a.name, cases, {"series": "B", "purpose": "surface transformation with PBR swelling",
                                        "design": a.design, "exponents": a.exponents, "seeds": a.seeds,
                                        "depth_end_m": a.depth, "t_end_s": a.t_end, "base": BASE_P},
                  BASE, threads=a.threads, card=__doc__.split("Metrics")[0])


def run_row(d):
    r = stripio.read_run(d)
    h, s = r["history"], r["summary"]
    row = dict(onset_depth_um=s.get("onset_front_depth_um"), final_depth_um=s.get("final_front_depth_um"),
               n_channel_final=s.get("final_n_channel_cracks"), spacing_final_mm=s.get("final_mean_spacing_mm"),
               delaminated_final_mm2=s.get("final_delaminated_area_mm2"),
               max_crack_depth_um=s.get("final_max_crack_depth_um"), curvature_final_1pm=s.get("final_curvature_1pm"),
               dissipated_final_mJ=s.get("final_dissipated_energy_mJ"), wall_s=s.get("final_wall_s"))
    for v, n, dl in zip(DEPTH_STAR_UM, stripio.value_at(h, "front_depth_um", "n_channel_cracks", DEPTH_STAR_UM),
                        stripio.value_at(h, "front_depth_um", "delaminated_area_mm2", DEPTH_STAR_UM)):
        row[f"n_channel_s{v}"] = n
        row[f"delam_s{v}_mm2"] = dl
    return row


def analyze(a):
    import pandas as pd
    sdir = a.study_dir
    cases = pd.read_csv(os.path.join(sdir, "cases.csv"))
    rows = []
    for _, c in cases.iterrows():
        d = os.path.join(sdir, "runs", c.case_id)
        if stripio.run_complete(d):
            rows.append(dict(case_id=c.case_id, pbr=c.pbr, anisotropy=c.anisotropy, Gc_phase_Jpm2=c.Gc_phase_Jpm2,
                             Gc_front_Jpm2=c.Gc_front_Jpm2, exponent=c.exponent, seed=c.seed, **run_row(d)))
    df = pd.DataFrame(rows)
    dd = os.path.join(sdir, "derived")
    sc.save_table(df, os.path.join(dd, "transform_runs.csv"), "One row per completed run",
                  sources=list(df.get("case_id", [])))
    if df.empty:
        print("no completed runs yet")
        return
    keys = ["pbr", "anisotropy", "Gc_phase_Jpm2", "Gc_front_Jpm2", "exponent"]
    agg = []
    for key, g in df.groupby(keys):
        row = dict(zip(keys, key), n=len(g))
        for col in ["onset_depth_um", "n_channel_final", "spacing_final_mm", "delaminated_final_mm2",
                    "max_crack_depth_um", "curvature_final_1pm"]:
            v = g[col].astype(float)
            row[f"{col}_mean"], row[f"{col}_sd"] = v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0
        agg.append(row)
    agg = pd.DataFrame(agg)
    sc.save_table(agg, os.path.join(dd, "transform_levels.csv"), "Seed statistics per factor combination")
    print(agg.to_string(index=False))


def plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    dd, fd = os.path.join(a.study_dir, "derived"), os.path.join(a.study_dir, "figures")
    os.makedirs(fd, exist_ok=True)
    agg = pd.read_csv(os.path.join(dd, "transform_levels.csv"))
    b = BASE_POINT
    names = ["pbr", "anisotropy", "Gc_phase_Jpm2", "Gc_front_Jpm2"]
    base = dict(zip(names, b))
    fig, axs = plt.subplots(4, 3, figsize=(13, 13), constrained_layout=True)
    for row, fac in enumerate(names):
        others = [o for o in names if o != fac]
        sel = np.ones(len(agg), bool)
        for o in others:
            sel &= np.isclose(agg[o], base[o])
        g = agg[sel].sort_values(fac)
        for col, (m, lab) in enumerate((("onset_depth_um", "critical depth [µm]"),
                                        ("n_channel_final", "channel cracks"),
                                        ("delaminated_final_mm2", "delaminated area [mm²]"))):
            ax = axs[row, col]
            for n, gg in g.groupby("exponent"):
                ax.errorbar(gg[fac], gg[f"{m}_mean"], yerr=gg[f"{m}_sd"], marker="o", capsize=2, label=f"n = {n:g}")
            ax.set_xlabel(fac)
            ax.set_ylabel(lab)
            if fac in ("anisotropy", "Gc_front_Jpm2", "Gc_phase_Jpm2"):
                ax.set_xscale("log")
            ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(fd, "transform_oat.png"), dpi=150)
    plt.close(fig)
    print("figures ->", fd)


def collect(a):
    import pandas as pd
    sdir = a.study_dir
    sdef = stripio.read_json(os.path.join(sdir, "study.json"))
    cases = pd.read_csv(os.path.join(sdir, "cases.csv"))
    rows = []
    for _, c in cases.iterrows():
        en, ep = strains(c.pbr, c.anisotropy)
        p = dict(sdef.get("base", BASE_P), pbr=c.pbr, anis=c.anisotropy, Gc_p=c.Gc_phase_Jpm2, Gc_f=c.Gc_front_Jpm2,
                 e_inplane=ep,
                 s_end=sdef["depth_end_m"])
        row = dict(case_id=c.case_id, split=c.split, **ft.groups_B(p))
        d = os.path.join(sdir, "runs", c.case_id)
        if stripio.run_complete(d):
            r = stripio.read_run(d)
            row.update(status="complete", **ft.targets_B(r["summary"], r["history"], p))
        else:
            row["status"] = "missing"
        rows.append(row)
    df = pd.DataFrame(rows)
    sc.save_table(df, os.path.join(sdir, "derived", "dataset.csv"), "Series B features and targets",
                  sources=list(df.case_id))
    print(df.status.value_counts().to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("plan")
    p.add_argument("--root", default="studies")
    p.add_argument("--name", default="transform")
    p.add_argument("--design", choices=["oat", "full"], default="oat")
    p.add_argument("--exponents", type=float, nargs="*", default=[3.0])
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--depth", type=float, default=100e-6, help="front depth at t_end [m]")
    p.add_argument("--t-end", type=float, default=1e4)
    p.add_argument("--threads", type=int, default=0)
    for name in ("analyze", "plot", "collect"):
        q = sp.add_parser(name)
        q.add_argument("study_dir")
    a = ap.parse_args()
    {"plan": plan, "analyze": analyze, "plot": plot, "collect": collect}[a.cmd](a)


if __name__ == "__main__":
    main()
