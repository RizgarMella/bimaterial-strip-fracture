#!/usr/bin/env python3
"""mesh_convergence.py - discretisation study on CRACK SURFACE AREA for either series.

Levels refine the in-plane and through-thickness element counts together (h = L/nx with nz scaled
alike), keeping crack planes at every element boundary. Families:
  field     Weibull properties from correlated fields of FIXED physical correlation length in every zone
            (refining samples the same material: the convergence test proper)
  element   independent draws per interface element (correlation_length = 0) with weakest-link scaling
  constant  no scatter (deterministic)
Metrics: crack area and channel-crack count at load levels (dT* for series A, front depth s* for B),
onset load, delaminated area, dissipated energy; zeta = l_cz / h from the manifest.
Analysis: seed mean / sd / sem per level, Celik et al. (2008) GCI on the three finest levels with
Richardson extrapolation, seed-noise test; the coarsest level within --tol of the extrapolated crack area
is written to derived/convergence.json.

  python tools/mesh_convergence.py plan --series A [--seeds 3 --levels 40 60 80 120 160]
  python tools/mesh_convergence.py plan --series A --dim 3d     # optional 3D plate patch, nx = ny = 20..60
  python tools/study_common.py run studies/meshconv-A --parallel 1 --threads 2
  python tools/mesh_convergence.py analyze studies/meshconv-A
  python tools/mesh_convergence.py plot    studies/meshconv-A
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stripio  # noqa: E402
import study_common as sc  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASES = {"A": os.path.join(REPO, "inputs", "bimaterial_strip.ini"),
         "B": os.path.join(REPO, "inputs", "transforming_strip.ini"),
         "A3d": os.path.join(REPO, "inputs", "bimaterial_strip_3d.ini"),
         "B3d": os.path.join(REPO, "inputs", "transforming_strip_3d.ini")}
LEVELS_3D = [20, 30, 40, 60]  # nx = ny on the 2 x 2 mm patch (100 .. 33 um)
LOADS = {"A": ("dT_K", [100, 200, 300, 400, 500]), "B": ("front_depth_um", [10, 20, 50, 100])}
ZONES = {"A": ("top", "bottom", "interface"), "B": ("phase", "substrate", "front")}
NX_BASE = 80


def level_overrides(series: str, nx: int, three_d: bool = False) -> dict:
    if three_d:  # 2 x 2 mm patch, base nx = ny = 40 (50 um), same through-thickness scaling
        f = nx / 40
        if series == "A":
            return {"mesh.nx": nx, "mesh.ny": nx, "mesh.nz_total": max(4, int(round(20 * f)))}
        return {"mesh.nx": nx, "mesh.ny": nx, "layer0.nz": max(4, int(round(24 * f)))}
    f = nx / NX_BASE
    if series == "A":
        return {"mesh.nx": nx, "mesh.nz_total": max(4, int(round(20 * f)))}
    return {"mesh.nx": nx, "layer0.nz": max(4, int(round(24 * f)))}


def plan(a):
    three_d = a.dim == "3d"
    sdir = os.path.join(a.root, a.name or f"meshconv-{a.series}" + ("-3d" if three_d else ""))
    levels = a.levels or (LEVELS_3D if three_d else [40, 60, 80, 120, 160])
    cases, k = [], 0
    for fam in a.families:
        for nx in levels:
            for seed in range(1, (1 if fam == "constant" else a.seeds) + 1):
                ov = {**level_overrides(a.series, nx, three_d), "output.times": "", "output.on_crack_area_growth": 0}
                for z in ZONES[a.series]:
                    ov[f"fracture.{z}.seed"] = 1000 * seed + ZONES[a.series].index(z)
                    if fam == "element":
                        ov[f"fracture.{z}.correlation_length"] = 0.0
                        ov[f"fracture.{z}.reference_area"] = (4e-3 / NX_BASE) * (50e-6 if three_d else 0.1e-3)
                    elif fam == "constant":
                        ov[f"fracture.{z}.strength.distribution"] = "constant"
                        ov[f"fracture.{z}.toughness.distribution"] = "constant"
                cid = sc.case_id("mc", k, ov)
                h_um = (2e-3 if three_d else 4e-3) / nx * 1e6
                cases.append(sc.Case(cid, ov, factors={"family": fam, "nx": nx, "seed": seed, "h_um": h_um}))
                k += 1
    sc.plan_study(sdir, os.path.basename(sdir), cases,
                  {"series": a.series, "dim": a.dim, "purpose": "mesh convergence on crack area", "levels": levels,
                   "families": a.families, "seeds": a.seeds},
                  BASES[a.series + ("3d" if three_d else "")], threads=a.threads, card=__doc__.split("Analysis")[0])
    if three_d:
        print("3D study planned, NOT run. Size and cost: python tools/study_common.py estimate", sdir)


def celik_gci(h, phi, fs=1.25):
    h1, h2, h3 = h
    f1, f2, f3 = phi
    out = dict(phi1=f1, phi2=f2, phi3=f3, p=np.nan, phi_ext=np.nan, gci21=np.nan, kind="undetermined")
    e21, e32 = f2 - f1, f3 - f2
    if not np.all(np.isfinite([f1, f2, f3])) or e21 == 0 or e32 == 0 or f1 == 0:
        return out
    r21, r32 = h2 / h1, h3 / h2
    s = np.sign(e32 / e21)
    p = abs(np.log(abs(e32 / e21))) / np.log(r21)
    for _ in range(200):
        num, den = r21 ** p - s, r32 ** p - s
        if num <= 0 or den <= 0:
            return {**out, "kind": "oscillatory-diverging"}
        pn = abs(np.log(abs(e32 / e21)) + np.log(num / den)) / np.log(r21)
        if abs(pn - p) < 1e-10:
            break
        p = pn
    fext = (r21 ** p * f1 - f2) / (r21 ** p - 1)
    ea = abs((f1 - f2) / f1)
    out.update(p=p, phi_ext=fext, gci21=fs * ea / (r21 ** p - 1), kind="monotone" if s > 0 else "oscillatory")
    return out


def analyze(a):
    import pandas as pd
    sdir = a.study_dir
    series = stripio.read_json(os.path.join(sdir, "study.json"))["series"]
    lcol, levels = LOADS[series]
    cases = pd.read_csv(os.path.join(sdir, "cases.csv"))
    rows = []
    for _, c in cases.iterrows():
        d = os.path.join(sdir, "runs", c.case_id)
        if not stripio.run_complete(d):
            continue
        r = stripio.read_run(d)
        h, s, m = r["history"], r["summary"], r["manifest"]
        row = dict(case_id=c.case_id, family=c.family, nx=c.nx, seed=c.seed, h_um=c.h_um,
                   zeta=m.get("derived", {}).get("zone1_zeta_min"),
                   crack_area_final_mm2=s.get("final_crack_area_mm2"), n_channel_final=s.get("final_n_channel_cracks"),
                   delaminated_final_mm2=s.get("final_delaminated_area_mm2"),
                   onset=s.get("onset_dT_K") if series == "A" else s.get("onset_front_depth_um"))
        for v, ar, n in zip(levels, stripio.value_at(h, lcol, "crack_area_mm2", levels),
                            stripio.value_at(h, lcol, "n_channel_cracks", levels)):
            row[f"A_{v}_mm2"] = ar
            row[f"n_{v}"] = n
        rows.append(row)
    per = pd.DataFrame(rows)
    dd = os.path.join(sdir, "derived")
    sc.save_table(per, os.path.join(dd, "meshconv_runs.csv"), "One row per completed run",
                  sources=list(per.get("case_id", [])))
    if per.empty:
        print("no completed runs yet")
        return
    metrics = [c for c in per.columns if c.startswith(("A_", "n_"))] + [
        "onset", "crack_area_final_mm2", "n_channel_final", "delaminated_final_mm2"]
    agg = []
    for (fam, nx), g in per.groupby(["family", "nx"]):
        row = dict(family=fam, nx=nx, h_um=g.h_um.iloc[0], zeta=g.zeta.iloc[0], n_seeds=len(g))
        for mname in metrics:
            v = g[mname].astype(float)
            row[f"{mname}_mean"] = v.mean()
            row[f"{mname}_sd"] = v.std(ddof=1) if v.count() > 1 else 0.0
            row[f"{mname}_sem"] = row[f"{mname}_sd"] / np.sqrt(max(1, v.count()))
        agg.append(row)
    agg = pd.DataFrame(agg)
    sc.save_table(agg, os.path.join(dd, "meshconv_levels.csv"), "Seed statistics per (family, level)")
    gci = []
    for fam, g in agg.groupby("family"):
        g = g.sort_values("h_um")
        if len(g) < 3:
            continue
        f3 = g.iloc[:3]
        for mname in metrics:
            res = celik_gci(tuple(f3.h_um), tuple(f3[f"{mname}_mean"]))
            sem12 = float(np.hypot(f3[f"{mname}_sem"].iloc[0], f3[f"{mname}_sem"].iloc[1]))
            res.update(family=fam, metric=mname,
                       within_seed_noise=bool(abs(res["phi1"] - res["phi2"]) <= 2 * sem12) if sem12 > 0 else None)
            gci.append(res)
    gci = pd.DataFrame(gci)
    sc.save_table(gci, os.path.join(dd, "meshconv_gci.csv"), "Celik GCI on the three finest levels")
    key = "crack_area_final_mm2"
    rec = {"series": series, "tol": a.tol, "metric": key, "candidates": []}
    for fam, g in agg.groupby("family"):
        g = g.sort_values("h_um")
        gr = gci[(gci.family == fam) & (gci.metric == key)] if len(gci) else gci
        ref = gr.phi_ext.iloc[0] if len(gr) and np.isfinite(gr.phi_ext.iloc[0]) else g[f"{key}_mean"].iloc[0]
        g = g.assign(rel_err=(g[f"{key}_mean"] - ref).abs() / abs(ref) if ref else np.nan)
        ok = g[g.rel_err <= a.tol]
        pick = ok.sort_values("h_um", ascending=False).iloc[0] if len(ok) else g.iloc[0]
        rec["candidates"].append(dict(family=fam, nx=int(pick.nx), h_um=float(pick.h_um), rel_err=float(pick.rel_err),
                                      reference=float(ref)))
    with open(os.path.join(dd, "convergence.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps(rec, indent=1))


def plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    dd, fd = os.path.join(a.study_dir, "derived"), os.path.join(a.study_dir, "figures")
    os.makedirs(fd, exist_ok=True)
    series = stripio.read_json(os.path.join(a.study_dir, "study.json"))["series"]
    lcol, levels = LOADS[series]
    agg = pd.read_csv(os.path.join(dd, "meshconv_levels.csv"))
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for fam, g in agg.groupby("family"):
        g = g.sort_values("h_um")
        axs[0].errorbar(g.h_um, g.crack_area_final_mm2_mean, yerr=g.crack_area_final_mm2_sem, marker="o", capsize=2,
                        label=fam)
        axs[1].errorbar(g.h_um, g.n_channel_final_mean, yerr=g.n_channel_final_sem, marker="o", capsize=2, label=fam)
    for ax, yl in zip(axs, ["final crack area [mm²]", "channel cracks"]):
        ax.set_xscale("log")
        ax.set_xlabel("element size h [µm]")
        ax.set_ylabel(yl)
        ax.legend(frameon=False)
    fig.savefig(os.path.join(fd, "meshconv_vs_h.png"), dpi=150)
    plt.close(fig)
    print("figures ->", fd)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("plan")
    p.add_argument("--series", choices=["A", "B"], required=True)
    p.add_argument("--root", default="studies")
    p.add_argument("--name", default=None)
    p.add_argument("--levels", type=int, nargs="*", default=None, help="2d: nx levels; 3d: nx = ny levels")
    p.add_argument("--dim", choices=["2d", "3d"], default="2d")
    p.add_argument("--families", nargs="*", default=["field", "element", "constant"])
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--threads", type=int, default=0)
    for name in ("analyze", "plot"):
        q = sp.add_parser(name)
        q.add_argument("study_dir")
        q.add_argument("--tol", type=float, default=0.10)
    a = ap.parse_args()
    {"plan": plan, "analyze": analyze, "plot": plot}[a.cmd](a)


if __name__ == "__main__":
    main()
