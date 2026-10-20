#!/usr/bin/env python3
"""render.py - pictures of a stripcz run (matplotlib only; ParaView reads the same .vtu/.pvd files).

    python tools/render.py runs/bimaterial_strip                       # figures -> <run>/figures
    python tools/render.py runs/X --frames 2,5,-1 --warp 20 --out docs/images --prefix demo_

Figures
  elevation.png : side view (y = 0 face) of the deformed strip per frame, coloured by sigma_xx, with the
                  cracked cohesive faces drawn in black (channel cracks, delamination / spallation) and the
                  layer interface / transformation front marked
  plan.png      : (3D runs, ny > 1) top-surface plan view with the crack network (mud cracks)
  history.png   : cracking vs load (dT for bi-material strips, front depth for transforming strips):
                  channel cracks and spacing, crack area by plane type, delamination, curvature, stresses
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stripio  # noqa: E402


def _frames(series, spec):
    if not spec:
        n = len(series)
        return sorted(set([max(1, n // 3), max(1, 2 * n // 3), n - 1]))
    return [int(t) if int(t) >= 0 else len(series) + int(t) for t in spec.split(",")]


def _hist_at(h, t):
    return h.iloc[int(np.argmin(np.abs(h.time_s.to_numpy() - t)))]


def fig_elevation(run, frames, warp, dcut, out, field="sxx_MPa", xlim=None, zlim=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection
    man = run["manifest"]
    mode = man.get("mode", "bimaterial")
    cfg = man.get("config", {})
    bs = stripio.series(run["dir"], "bulk")
    cs = stripio.series(run["dir"], "cracks")
    idx = _frames(bs, frames)
    h = run["history"]
    data, lim = [], 0
    for i in idx:
        b = stripio.read_vtu(bs[i][1])
        c = stripio.read_vtu(cs[i][1])
        P, U, cells = b["points"], b["point_data"]["displacement_m"], b["cells"]
        y0 = P[:, 1].min()
        # y-min face of the elements touching y = y0 (nodes 0, 1, 5, 4)
        face = cells[:, [0, 1, 5, 4]]
        on = np.all(np.abs(P[face][:, :, 1] - y0) < 1e-12, axis=1)
        W = P + warp * U
        polys = W[face[on]][:, :, [0, 2]] * 1e3
        val = b["cell_data"][field][on]
        lim = max(lim, np.percentile(np.abs(val), 99))
        cq, cp = c["cells"], c["points"] + warp * c["point_data"]["displacement_m"]
        cracked = c["cell_data"]["damage"] >= dcut
        segs = []
        for q in cq[cracked]:
            pts = c["points"][q]
            sel = np.abs(pts[:, 1] - y0) < 1e-12
            if sel.sum() == 2:
                segs.append(cp[q[sel]][:, [0, 2]] * 1e3)
        data.append((bs[i][0], polys, val, segs, b["cell_data"].get("phase_fraction")[on]))
    n = len(data)
    L = float(cfg.get("geometry.length", 4e-3)) * 1e3
    if xlim is not None:  # colour scale from the zoomed window only
        lim = 0
        for (_, polys, val, _, _) in data:
            xc, zc = polys[:, :, 0].mean(1), polys[:, :, 1].mean(1)
            sel = (xc >= xlim[0]) & (xc <= xlim[1]) & ((zc >= zlim[0]) & (zc <= zlim[1]) if zlim else True)
            if sel.any():
                lim = max(lim, np.percentile(np.abs(val[sel]), 99))
    fig, axs = plt.subplots(n, 1, figsize=(11, 1.0 + (2.0 if zlim is None else 2.6) * n), squeeze=False,
                            gridspec_kw={"hspace": 0.45})
    for ax, (t, polys, val, segs, phi) in zip(axs[:, 0], data):
        pc = PolyCollection(polys, array=val, cmap="RdBu_r", clim=(-lim, lim), edgecolors="none")
        ax.add_collection(pc)
        if mode == "transformation" and phi is not None:
            # outline the transformed phase
            tr = PolyCollection(polys[phi >= 0.5], facecolors="none", edgecolors=(0.2, 0.2, 0.2, 0.25), linewidths=0.3)
            ax.add_collection(tr)
        if segs:
            ax.add_collection(LineCollection(segs, colors="k", linewidths=1.4))
        ax.autoscale_view()
        if zlim is None:
            ax.set_aspect("equal")
        r = _hist_at(h, t)
        lab = (f"ΔT = {r.dT_K:.0f} K" if mode == "bimaterial" else f"front depth = {r.front_depth_um:.1f} µm")
        dl = r.delaminated_area_mm2 / (float(cfg.get("geometry.width", 1e-4)) * 1e3)  # mm of strip length
        ax.set_title(f"t = {t:.4g} s, {lab}: {int(r.n_channel_cracks)} channel cracks, "
                     f"delaminated length {dl:.2f} mm", fontsize=9)
        ax.set_xlim(*(xlim or (-0.03 * L, 1.03 * L)))
        if zlim is not None:
            ax.set_ylim(*zlim)
        ax.set_ylabel("z [mm]")
    axs[-1, 0].set_xlabel("x [mm]")
    fig.colorbar(pc, ax=axs[:, 0].tolist(), label="σ_xx [MPa]", shrink=0.9, pad=0.01)
    kind = "bi-material strip, differential expansion" if mode == "bimaterial" else "transforming strip, PBR swelling"
    zoom = "" if zlim is None else " (zoom; vertical scale exaggerated)"
    fig.suptitle(f"{kind} - side view{zoom}, displacements x{warp:g}; black = cracked cohesive faces", fontsize=10)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_history(run, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = run["history"]
    mode = run["manifest"].get("mode", "bimaterial")
    x, xl = (h.dT_K, "ΔT [K]") if mode == "bimaterial" else (h.front_depth_um, "transformed depth s [µm]")
    fig, axs = plt.subplots(2, 2, figsize=(10.5, 6.6), constrained_layout=True)
    ax = axs[0, 0]
    ax.step(x, h.n_channel_cracks, where="post", color="k", lw=1.8)
    ax.set_xlabel(xl)
    ax.set_ylabel("channel cracks through the brittle layer")
    ax2 = ax.twinx()
    ax2.plot(x, h.mean_spacing_mm * 1e3, color="C3", lw=1.2)
    ax2.set_ylabel("mean crack spacing [µm]", color="C3")
    ax.set_title("channel cracking")
    ax = axs[0, 1]
    for col, lab in (("crack_area_x_mm2", "x-planes (channel)"), ("crack_area_z_mm2", "z-planes (spallation)"),
                     ("crack_area_layer_mm2", "layer interface"), ("crack_area_mm2", "total")):
        if h[col].max() > 0 or lab == "total":
            ax.plot(x, h[col] * 1e3, lw=2 if lab == "total" else 1.3, color="k" if lab == "total" else None, label=lab)
    ax.set_xlabel(xl)
    ax.set_ylabel("crack area [10³ µm²]")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("crack surface area by plane type")
    ax = axs[1, 0]
    ax.plot(x, h.sxx_brittle_MPa, label="brittle layer / phase", color="C0")
    ax.plot(x, h.sxx_substrate_MPa, label="substrate", color="C1")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel(xl)
    ax.set_ylabel("mean σ_xx, middle third [MPa]")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("in-plane stress")
    ax = axs[1, 1]
    ax.plot(x, h.curvature_1pm, color="C2")
    ax.set_xlabel(xl)
    ax.set_ylabel("curvature [1/m]", color="C2")
    ax3 = ax.twinx()
    ax3.plot(x, h.max_crack_depth_um, color="C4")
    ax3.set_ylabel("deepest crack below the surface [µm]", color="C4")
    ax.set_title("bending and crack penetration")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def fig_plan(run, frames, dcut, out, field="max_principal_MPa"):
    """Plan view of the top surface (3D runs, ny > 1): brittle-layer stress and the crack network."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection, PolyCollection
    cfg = run["manifest"].get("config", {})
    ss = stripio.series(run["dir"], "surface")
    cs = stripio.series(run["dir"], "cracks")
    idx = _frames(ss, frames)
    h = run["history"]
    n = len(idx)
    fig, axs = plt.subplots(1, n, figsize=(3.6 * n + 1.2, 3.8), squeeze=False)
    lim, data = 0, []
    for i in idx:
        sf = stripio.read_vtu(ss[i][1])
        c = stripio.read_vtu(cs[i][1])
        P = sf["points"]
        quads = P[sf["cells"]]
        zt = P[:, 2].max()
        top = np.all(np.abs(quads[:, :, 2] - zt) < 1e-12, axis=1)
        polys = quads[top][:, :, :2] * 1e3
        val = sf["cell_data"][field][top]
        lim = max(lim, np.percentile(np.abs(val), 99))
        cq, cp = c["cells"], c["points"]
        k = c["cell_data"]["kind"]
        cracked = (c["cell_data"]["damage"] >= dcut) & np.isin(k, [0, 1])
        segs = []
        for q in cq[cracked]:
            pts = cp[q]
            sel = np.abs(pts[:, 2] - zt) < 1e-12  # crack face reaching the top surface
            if sel.sum() == 2:
                segs.append(pts[sel][:, :2] * 1e3)
        data.append((ss[i][0], polys, val, segs))
    for ax, (t, polys, val, segs) in zip(axs[0], data):
        pc = PolyCollection(polys, array=val, cmap="RdBu_r", clim=(-lim, lim), edgecolors="none")
        ax.add_collection(pc)
        if segs:
            ax.add_collection(LineCollection(segs, colors="k", linewidths=1.0))
        ax.autoscale_view()
        ax.set_aspect("equal")
        r = _hist_at(h, t)
        lab = (f"ΔT = {r.dT_K:.0f} K" if run["manifest"].get("mode") == "bimaterial"
               else f"s = {r.front_depth_um:.1f} µm")
        cells = int(r.n_brittle_cells) if "n_brittle_cells" in r else 0
        ax.set_title(f"{lab}: {cells} cells", fontsize=9)
        ax.set_xlabel("x [mm]")
    axs[0][0].set_ylabel("y [mm]")
    fig.colorbar(pc, ax=axs[0].tolist(), label=field.replace("_MPa", " [MPa]"), shrink=0.85)
    fig.suptitle("top surface (plan view); black = cracks reaching the surface", fontsize=10)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--frames", default="")
    ap.add_argument("--warp", type=float, default=20.0)
    ap.add_argument("--dcut", type=float, default=0.99)
    ap.add_argument("--field", default="sxx_MPa")
    ap.add_argument("--out", default=None)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--xlim", type=float, nargs=2, help="zoom window x0 x1 [mm]")
    ap.add_argument("--zlim", type=float, nargs=2, help="zoom window z0 z1 [mm] (vertical scale exaggerated)")
    ap.add_argument("--no-history", action="store_true")
    a = ap.parse_args(argv)
    run = stripio.read_run(a.run_dir)
    od = a.out or os.path.join(a.run_dir, "figures")
    os.makedirs(od, exist_ok=True)
    fig_elevation(run, a.frames, a.warp, a.dcut, os.path.join(od, a.prefix + "elevation.png"), a.field,
                  a.xlim, a.zlim)
    if not a.no_history:
        fig_history(run, os.path.join(od, a.prefix + "history.png"))
    if int(run["manifest"].get("config", {}).get("mesh.ny", 1)) > 1:
        fig_plan(run, a.frames, a.dcut, os.path.join(od, a.prefix + "plan.png"))


if __name__ == "__main__":
    main()
