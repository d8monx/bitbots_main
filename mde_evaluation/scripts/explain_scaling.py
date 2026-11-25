"""Picture that explains, with the real numbers of one recording, how DA3 is scaled with the floor.

Step 1: the floor 0.5-2.5 m in front of the robot (green). For each of its pixels the camera pose (TF) gives
        the depth of the ground plane, the same ray-meets-floor geometry as the Bit-Bots IPM.
Step 2: per pixel, plane depth / DA3 depth. The median over the green area is the scale factor of this image.
Step 3: the whole DA3 depth map is multiplied by this one factor, then the ball distance is read from it.

Usage: python scripts/explain_scaling.py [bag]   (default mde_s1_performance_2m)
       -> results/real_26-11-04/skalierung_erklaert.png
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from eval_real import (  # noqa: E402
    OUT, add_da3, floor_scale, ipm_distance, load_frames, object_distance, rays, true_distance)


def de(x, n=2):
    return f"{x:.{n}f}".replace(".", ",")


def main():
    bag = sys.argv[1] if len(sys.argv) > 1 else "mde_s1_performance_2m"
    data = [d for d in load_frames() if d["bag"] == bag][:1]
    add_da3(data)
    d = data[0]
    K, T, gt, near = d["K"], d["T"], d["gt"], d["near"]
    i_obj, obj, core, shape = next(o for o in d["objs"] if o[1]["kind"] == "ball")
    raw = {"DA3 Base": d["DA3-BASE"], "DA3 Metric-Large": d["DA3METRIC-LARGE"]}
    factor = {k: floor_scale(v, gt, near) for k, v in raw.items()}

    # Ground position (x ahead of base_footprint) of every pixel on the plane
    pts = (rays(K) * gt[..., None]) @ T[:3, :3].T + T[:3, 3]
    ahead = pts[..., 0]

    fig = plt.figure(figsize=(17, 10.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1], width_ratios=[1.25, 1])
    fig.suptitle(f"So wird DA3 skaliert: Messung {bag.removeprefix('mde_s1_')}, {d['file']}", fontsize=15,
                 fontweight="bold")

    # Step 1: image with the scale region and sample points
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(d["rgb"])
    overlay = np.zeros((*near.shape, 4))
    overlay[near] = (0.1, 0.9, 0.2, 0.35)
    ax.imshow(overlay)
    v, u = np.nonzero(shape)
    ax.add_patch(plt.Circle((u.mean(), v.mean()), (u.max() - u.min()) / 2 + 4, fill=False, color="yellow", lw=2))
    ax.text(u.mean() + 14, v.mean() - 10, "Ball", color="yellow", fontsize=11, fontweight="bold")
    col = int(K["cx"]) - 70  # a bit left of the centre, away from the ball
    free = near.copy()
    for *_x, _core, obj_shape in d["objs"]:
        free &= ~obj_shape
    samples = []
    for target in (0.75, 1.25, 1.75, 2.25):
        rows = np.nonzero(free[:, col])[0]
        r = rows[np.argmin(np.abs(ahead[rows, col] - target))]
        samples.append((r, col))
    for n, (r, c) in enumerate(samples, start=1):
        ax.plot(c, r, "o", color="white", ms=9, mec="black")
        ax.text(c + 10, r + 4, str(n), color="white", fontsize=12, fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.6, pad=1))
    ax.set_title("Schritt 1: Boden 0,5–2,5 m vor dem Roboter (grün). Dort kennen wir die Tiefe aus der Kamerapose.",
                 fontsize=11)
    ax.axis("off")

    # Sample table
    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    head = ["Punkt", "Boden\nvor Roboter", "Tiefe aus\nKamerapose", "DA3 Base\nroh", "Verhältnis", "DA3 Metric\nroh",
            "Verhältnis"]
    cells = []
    for n, (r, c) in enumerate(samples, start=1):
        g, b, m = gt[r, c], raw["DA3 Base"][r, c], raw["DA3 Metric-Large"][r, c]
        cells.append([str(n), f"{de(ahead[r, c])} m", f"{de(g)} m", de(b, 3), f"× {de(g / b)}", f"{de(m)} m",
                      f"× {de(g / m)}"])
    cells.append(["Median\ngrün", "", "", "", f"× {de(factor['DA3 Base'])}", "", f"× {de(factor['DA3 Metric-Large'])}"])
    tab = ax.table(cellText=cells, colLabels=head, loc="upper center", cellLoc="center",
                   colWidths=[0.1, 0.15, 0.15, 0.13, 0.14, 0.15, 0.14])
    tab.auto_set_font_size(False)
    tab.set_fontsize(10.5)
    tab.scale(1, 2.6)
    for c in (4, 6):
        tab[len(cells), c].set_facecolor("#C6EFCE")
        tab[len(cells), c].get_text().set_fontweight("bold")
    ax.set_title("Schritt 2: pro Pixel Tiefe aus Kamerapose ÷ DA3-Tiefe.\n"
                 f"Der Median über alle {near.sum()} grünen Pixel ist der Faktor für dieses Bild.", fontsize=11)
    ax.text(0.02, 0.02, "DA3 Base liefert nur relative Werte (keine Meter), daher der große Faktor.\n"
            "DA3 Metric-Large liefert schon Meter, der Faktor korrigiert nur seine Abweichung.\n"
            "Tiefe = Abstand entlang der Blickrichtung der Kamera (z-Tiefe).",
            transform=ax.transAxes, fontsize=10, va="bottom")

    # Step 2 graphically: ratio over the green area
    ax = fig.add_subplot(gs[1, 0])
    for name, color in (("DA3 Base", "tab:blue"), ("DA3 Metric-Large", "tab:orange")):
        ratio = gt[near] / raw[name][near]
        x = ahead[near]
        sel = np.random.default_rng(0).choice(ratio.size, min(3000, ratio.size), replace=False)
        rel = ratio[sel] / factor[name]
        ax.scatter(x[sel], rel, s=3, alpha=0.3, color=color, label=f"{name} (Faktor × {de(factor[name])})")
    ax.axhline(1, color="black", lw=1.5)
    ax.set_xlabel("Bodenpunkt vor dem Roboter (m)")
    ax.set_ylabel("Verhältnis ÷ Faktor  (1 = genau der Median)")
    ax.set_ylim(0.8, 1.2)
    ax.legend(loc="upper right", markerscale=4)
    ax.set_title("Wie gleichmäßig ist das Verhältnis im grünen Bereich? Nahe 1 = ein Faktor passt überall.",
                 fontsize=11)
    ax.grid(alpha=0.3)

    # Step 3: result at the ball
    ax = fig.add_subplot(gs[1, 1])
    ax.axis("off")
    truth = true_distance(T, K, obj, shape)
    zb = {k: float(np.nanmedian(v[core])) for k, v in raw.items()}
    rows = [["Wahr (Maßband + 9,9 cm)", "", "", f"{de(truth)} m", ""],
            ["IPM (Bit-Bots)", "", "", f"{de(ipm_distance(T, K, obj, shape))} m", ""],
            ["ZED", f"{de(float(np.nanmedian(d['zed'][core])))} m", "", f"{de(object_distance(d['zed'], T, K, obj, core))} m",
             ""]]
    for name in raw:
        scaled = raw[name] * factor[name]
        dist_raw = object_distance(raw[name], T, K, obj, core) if name == "DA3 Metric-Large" else np.nan
        rows.append([f"{name} roh", de(zb[name], 3 if name == "DA3 Base" else 2) + ("" if name == "DA3 Base" else " m"),
                     "", "– (keine Meter)" if name == "DA3 Base" else f"{de(dist_raw)} m", ""])
        dist = object_distance(scaled, T, K, obj, core)
        rows.append([f"{name} skaliert", f"{de(zb[name] * factor[name])} m", f"× {de(factor[name])}", f"{de(dist)} m",
                     f"{(dist - truth) / truth * 100:+.1f} %".replace(".", ",")])
    rows[1][4] = f"{(ipm_distance(T, K, obj, shape) - truth) / truth * 100:+.1f} %".replace(".", ",")
    zd = object_distance(d["zed"], T, K, obj, core)
    rows[2][4] = f"{(zd - truth) / truth * 100:+.1f} %".replace(".", ",")
    rows[5][4] = f"{(object_distance(raw['DA3 Metric-Large'], T, K, obj, core) - truth) / truth * 100:+.1f} %".replace(
        ".", ",")
    tab = ax.table(cellText=rows, colLabels=["Methode", "Tiefe am Ball", "Faktor", "Abstand ab\nRobotermitte",
                                             "Fehler"], loc="upper center", cellLoc="center",
                   colWidths=[0.33, 0.17, 0.13, 0.2, 0.13])
    tab.auto_set_font_size(False)
    tab.set_fontsize(10.5)
    tab.scale(1, 1.9)
    for r in (4, 6):
        tab[r + 1, 3].set_facecolor("#C6EFCE")
    ax.set_title("Schritt 3: die ganze DA3-Tiefenkarte × Faktor, dann den Ball darin ablesen", fontsize=11)
    fig.text(0.53, 0.012, "Abstand ab Robotermitte = Tiefe am Ball in einen Bodenpunkt umgerechnet (bis zur Ballmitte).\n"
             "Wichtig: Für den Faktor wird nur der nahe Boden gebraucht, nicht der Ball und nicht das IPM.",
             fontsize=10, va="bottom")

    fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = OUT / "skalierung_erklaert.png"
    fig.savefig(out, dpi=110)
    print("saved", out)


if __name__ == "__main__":
    main()
