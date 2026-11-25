"""How much does a wrong head pitch in TF move the estimated ball distance: Bit-Bots IPM vs. floor-scaled DA3?

Usage: KMP_DUPLICATE_LIB_OK=TRUE python scripts/pitch_sensitivity.py   (run eval_real.py first, it computes DA3)

The camera pose from TF is not exact (worn or misadjusted parts). Both the IPM and our floor scale rely on it.
Here the pose of every frame is tilted by a few tenths of a degree on purpose, as if TF were wrong by that much,
and the ball distance is estimated again. Reported is how far the estimate moves compared to the untouched pose.
Writes results/real_26-11-04/pitch_sensitivity.txt and pitch_sensitivity.png.
"""

import numpy as np
from eval_real import OUT, SCALED, add_da3, depth_variants, ipm_distance, load_frames, object_distance, room_depth

import matplotlib.pyplot as plt

DELTAS = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]  # degrees; positive = TF believes the camera looks further down
METHODS = ["IPM", SCALED["DA3-BASE"], SCALED["DA3METRIC-LARGE"]]
CAMERA_HEIGHT = 0.565  # m, for the expected curve


def tilted(T, delta_deg):
    """The same pose, but looking delta_deg further down (rotation about the camera's x axis)."""
    a = np.radians(-delta_deg)
    R = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    T2 = T.copy()
    T2[:3, :3] = T[:3, :3] @ R
    return T2


def main():
    data = [d for d in load_frames()
            if any(o["kind"] == "ball" and "note" not in o for _, o, _, _ in d["objs"])]
    add_da3(data)

    est = {}  # (bag, method, delta) -> [estimate per frame]
    truth = {}
    for d in data:
        for delta in DELTAS:
            T = tilted(d["T"], delta)
            gt, _, _, near = room_depth(T, d["K"])
            variants = depth_variants(d, gt, near)
            for _, obj, core, shape in d["objs"]:
                if obj["kind"] != "ball" or "note" in obj:
                    continue
                truth[d["bag"]] = obj["tape"]
                est.setdefault((d["bag"], "IPM", delta), []).append(ipm_distance(T, d["K"], obj, shape))
                for m in METHODS[1:]:
                    est.setdefault((d["bag"], m, delta), []).append(
                        object_distance(variants[m], T, d["K"], obj, core))
    est = {k: float(np.median(v)) for k, v in est.items()}

    # Shift of the estimate against the untouched pose, grouped by tape distance
    groups = {"bis 1 m": (0, 1.2), "1,5-2,35 m": (1.2, 2.6), "3 m": (2.6, 3.5), "4,5 m": (3.5, 6)}
    shift = {}  # (group, method, delta) -> mean shift in m over the bags
    for g, (lo, hi) in groups.items():
        bags = [b for b, t in truth.items() if lo <= t < hi]
        for m in METHODS:
            for delta in DELTAS:
                shift[g, m, delta] = float(np.mean([est[b, m, delta] - est[b, m, 0.0] for b in bags]))
        groups[g] = (lo, hi, len(bags), float(np.mean([truth[b] for b in bags])))

    lines = ["Wie weit verschiebt sich der geschätzte Ballabstand (cm), wenn die Kopfneigung in TF falsch ist?",
             "Positiv = TF glaubt, die Kamera schaut steiler nach unten, als sie es tut. Mittel über die Bags.", ""]
    for g, (_, _, n, mean_t) in groups.items():
        lines.append(f"Ball {g} ({n} Bags)")
        lines.append(f"{'Fehler in TF':22} " + " ".join(f"{d:>+7.1f}°" for d in DELTAS))
        for m in METHODS:
            lines.append(f"{m:22} " + " ".join(f"{shift[g, m, d] * 100:>+8.1f}" for d in DELTAS))
        lines.append("")
    text = "\n".join(lines)
    print(text)
    (OUT / "pitch_sensitivity.txt").write_text(text)

    fig, axes = plt.subplots(1, len(groups), figsize=(3.4 * len(groups), 3.4), sharey=True, layout="constrained")
    for ax, (g, (_, _, n, mean_t)) in zip(axes, groups.items()):
        for m in METHODS:
            ax.plot(DELTAS, [shift[g, m, d] * 100 for d in DELTAS], "o-", ms=4, label=m)
        d_cam = mean_t + 0.05  # tape starts about 5 cm in front of the point below the camera
        expected = [-(CAMERA_HEIGHT**2 + d_cam**2) / CAMERA_HEIGHT * np.radians(d) * 100 for d in DELTAS]
        ax.plot(DELTAS, expected, ":", color="gray", label="Formel für IPM")
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_title(f"Ball {g} ({n} Bags)", fontsize=10)
        ax.set_xlabel("Fehler der Kopfneigung in TF [°]")
    axes[0].set_ylabel("Verschiebung der Schätzung [cm]")
    axes[0].legend(fontsize=7)
    fig.suptitle("Wie stark reagiert der Ballabstand auf eine falsche Kopfneigung?")
    fig.savefig(OUT / "pitch_sensitivity.png", dpi=110)
    print("saved", OUT / "pitch_sensitivity.png")


if __name__ == "__main__":
    main()
