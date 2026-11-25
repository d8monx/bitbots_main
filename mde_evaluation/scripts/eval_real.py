"""Evaluate DA3, the ZED depth and the Bit-Bots IPM on the real recordings of 04.11.2026.

Usage: KMP_DUPLICATE_LIB_OK=TRUE python scripts/eval_real.py   (run extract_rosbag.py on the bags first)

All distances are horizontal distances from base_footprint (the point between the robot's feet), the frame the
Bit-Bots IPM reports in. The tape started at the toe, which TF puts TOE = 9.9 cm in front of base_footprint.
References:
- floor:   each pixel's viewing ray cut with the ground plane (camera pose from TF). It inherits the head-pitch
           error of TF, so it is a reference with its own uncertainty, not ground truth.
- wall:    the same rays cut with the wall plane (tape + TOE)
- objects: balls and robots marked per bag in annotations/real_26-11-04.json; distance = tape + TOE
Compared:
- ZED depth as recorded, and DA3-Metric-Large raw
- DA3 Small / Base / Metric-Large scaled per image with one factor, so that the median depth on the floor
  0.5-2.5 m in front of the robot matches the ground plane there. Needs no knowledge of wall or objects.
- IPM as the Bit-Bots do it (objects only): ball = ray through its centre onto a plane one ball radius above the
  ground; robot = bottom centre of its box onto the ground, no result if that point is in the lowest 20 % of the image.
DA3 depth maps are computed once and kept in data/da3/ (delete that folder to recompute them).
Writes results/real_26-11-04/: frames.csv, objects.csv, summary.txt, overview.png, object_distance.png,
markings.png, masks.png (masks.png and markings.png are for checking the masks by eye).
"""

import csv
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from eval_da3 import MODELS
from gpu import free_memory, pick_device

ROOT = Path(__file__).resolve().parents[1]
ANNOT_DIR = ROOT / "annotations"
ANNOT = json.loads((ANNOT_DIR / "real_26-11-04.json").read_text())
TOE = ANNOT["toe_offset"]  # toe of the camera robot in front of base_footprint (from TF)
WALL = ANNOT["wall"] + TOE  # wall distance from base_footprint
BALL_RADIUS = 0.0765  # m, the Bit-Bots IPM setting (ball_diameter 0.153); our ball still has to be measured
FOOTPOINT_LIMIT = 0.8  # Bit-Bots IPM: no robot position if its foot point is below this share of the image height
WALL_HALF_WIDTH = 0.8  # m left/right of the robot; further out there are shelves and side walls
FLOOR_HALF_WIDTH = 1.5  # m; further out the side wall and a cupboard stand on the floor
SCALE_REGION = (0.5, 2.5, 1.0)  # floor used for the scale: from, to, half width (m, in base_footprint)
STEEP = 30.0  # degrees; the 30 cm bags look down 38-41 degrees, all others 12-16
BINS = [(0, 1), (1, 3), (3, 6)]  # depth ranges in m, as in the exposé
OUT = ROOT / "results/real_26-11-04"
CACHE = ROOT / "data/da3"
# Scenes with robots or a person: they cover floor and wall, so these bags are left out of the floor/wall table
CLUTTERED = ("offen", "verdeckt", "ball_vorne", "ball_hinten", "mensch")
# Head turns at about 125 deg/s and extract_rosbag.py takes the latest TF instead of the one at the image stamp
SKIP = ("kopfbewegung",)
COLORS = {"blue": (255, 0, 0), "red": (0, 0, 255), "green": (0, 255, 0)}  # BGR as painted
DA3 = {name: name.split("/")[1] for name in MODELS}  # model id -> short name
SCALED = {short: short.replace("DA3METRIC-LARGE", "Metric-Large").replace("DA3-", "").title() + " skaliert"
          for short in DA3.values()}
DEPTH_METHODS = ["ZED", "Metric-Large roh", *SCALED.values()]
METHODS = ["IPM", *DEPTH_METHODS]


def rays(K):
    u, v = np.meshgrid(np.arange(K["w"]), np.arange(K["h"]))
    return np.stack([(u - K["cx"]) / K["fx"], (v - K["cy"]) / K["fy"], np.ones_like(u, float)], -1)


def room_depth(T, K):
    """z-depth of floor and wall per pixel, and which pixels see the floor, the wall, and the scale region."""
    d = rays(K) @ T[:3, :3].T  # ray directions in base_footprint (z-depth 1)
    t = T[:3, 3]
    with np.errstate(divide="ignore", invalid="ignore"):
        s_floor = np.where(d[..., 2] < 0, -t[2] / d[..., 2], np.inf)
        s_wall = np.where(d[..., 0] > 0, (WALL - t[0]) / d[..., 0], np.inf)
        hit = t + np.minimum(s_floor, s_wall)[..., None] * d
    x, y, z = hit[..., 0], hit[..., 1], hit[..., 2]
    floor = (s_floor <= s_wall) & np.isfinite(s_floor) & (x < WALL - 0.3) & (np.abs(y) < FLOOR_HALF_WIDTH)
    wall = (s_wall < s_floor) & (z > 0.1) & (z < 1.5) & (np.abs(y) < WALL_HALF_WIDTH)
    near = floor & (x > SCALE_REGION[0]) & (x < SCALE_REGION[1]) & (np.abs(y) < SCALE_REGION[2])
    return np.where(floor | wall, np.minimum(s_floor, s_wall), np.nan), floor, wall, near


def floor_scale(pred, gt, near):
    """One factor per image: ground-plane depth / predicted depth on the near floor."""
    return float(np.median(gt[near]) / np.median(pred[near]))


def object_masks(obj, K):
    """(core, full, shape): core = pixels used to read the depth, full = object with a margin (left out of
    floor and wall), shape = the object as marked."""
    if "circle" in obj:
        cx, cy, r = obj["circle"]
        u, v = np.meshgrid(np.arange(K["w"]), np.arange(K["h"]))
        dist2 = (u - cx) ** 2 + (v - cy) ** 2
        return dist2 <= max(0.6 * r, 1.0) ** 2, dist2 <= (r + 3) ** 2, dist2 <= r**2
    img = cv2.imread(str(ANNOT_DIR / obj["mask"]))
    shape = np.all(np.abs(img.astype(int) - COLORS[obj["color"]]) < 60, axis=-1)
    core = cv2.erode(shape.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0  # no mixed pixels at the edge
    if core.sum() < 20:
        core = shape
    return core, cv2.dilate(shape.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0, shape


def plane_point(T, K, u, v, height=0.0):
    """Where the viewing ray of pixel (u, v) meets the horizontal plane at this height, in base_footprint.
    Same calculation as the Bit-Bots ipm_library. None if the ray does not go down."""
    d = T[:3, :3] @ np.array([(u - K["cx"]) / K["fx"], (v - K["cy"]) / K["fy"], 1.0])
    if d[2] >= 0:
        return None
    return T[:3, 3] + (height - T[2, 3]) / d[2] * d


def ipm_distance(T, K, obj, shape):
    """Distance as the Bit-Bots soccer_ipm gives it, from the marked object instead of a detection."""
    v, u = np.nonzero(shape)
    if obj["kind"] == "ball":
        p = plane_point(T, K, u.mean(), v.mean(), BALL_RADIUS)
    elif v.max() > FOOTPOINT_LIMIT * K["h"]:
        return np.nan  # soccer_ipm drops robots whose foot point is at the bottom of the image
    else:
        p = plane_point(T, K, (u.min() + u.max()) / 2, v.max())
    return np.nan if p is None else float(np.hypot(p[0], p[1]))


def true_distance(T, K, obj, shape):
    """Tape value moved from the toe to base_footprint. The tape ran straight from the toe to the object; its
    direction is taken from the image, which only matters for the ball placed to the left."""
    v, u = np.nonzero(shape)
    p = plane_point(T, K, u.mean(), v.max())
    bearing = 0.0 if p is None else np.arctan2(p[1], p[0] - TOE)
    return float(np.hypot(TOE + obj["tape"] * np.cos(bearing), obj["tape"] * np.sin(bearing)))


def object_distance(depth, T, K, obj, core):
    """Horizontal distance from base_footprint to the object, read from a depth map.
    Ball: to its centre (one radius behind the visible surface). Robot: to its visible front."""
    z = depth[core]
    z = z[np.isfinite(z) & (z > 0)]
    if z.size == 0:
        return np.nan
    v, u = np.nonzero(core)
    ray = np.array([(u.mean() - K["cx"]) / K["fx"], (v.mean() - K["cy"]) / K["fy"], 1.0])
    p = np.median(z) * ray
    if obj["kind"] == "ball":
        p = p + BALL_RADIUS * ray / np.linalg.norm(ray)
    g = T[:3, :3] @ p + T[:3, 3]
    return float(np.hypot(g[0], g[1]))


def errors(pred, gt, mask):
    m = mask & np.isfinite(pred) & (pred > 0) & np.isfinite(gt)
    if m.sum() < 50:
        return {"abs_rel": np.nan, "factor": np.nan, "coverage": float(m.sum() / max(mask.sum(), 1))}
    return {"abs_rel": float(np.mean(np.abs(pred[m] - gt[m]) / gt[m])),
            "factor": float(np.median(pred[m] / gt[m])),  # >1: too far, <1: too near
            "coverage": float(m.sum() / mask.sum())}


def object_label(obj):
    kind = "Ball" if obj["kind"] == "ball" else "Roboter"
    note = {"in the air": " (in der Luft)", "mostly hidden": " (fast verdeckt)", "half hidden": " (halb verdeckt)",
            "at image edge": " (am Bildrand)"}
    frames = f" Bild {','.join(map(str, obj['frames']))}" if "frames" in obj and len(obj["frames"]) == 1 else ""
    return f"{kind} {obj['tape']:g} m" + note.get(obj.get("note"), "") + frames


def pitch_deg(T):
    """How far the camera looks down, in degrees."""
    return float(np.degrees(np.arcsin(-T[2, 2])))


def load_frames():
    """All extracted frames with ZED depth, their references and their marked objects."""
    folders = sorted(p for p in (ROOT / "data/frames").glob("mde_s1_*")
                     if (p / "frames.json").exists() and not any(s in p.name for s in SKIP))
    data = []
    for folder in folders:
        K = json.loads((folder / "intrinsics.json").read_text())
        for idx, f in enumerate(json.loads((folder / "frames.json").read_text())):
            if "zed_depth" not in f:
                continue
            T = np.array(f["T_ground_camera"])
            gt, floor, wall, near = room_depth(T, K)
            objs = []
            for i_obj, obj in enumerate(ANNOT["objects"].get(folder.name, [])):
                if "frames" in obj and idx not in obj["frames"]:
                    continue
                if "mask" in obj and not (ANNOT_DIR / obj["mask"]).exists():
                    print("missing mask, object skipped:", obj["mask"])
                    continue
                core, full, shape = object_masks(obj, K)
                floor, wall = floor & ~full, wall & ~full  # `near` keeps them: the robot would not know them
                objs.append((i_obj, obj, core, shape))
            data.append({"bag": folder.name, "idx": idx, "file": f["file"], "K": K, "T": T, "gt": gt,
                         "floor": floor, "wall": wall, "near": near, "objs": objs,
                         "zed": np.load(folder / f["zed_depth"]),
                         "rgb": cv2.cvtColor(cv2.imread(str(folder / f["file"])), cv2.COLOR_BGR2RGB)})
    print(f"{len(data)} frames from {len(folders)} bags")
    return data


def add_da3(data):
    """Put the DA3 depth maps (metres for the metric model, relative otherwise) into each frame.
    Computed once per frame and model, then read from data/da3/."""
    for name, short in DA3.items():
        paths = [CACHE / d["bag"] / f"{short}_{d['idx']:04d}.npy" for d in data]
        todo = [(d, p) for d, p in zip(data, paths) if not p.exists()]
        if todo:
            from depth_anything_3.api import DepthAnything3  # slow import, only needed here
            from eval_rosbag import predict

            device = pick_device()
            model = DepthAnything3.from_pretrained(name).to(device).eval()
            for d, p in todo:
                p.parent.mkdir(parents=True, exist_ok=True)
                np.save(p, predict(model, d["rgb"], d["K"], "METRIC" in name).astype(np.float32))
            del model
            free_memory(device)
            print(f"computed {short} for {len(todo)} frames")
        for d, p in zip(data, paths):
            d[short] = np.load(p).astype(float)


def depth_variants(d, gt=None, near=None):
    """Depth map per method. gt and near can be replaced to test another camera pose."""
    gt, near = (d["gt"], d["near"]) if gt is None else (gt, near)
    variants = {"ZED": d["zed"], "Metric-Large roh": d["DA3METRIC-LARGE"]}
    for short, label in SCALED.items():
        variants[label] = d[short] * floor_scale(d[short], gt, near)
    return variants


def session(bag):
    name = bag.removeprefix("mde_s1_")
    return "neural_plus" if name.startswith("neural_plus") else name.split("_")[0]


def fmt(x, spec):
    width = int(spec.lstrip(">").split(".")[0])
    return f"{x:{spec}}" if np.isfinite(x) else f"{'–':>{width}}"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_frames()
    add_da3(data)

    rows, obj_rows = [], []
    for d in data:
        d["variants"] = depth_variants(d)
        base = {"bag": d["bag"], "file": d["file"], "pitch": round(pitch_deg(d["T"]), 1)}
        for method, pred in d["variants"].items():
            fl, wa = errors(pred, d["gt"], d["floor"]), errors(pred, d["gt"], d["wall"])
            row = {**base, "method": method,
                   "floor_abs_rel": fl["abs_rel"], "floor_factor": fl["factor"], "floor_coverage": fl["coverage"],
                   "wall_abs_rel": wa["abs_rel"], "wall_factor": wa["factor"], "wall_coverage": wa["coverage"]}
            for lo, hi in BINS:
                e = errors(pred, d["gt"], d["floor"] & (d["gt"] >= lo) & (d["gt"] < hi))
                row[f"floor_abs_rel_{lo}_{hi}"], row[f"floor_factor_{lo}_{hi}"] = e["abs_rel"], e["factor"]
            rows.append(row)
        for i_obj, obj, core, shape in d["objs"]:
            estimates = {"IPM": ipm_distance(d["T"], d["K"], obj, shape)}
            estimates.update({m: object_distance(p, d["T"], d["K"], obj, core) for m, p in d["variants"].items()})
            for method, estimate in estimates.items():
                obj_rows.append({"bag": d["bag"], "file": d["file"], "object": i_obj, "label": object_label(obj),
                                 "kind": obj["kind"], "note": obj.get("note", ""), "method": method,
                                 "tape": obj["tape"], "truth": true_distance(d["T"], d["K"], obj, shape),
                                 "estimate": estimate})

    for path, table in ((OUT / "frames.csv", rows), (OUT / "objects.csv", obj_rows)):
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)

    lines = []

    def out(s=""):
        print(s)
        lines.append(s)

    # One value per bag (median over its frames): the frames of a bag are almost identical
    keys = [k for k in rows[0] if k.startswith(("floor_", "wall_"))]
    per_bag = {}  # (bag, method) -> {key: median over frames}
    for r in rows:
        per_bag.setdefault((r["bag"], r["method"]), []).append(r)
    with np.errstate(all="ignore"):
        per_bag = {k: {"pitch": float(np.median([r["pitch"] for r in rs]))}
                   | {key: float(np.nanmedian([r[key] for r in rs])) if np.isfinite([r[key] for r in rs]).any()
                      else np.nan for key in keys} for k, rs in per_bag.items()}
    empty = sorted({bag for bag, _ in per_bag if not any(c in bag for c in CLUTTERED)})

    def over_bags(bags, method, key):
        vals = [per_bag[b, method][key] for b in bags]
        return float(np.nanmedian(vals)) if np.isfinite(vals).any() else np.nan

    # Table 1: floor and wall per recording session and head pitch
    out("Boden und Wand, nur Szenen ohne Roboter/Mensch. Erst Median pro Bag, dann Median über die Bags.")
    out(f"Abstände ab base_footprint; Maßband ab Fußspitze, also + {TOE:.3f} m. Wand: {WALL:.3f} m.")
    out("Faktor = geschätzt / Referenz (1,00 = gleich, >1 zu weit, <1 zu nah); AbsRel = mittlerer relativer Fehler")
    out("Boden-Referenz = Ebene aus der Kamerapose (TF), selbst auf etwa 0,1-0,5 Grad Kopfneigung unsicher.")
    out(f"DA3 skaliert: ein Faktor pro Bild aus dem Boden {SCALE_REGION[0]:g}-{SCALE_REGION[1]:g} m vor dem Roboter.")
    for mode in ("performance", "neural", "neural_plus"):
        for steep in (False, True):
            bags = [b for b in empty if session(b) == mode and (per_bag[b, "ZED"]["pitch"] >= STEEP) == steep]
            if not bags:
                continue
            pitches = [per_bag[b, "ZED"]["pitch"] for b in bags]
            out(f"\nSitzung {mode}, Kopf {'steil' if steep else 'flach'} "
                f"({min(pitches):.1f}-{max(pitches):.1f} Grad nach unten), {len(bags)} Bags")
            out(f"{'Methode':22} {'Boden AbsRel':>13} {'Boden Faktor':>13} {'Wand AbsRel':>12} {'Wand Faktor':>12} "
                f"{'ZED Boden':>10} {'ZED Wand':>9}")
            for m in DEPTH_METHODS:
                v = {k: over_bags(bags, m, k) for k in keys}
                cov = f"{v['floor_coverage']:>10.0%} {v['wall_coverage']:>9.0%}" if m == "ZED" else ""
                out(f"{m:22} {fmt(v['floor_abs_rel'], '>13.3f')} {fmt(v['floor_factor'], '>13.2f')} "
                    f"{fmt(v['wall_abs_rel'], '>12.3f')} {fmt(v['wall_factor'], '>12.2f')} {cov}")

    # Table 2: floor by depth range, all empty scenes with a flat head
    flat = [b for b in empty if per_bag[b, "ZED"]["pitch"] < STEEP]
    out(f"\nBoden nach Entfernung (Tiefe in m), {len(flat)} Bags mit flachem Kopf: AbsRel (Faktor)")
    out(f"{'Methode':22} " + " ".join(f"{f'{lo}-{hi} m':>14}" for lo, hi in BINS))
    for m in DEPTH_METHODS:
        cells = [f"{over_bags(flat, m, f'floor_abs_rel_{lo}_{hi}'):.3f} ({over_bags(flat, m, f'floor_factor_{lo}_{hi}'):.2f})"
                 for lo, hi in BINS]
        out(f"{m:22} " + " ".join(f"{c:>14}" for c in cells))

    # Table 3: object distances (median over the frames an object is marked in)
    out("\nAbstand zu Bällen und Robotern in m, waagrecht ab base_footprint. Wahr = Maßband + Fußspitze.")
    out(f"Ball: bis Ballmitte (Radius {BALL_RADIUS * 100:.2f} cm wie im Bit-Bots-IPM, unser Ball ist noch nicht gemessen).")
    out("Roboter: Maßband bis zu seiner Fußspitze. Tiefe: sichtbare Vorderseite. IPM: unterer Rand der Box;")
    out("'–' = das IPM gibt nichts aus, weil der Fußpunkt am unteren Bildrand liegt.")
    short = [m.replace(" skaliert", " sk.").replace("Metric-Large", "Metric") for m in METHODS]
    out(f"{'Bag':18} {'Objekt':31} {'Wahr':>5} " + " ".join(f"{m:>10}" for m in short))
    groups = {}  # (bag, object index or "air") -> rows
    for r in obj_rows:
        groups.setdefault((r["bag"], r["object"] if r["note"] != "in the air" else "air"), []).append(r)
    per_kind = {}  # category -> method -> [(truth, estimate)]
    for (bag, _), rs in sorted(groups.items(), key=lambda kv: (kv[1][0]["kind"], kv[1][0]["truth"], kv[0][0])):
        r0 = rs[0]
        cat = ("Ball in der Luft" if r0["note"] == "in the air" else
               "Ball verdeckt" if r0["note"] == "mostly hidden" else
               "Ball am Boden" if r0["kind"] == "ball" else "Roboter")
        vals = []
        for m in METHODS:
            est = [r["estimate"] for r in rs if r["method"] == m]
            v = float(np.nanmedian(est)) if np.isfinite(est).any() else np.nan
            per_kind.setdefault(cat, {}).setdefault(m, []).append((r0["truth"], v))
            vals.append(v)
        out(f"{bag.removeprefix('mde_s1_'):18} {r0['label']:31} {r0['truth']:>5.2f} "
            + " ".join(fmt(v, ">10.2f") for v in vals))

    out("\nMittlerer relativer Fehler beim Abstand (gegen 'Wahr'), je Objektart:")
    out(f"{'':22} " + " ".join(f"{m:>10}" for m in short) + "   Anzahl")
    for cat, by_m in per_kind.items():
        errs = [[abs(v - t) / t for t, v in by_m[m] if np.isfinite(v)] for m in METHODS]
        out(f"{cat:22} " + " ".join(fmt(np.mean(e) if e else np.nan, ">10.3f") for e in errs)
            + f"   {len(by_m[METHODS[0]])}")
    balls = per_kind.get("Ball am Boden", {})
    for lo, hi in ((0, 1.5), (1.5, 3.5), (3.5, 6)):
        errs = [[abs(v - t) / t for t, v in balls.get(m, []) if np.isfinite(v) and lo <= t < hi] for m in METHODS]
        out(f"{f'  Ball am Boden {lo:g}-{hi:g} m':22} " + " ".join(fmt(np.mean(e) if e else np.nan, ">10.3f")
                                                                  for e in errs) + f"   {len(errs[0])}")
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")

    # Figure 1: example frames
    show = ["mde_s1_performance", "mde_s1_neural", "mde_s1_neural_plus", "mde_s1_performance_2m",
            "mde_s1_performance_ball_hinten"]
    picks = [next(d for d in data if d["bag"] == b) for b in show if any(d["bag"] == b for d in data)]
    cols = ["Bild", "Referenz (Boden + Wand)", "ZED", "Metric-Large roh", "Metric-Large skaliert", "Base skaliert"]
    fig, axes = plt.subplots(len(picks), len(cols), figsize=(3.2 * len(cols), 2.0 * len(picks)), squeeze=False,
                             layout="constrained")
    for r, d in enumerate(picks):
        v = d["variants"]
        imgs = [d["rgb"], d["gt"], d["zed"], v["Metric-Large roh"], v["Metric-Large skaliert"], v["Base skaliert"]]
        for c, img in enumerate(imgs):
            ax = axes[r, c]
            if c == 0:
                ax.imshow(img)
                ax.set_ylabel(d["bag"].removeprefix("mde_s1_"), fontsize=8)
            else:
                ax.imshow(np.where(np.isfinite(img) & (img > 0), img, np.nan), cmap="turbo", vmin=0, vmax=7)
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=9)
    sm = plt.cm.ScalarMappable(cmap="turbo", norm=plt.Normalize(0, 7))
    fig.colorbar(sm, ax=axes, shrink=0.6, label="Tiefe [m] (weiß = kein Wert)")
    fig.savefig(OUT / "overview.png", dpi=100)

    # Figure 2: object distance, estimated vs. true, one panel per method
    marker = {"Ball am Boden": "o", "Ball in der Luft": "^", "Ball verdeckt": "x", "Roboter": "s"}
    fig, axes = plt.subplots(1, len(METHODS), figsize=(3.3 * len(METHODS), 3.6), sharey=True, layout="constrained")
    top = max(v for by_m in per_kind.values() for pts in by_m.values() for t, v in pts if np.isfinite(v))
    lim = [0, max(5.5, np.ceil(top + 0.3))]  # never cut off a point
    for ax, m in zip(axes, METHODS):
        for cat, by_m in per_kind.items():
            pts = np.array([p for p in by_m[m] if np.isfinite(p[1])])
            if len(pts):
                ax.plot(pts[:, 0], pts[:, 1], marker[cat], ms=5, label=cat, alpha=0.8)
        ax.plot(lim, lim, color="gray", lw=1)
        ax.set_xlim(lim), ax.set_ylim(lim)
        ax.set_title(m, fontsize=10)
        ax.set_xlabel("wahr [m] (Maßband + Fußspitze)")
    axes[0].set_ylabel("geschätzt [m]")
    axes[0].legend(fontsize=7)
    fig.suptitle("Abstand zu Ball und Robotern ab base_footprint: auf der grauen Linie = richtig")
    fig.savefig(OUT / "object_distance.png", dpi=100)

    # Figure 3: all markings on the image, to check them by eye
    seen, picks = set(), []
    for d in data:  # one frame per bag, plus every frame with per-frame markings
        if not d["objs"]:
            continue
        key = d["bag"] if not any("frames" in o for _, o, _, _ in d["objs"]) else (d["bag"], d["idx"])
        if key not in seen:
            seen.add(key)
            picks.append(d)
    ncol = 4
    nrow = int(np.ceil(len(picks) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.4 * nrow), squeeze=False, layout="constrained")
    for ax in axes.flat:
        ax.axis("off")
    for ax, d in zip(axes.flat, picks):
        img = d["rgb"].copy()
        for _, obj, core, _ in d["objs"]:
            col = (255, 60, 60) if obj["kind"] == "ball" else (60, 220, 255)
            img[core] = (0.5 * img[core] + 0.5 * np.array(col)).astype(np.uint8)
            v, u = np.nonzero(core)
            cv2.putText(img, object_label(obj), (int(u.min()), max(int(v.min()) - 4, 10)), 0, 0.38, col, 1)
        ax.imshow(img)
        ax.set_title(f"{d['bag'].removeprefix('mde_s1_')} {d['file']}", fontsize=8)
    fig.savefig(OUT / "markings.png", dpi=100)

    # Figure 4: floor, wall and scale region of the first frame of every bag, to check them by eye
    firsts = [d for d in data if d["idx"] == min(e["idx"] for e in data if e["bag"] == d["bag"])]
    nrow = int(np.ceil(len(firsts) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.4 * nrow), squeeze=False, layout="constrained")
    for ax in axes.flat:
        ax.axis("off")
    for ax, d in zip(axes.flat, firsts):
        img = d["rgb"].copy()
        for mask, col in ((d["floor"], (60, 220, 60)), (d["wall"], (60, 120, 255))):
            img[mask] = (0.6 * img[mask] + 0.4 * np.array(col)).astype(np.uint8)
        contours, _ = cv2.findContours(d["near"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, contours, -1, (255, 230, 0), 2)
        ax.imshow(img)
        ax.set_title(f"{d['bag'].removeprefix('mde_s1_')} ({pitch_deg(d['T']):.1f}°)", fontsize=8)
    fig.suptitle("Grün = Boden, blau = Wand (ausgewertet); gelber Rahmen = Boden für den Maßstab")
    fig.savefig(OUT / "masks.png", dpi=100)
    print("saved", OUT)


if __name__ == "__main__":
    main()
