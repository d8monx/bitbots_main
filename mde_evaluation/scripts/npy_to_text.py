"""Convert a per-pixel depth map (e.g. zed_depth_XXXX.npy from extract_rosbag.py) into readable text,
or show it as a 3D point cloud.

Usage: python scripts/npy_to_text.py <file.npy|file.csv> [--format list|grid|xlsx] [--out <file>] [--decimals N]
       python scripts/npy_to_text.py <file.npy|file.csv> --show3d [--stride N]

  list (default): one line per pixel "row col depth_m"; easy to filter or load into a spreadsheet
  grid:           CSV with one line per image row and one column per image column
  xlsx:           Excel workbook (needs openpyxl): the grid laid out like the image with row/column
                  indices and a depth colour scale, plus an "info" sheet with size and depth range
  --show3d:       instead of writing text, back-project every pixel into 3D with the camera intrinsics
                  (intrinsics.json in the same folder) and show the point cloud. Points are coloured
                  from the matching RGB frame (zed_depth_XXXX -> frame_XXXX.png) if it exists,
                  otherwise by depth. If frames.json has the camera pose for this frame, the cloud is
                  shown in base_footprint (floor at z = 0), otherwise in the camera frame.
                  Two sliders select an image pixel (x = column, y = row); its 3D point is marked and
                  the distance from the camera to it is shown below the plot.

The input can be the .npy depth map or a grid .csv written by this script.
Pixels without a valid depth (NaN / inf / <= 0) are written as "nan" (empty cells in xlsx) and left
out of the 3D view.
Without --out, the output is written next to the input file with a .txt / .csv / .xlsx suffix.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np


def load_depth(path: Path) -> np.ndarray:
    """Load a 2D depth map from .npy or grid .csv; invalid pixels become NaN."""
    if path.suffix == ".csv":
        depth = np.loadtxt(path, delimiter=",")
    else:
        depth = np.load(path)
    if depth.ndim != 2:
        raise SystemExit(f"expected a 2D depth map, got shape {depth.shape}")
    depth = depth.astype(np.float64)
    # Mark every invalid pixel uniformly so there is a single "no value" marker
    depth[(depth >= 50) | (depth <= 0)] = np.nan
    return depth


def write_text(depth: np.ndarray, out: Path, fmt_name: str, decimals: int):
    fmt = f"%.{decimals}f"
    if fmt_name == "grid":
        np.savetxt(out, depth, fmt=fmt, delimiter=",")
    else:
        rows, cols = np.indices(depth.shape)
        table = np.column_stack([rows.ravel(), cols.ravel(), depth.ravel()])
        np.savetxt(out, table, fmt=["%d", "%d", fmt], header="row col depth_m", comments="")


def write_xlsx(depth: np.ndarray, out: Path, decimals: int, source: Path):
    """Excel workbook with the depth map laid out like the image, coloured by depth."""
    from openpyxl import Workbook
    from openpyxl.formatting.rule import ColorScaleRule
    from openpyxl.styles import Font
    from openpyxl.utils import get_column_letter

    h, w = depth.shape
    wb = Workbook()
    ws = wb.active
    ws.title = "depth_m"
    # Row 1 holds the pixel column indices, column A the pixel row indices; the depth grid starts at B2
    ws.cell(1, 1, "row \\ col").font = Font(bold=True)
    for col in range(w):
        ws.cell(1, col + 2, col).font = Font(bold=True)
    number_format = "0." + "0" * decimals if decimals > 0 else "0"
    rounded = np.round(depth, decimals)
    for row in range(h):
        ws.cell(row + 2, 1, row).font = Font(bold=True)
        for col in range(w):
            value = rounded[row, col]
            if np.isfinite(value):  # invalid pixels stay empty cells
                ws.cell(row + 2, col + 2, float(value)).number_format = number_format
    ws.freeze_panes = "B2"
    for col in range(2, w + 2):
        ws.column_dimensions[get_column_letter(col)].width = 6
    # Colour scale (near = blue, far = red) so the sheet shows the depth image at a glance
    ws.conditional_formatting.add(
        f"B2:{get_column_letter(w + 1)}{h + 1}",
        ColorScaleRule(start_type="min", start_color="3B4CC0", mid_type="percentile", mid_value=50,
                       mid_color="F7F7F7", end_type="max", end_color="B40426"),
    )

    valid = np.isfinite(depth)
    info = wb.create_sheet("info")
    rows = [("source", str(source)), ("width [px]", w), ("height [px]", h), ("unit", "metres"),
            ("valid pixels", int(valid.sum())), ("invalid pixels (empty cells)", int((~valid).sum()))]
    if valid.any():
        rows += [("min depth [m]", float(depth[valid].min())), ("max depth [m]", float(depth[valid].max()))]
    for r in rows:
        info.append(r)
    info.column_dimensions["A"].width = 28
    wb.save(out)


def camera_pose(path: Path) -> np.ndarray | None:
    """Pose of the camera in base_footprint for this depth map from frames.json, or None if unavailable."""
    frames_file = path.parent / "frames.json"
    if not frames_file.exists():
        print(f"no {frames_file.name} found, plotting in the camera frame")
        return None
    for f in json.loads(frames_file.read_text()):
        if Path(f.get("zed_depth", "")).stem == path.stem:
            return np.array(f["T_ground_camera"])
    print(f"{path.stem} is not listed in {frames_file.name}, plotting in the camera frame")
    return None


def show_3d(depth: np.ndarray, path: Path, stride: int):
    import matplotlib.pyplot as plt
    from PIL import Image

    intrinsics_file = path.parent / "intrinsics.json"
    if not intrinsics_file.exists():
        raise SystemExit(f"--show3d needs the camera intrinsics, but {intrinsics_file} does not exist")
    K = json.loads(intrinsics_file.read_text())
    if (K["h"], K["w"]) != depth.shape:
        raise SystemExit(f"intrinsics are for {K['w']}x{K['h']}, depth map is {depth.shape[1]}x{depth.shape[0]}")

    from matplotlib.widgets import Slider

    def back_project(u, v, z):
        """Pinhole back-projection into the camera optical frame (z forward, x right, y down).

        ZED depth is the distance along the optical axis (z), not along the viewing ray.
        """
        return np.stack([(u - K["cx"]) / K["fx"] * z, (v - K["cy"]) / K["fy"] * z, z], -1)

    # Express points in an x forward / y left / z up frame for plotting
    T = camera_pose(path)
    if T is not None:
        to_plot = lambda p_cam: p_cam @ T[:3, :3].T + T[:3, 3]  # noqa: E731
        camera = T[:3, 3]
        frame = "base_footprint (ground below the robot)"
    else:
        to_plot = lambda p_cam: p_cam @ np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]]).T  # noqa: E731
        camera = np.zeros(3)
        frame = "camera"
    print(f"plotting in the {frame} frame")

    # Subsample the pixel grid; a full-resolution scatter plot is too slow to rotate interactively
    v, u = np.indices(depth.shape)
    z = depth[::stride, ::stride]
    u, v = u[::stride, ::stride], v[::stride, ::stride]
    valid = np.isfinite(z)
    z, u, v = z[valid], u[valid], v[valid]
    points = to_plot(back_project(u, v, z))

    # Colour from the matching RGB frame if available
    match = re.search(r"(\d+)$", path.stem)
    rgb_file = path.parent / f"frame_{match.group(1)}.png" if match else None
    if rgb_file is not None and rgb_file.exists():
        rgb = np.asarray(Image.open(rgb_file).convert("RGB"))
        color_kw = {"c": rgb[v, u] / 255.0}
        print(f"colouring points from {rgb_file.name}")
    else:
        color_kw = {"c": z, "cmap": "turbo"}
        print("no matching RGB frame found, colouring points by depth")

    fig = plt.figure(figsize=(10, 8))
    # computed_zorder=False so the selected point is drawn on top of the dense cloud
    ax = fig.add_axes([0.0, 0.14, 1.0, 0.78], projection="3d", computed_zorder=False)
    sc = ax.scatter(*points.T, s=1, depthshade=False, **color_kw)
    ax.scatter(*camera, c="red", s=40, marker="^", label="camera")
    if "cmap" in color_kw:
        fig.colorbar(sc, ax=ax, shrink=0.6, label="depth [m]")
    ax.set_xlabel("forward [m]")
    ax.set_ylabel("left [m]")
    ax.set_zlabel("up [m]")
    # Equal scale on all axes so the geometry is not distorted
    ax.set_box_aspect(np.ptp(np.vstack([points, camera]), axis=0))
    ax.view_init(elev=20, azim=-160)
    ax.set_title(f"{path.name} in {frame} frame\n({len(points)} points, every {stride}. pixel)")

    # Sliders select an image pixel (x = column, y = row) at full resolution; the selected 3D point
    # is marked and connected to the camera, and its distance is shown below the plot
    h, w = depth.shape
    slider_x = Slider(fig.add_axes([0.15, 0.09, 0.7, 0.03]), "x (column)", 0, w - 1, valstep=1, valinit=w // 2)
    slider_y = Slider(fig.add_axes([0.15, 0.05, 0.7, 0.03]), "y (row)", 0, h - 1, valstep=1, valinit=h // 2)
    (marker,) = ax.plot([], [], [], "o", color="magenta", mec="black", ms=9, zorder=10, label="selected pixel")
    (ray,) = ax.plot([], [], [], "-", color="magenta", lw=2, zorder=9)
    info = fig.text(0.5, 0.005, "", ha="center", va="bottom", fontsize=11, family="monospace")

    def update(_=None):
        col, row = int(slider_x.val), int(slider_y.val)
        d = depth[row, col]
        if not np.isfinite(d):
            marker.set_data_3d([], [], [])
            ray.set_data_3d([], [], [])
            info.set_text(f"pixel ({col}, {row}): no valid depth")
        else:
            p_cam = back_project(col, row, d)
            p = to_plot(p_cam)
            marker.set_data_3d([p[0]], [p[1]], [p[2]])
            ray.set_data_3d([camera[0], p[0]], [camera[1], p[1]], [camera[2], p[2]])
            # Euclidean distance along the viewing ray vs. the depth value stored in the map
            info.set_text(f"pixel ({col}, {row}): distance camera -> point {np.linalg.norm(p_cam):.3f} m"
                          f"  (depth along optical axis {d:.3f} m)")
        fig.canvas.draw_idle()

    slider_x.on_changed(update)
    slider_y.on_changed(update)
    update()
    ax.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("depth", type=Path, help="depth map as .npy or grid .csv")
    parser.add_argument("--format", choices=["list", "grid", "xlsx"], default="list")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--decimals", type=int, default=3)
    parser.add_argument("--show3d", action="store_true", help="show a 3D point cloud instead of writing text")
    parser.add_argument("--stride", type=int, default=2, help="use every N-th pixel in --show3d")
    args = parser.parse_args()
    if args.stride < 1:
        parser.error("--stride must be >= 1")

    depth = load_depth(args.depth)
    valid = np.isfinite(depth)
    print(f"{depth.shape[0]}x{depth.shape[1]} pixels, {valid.mean():.1%} valid", end="")
    if valid.any():
        print(f", depth {depth[valid].min():.3f} .. {depth[valid].max():.3f} m", end="")
    print()

    if args.show3d:
        show_3d(depth, args.depth, args.stride)
        return

    out = args.out or args.depth.with_suffix({"list": ".txt", "grid": ".csv", "xlsx": ".xlsx"}[args.format])
    if out.resolve() == args.depth.resolve():
        raise SystemExit(f"refusing to overwrite the input {out}; pass --out")
    if args.format == "xlsx":
        write_xlsx(depth, out, args.decimals, args.depth)
    else:
        write_text(depth, out, args.format, args.decimals)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
