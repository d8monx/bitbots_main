# MDE evaluation

Offline evaluation for the Bachelor project "Monocular Depth Estimation" (WiSe 2026/27): Depth Anything 3 (DA3) on images from the Pi plus, compared with ground-truth depth and the ZED Mini stereo depth.

This folder is independent of the ROS workspace and runs on macOS (Apple Silicon, MPS) and Linux.

## Setup

```bash
cd mde_evaluation
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
# DA3 without its CUDA-only dependency xformers
uv pip install --python .venv/bin/python --no-deps "depth-anything-3 @ git+https://github.com/ByteDance-Seed/Depth-Anything-3"
```

On Linux with an NVIDIA GPU, install DA3 with its dependencies instead (drop `--no-deps`); xformers is available there.
The scripts pick the device automatically (`scripts/gpu.py`): NVIDIA GPU (CUDA), then Apple GPU (MPS), then CPU.

## Run

```bash
.venv/bin/python scripts/sim_render.py                          # RGB + ground-truth depth from the MuJoCo world -> data/sim/
KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python scripts/eval_da3.py  # DA3 Small/Base/Metric-Large, metrics + figure -> results/
```

Real recordings (rosbags in `data/rosbags/`, not in git):

```bash
.venv/bin/python scripts/extract_rosbag.py data/rosbags/real_26-11-04/<bag>       # frames, ZED depth, camera pose -> data/frames/
KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python scripts/eval_real.py          # DA3, ZED and IPM against tape and ground plane -> results/real_26-11-04/
KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python scripts/pitch_sensitivity.py  # effect of a wrong head pitch on the ball distance
KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python scripts/depth_to_excel.py     # one Excel per recording: image, distances, depth per pixel of ZED and DA3 -> results/tiefe_excel_26-11-04/
KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python scripts/make_tables.py        # table for humans, one row per measurement with image -> results/real_26-11-04/Messungen.xlsx
```

`eval_real.py` computes the DA3 depth maps once and keeps them in `data/da3/`. The tape measurements and object markings are in `annotations/`.

`KMP_DUPLICATE_LIB_OK=TRUE` works around an OpenMP clash caused by `pycolmap`, which DA3 imports for its COLMAP export only.
