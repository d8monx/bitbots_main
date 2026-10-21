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

## Run

```bash
.venv/bin/python scripts/sim_render.py                          # RGB + ground-truth depth from the MuJoCo world -> data/sim/
KMP_DUPLICATE_LIB_OK=TRUE .venv/bin/python scripts/eval_da3.py  # DA3 Small/Base/Metric-Large, metrics + figure -> results/
```

`KMP_DUPLICATE_LIB_OK=TRUE` works around an OpenMP clash caused by `pycolmap`, which DA3 imports for its COLMAP export only.
