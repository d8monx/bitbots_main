# Integrating Depth Anything 3 as `bitbots_depth`

This document describes how
[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) (DA3)
monocular depth estimation is integrated into this workspace. DA3 is exported
to ONNX outside the workspace and runs inside a new ROS 2 package,
`bitbots_depth`, on ONNX Runtime, reusing the inference approach of
`bitbots_vision`. PyTorch is not part of the workspace or the robot software.

The
[GerdsenAI ROS 2 DA3 wrapper](https://github.com/GerdsenAI/GerdsenAI-Depth-Anything-3-ROS2-Wrapper)
is **not** imported or run. A fork of it is used only as offline export and
reference tooling.

It is a plan, not a finished integration. Steps that touch dependencies,
hardware, or robot configuration still need team confirmation before they are
implemented (see [AGENTS.md](AGENTS.md)).

---

## 1. Decisions

| # | Decision | Consequence for this plan |
|---|----------|---------------------------|
| D1 | **Target compute:** NVIDIA Jetson GPU on the robot. Developer PCs with an NVIDIA GPU are used for simulation testing and example images. | Both `linux-aarch64` (Jetson) and `linux-64` (dev PC) must be supported. See §9 for the execution provider per platform. |
| D2 | **Use:** distances to obstacles, robots and the ball. **Not** used for localisation (IPM already covers it). | Depth must be **metric** at the detected objects. A metric model or scale recovery is required (§10). Localisation packages stay unchanged. |
| D3 | **Model:** CC BY-NC 4.0 is acceptable (university project), so every DA3 variant may be used. DA3-Small and DA3-Base are considered safe for the compute budget; DA3-Large may fit. The final choice follows benchmarks. | The pipeline must support several variants with no code change (model = data package). A benchmark step is part of the plan (§11). |
| D4 | **No PyTorch** in the workspace. Depth-specific additions go into a **dedicated Pixi feature** rather than `feature.ros`. | Runtime uses ONNX Runtime only. The dedicated feature holds the DA3 model packages (§7). |
| D5 | **Fork** the GerdsenAI wrapper. Do **not** run the wrapper on the robot (or in the workspace at all). The fork is used only as a source for ONNX export and reference code. | No `src/lib` import, no `.gitrepo`, no wrapper patches in this repository. Export runs in the fork's own environment (§4). |
| D6 | **ONNX export** plus a new package **`bitbots_depth`** instead of extending `bitbots_vision`. | New `ament_cmake` package modelled on `bitbots_vision` (§6). `bitbots_vision` stays unchanged. |

---

## 2. Starting point

### 2.1 This workspace

| Aspect | Current state | Where |
|--------|---------------|-------|
| ROS distro / Python | ROS 2 **Jazzy** via RoboStack, Python 3.12, `numpy <2` | [pixi.toml](pixi.toml) |
| Env / build | Pixi environments `default`, `format`, `robot`; `colcon` via `pixi run build` / `pixi run test` | [pixi.toml](pixi.toml) `[tasks]`, `[environments]` |
| Platforms | `linux-64` and `linux-aarch64` (CI builds on both) | [pixi.toml](pixi.toml), [.github/workflows/ci.yml](.github/workflows/ci.yml) |
| Camera | ZED Mini via imported `zed-ros2-wrapper`, ZED SDK stereo depth disabled | [vision.launch](src/bitbots_misc/bitbots_bringup/launch/vision.launch), [common_stereo.yaml](src/lib/zed-ros2-wrapper/zed_wrapper/config/common_stereo.yaml) |
| Image topics | `zed/zed_node/rgb/image_rect_color` (`bgra8`) and `zed/zed_node/rgb/camera_info`; the MuJoCo sim publishes the same topics | [visionparams.yaml](src/bitbots_vision/config/visionparams.yaml), [simulation.py](src/bitbots_simulation/bitbots_mujoco_sim/bitbots_mujoco_sim/simulation.py) |
| C++ NN inference | `bitbots_vision` runs RF-DETR through ONNX Runtime (`libonnxruntime-webgpu`, self-built, `conda-misc` channel) with a TensorRT → CUDA → WebGPU → CPU fallback chain | [rfdetr_handler.cpp](src/bitbots_vision/src/rfdetr_handler.cpp), [rfdetr_vision.rst](src/bitbots_vision/docs/manual/rfdetr_vision.rst), [CMakeLists.txt](src/bitbots_vision/CMakeLists.txt) |
| Python NN inference | conda-forge `onnxruntime` (CPU only) | [pixi.toml](pixi.toml) `[feature.ros.dependencies]` |
| Model distribution | Model directories shipped as `bitbots_model_*` conda packages from `conda-misc`; the vision node resolves a package name to `$CONDA_PREFIX/share/bitbots_model_<name>` | [pixi.toml](pixi.toml), [vision_parameters.yaml](src/bitbots_vision/config/vision_parameters.yaml) |
| 3D object positions | `soccer_ipm` projects 2D detections onto the ground plane and publishes `soccer_vision_3d_msgs` | [ipm.launch](src/bitbots_misc/bitbots_ipm/launch/ipm.launch), [src/lib/soccer_ipm](src/lib/soccer_ipm) |
| GPU monitoring | `system_monitor` supports Jetson GPUs via sysfs | [gpu.py](src/bitbots_misc/system_monitor/system_monitor/gpu.py) |

**Important finding on ONNX Runtime providers:** the `libonnxruntime-webgpu`
package (both `linux-64` and `linux-aarch64`) ships only the core
`libonnxruntime.so`, without the separate CUDA/TensorRT provider libraries.
The TensorRT/CUDA registration in `RfdetrHandler` therefore falls back, and
the effective GPU path today is **WebGPU (Vulkan)**, on the dev PC and the
Jetson alike. See §9 for what this means for DA3.

### 2.2 DA3 model family (from the upstream model cards)

| Variant | Output | Licence | Notes |
|---------|--------|---------|-------|
| `DA3-SMALL` | relative depth (+ confidence, ray) | Apache 2.0 | cheapest |
| `DA3-BASE` | relative depth (+ confidence, ray) | Apache 2.0 | |
| `DA3-LARGE-1.1` | relative depth (+ confidence, ray) | CC BY-NC 4.0 | prefer `-1.1` over the deprecated original checkpoint |
| `DA3METRIC-LARGE` | **metric** depth (needs focal length) | Apache 2.0 | same backbone size as Large |
| `DA3MONO-LARGE` | relative monocular depth | Apache 2.0 | same backbone size as Large |
| `DA3-GIANT-1.1`, `DA3NESTED-GIANT-LARGE-1.1` | relative / metric | CC BY-NC 4.0 | far too large for the robot |

Upstream states that `DA3METRIC-LARGE` output converts to metres via
`focal * net_output / 300`, with `focal` in pixels. Verify how this interacts
with input resizing during export (§4).

---

## 3. Architecture overview

```text
  fork of GerdsenAI wrapper (outside this repo, own env with PyTorch)
      │  export + parity check per variant / resolution / precision
      ▼
  <model_dir>/onnx/da3.onnx + model_config.yaml
      │  rattler-build
      ▼
  conda-misc: bitbots_model_da3_<variant>   ──►  Pixi feature `depth` (this repo)
                                                   │
  zed/zed_node/rgb/image_rect_color ─┐             ▼
  zed/zed_node/rgb/camera_info  ─────┴──►  bitbots_depth (C++, ONNX Runtime)
                                                   │
                       depth image / camera_info / confidence (metric, §10)
                                                   │
                         ┌─────────────────────────┴──────────────┐
                         ▼                                        ▼
         object distances (balls/robots, §10.3)        obstacles (later, §10.4)
```

---

## 4. ONNX export (fork, outside the workspace)

1. Fork the GerdsenAI wrapper into the team's GitHub organisation and pin the
   upstream DA3 commit it builds on. Its `scripts/build_tensorrt_engine.py`
   already contains an ONNX export step that serves as the starting point.
   [ika-rwth-aachen/ros2-depth-anything-v3-trt](https://github.com/ika-rwth-aachen/ros2-depth-anything-v3-trt)
   is a second reference for a DA3 ONNX/TensorRT export and its pre-processing.
2. Keep PyTorch, DA3, and the export dependencies in the fork's own
   environment (e.g. its own `pixi.toml` or `requirements.txt`). Nothing from
   it is added to this repository's [pixi.toml](pixi.toml).
3. Adapt the export script so that it:
   - wraps the DA3 network for a **single view** and exports only the outputs
     we need (depth, and confidence if useful). Drop the ray/camera heads.
   - uses a **fixed input resolution** that is a multiple of the ViT patch size
     and matches the camera aspect ratio, so the node can resize without
     letterboxing and the camera intrinsics stay easy to scale. Several
     resolutions may be exported for benchmarking (§11).
   - bakes the input normalisation (and channel order) into the graph, so the
     node only has to resize and convert to float. This prevents mismatches
     between export and runtime pre-processing.
   - targets an opset supported by the ONNX Runtime version in
     [pixi.toml](pixi.toml), simplifies the graph, and produces fp32 and fp16
     variants (consistent with the RF-DETR fp16 exports).
   - runs a **parity check** of ONNX Runtime vs. PyTorch on sample frames
     (rosbag and MuJoCo images), including the metric conversion for
     `DA3METRIC-LARGE`.
   - writes the model directory described in §5.
4. Check ONNX Runtime's session logs with the WebGPU provider for nodes that
   fall back to CPU. Unsupported operators silently split the graph and
   destroy performance. Fix them in the export wrapper where possible.

---

## 5. Model packages (offline availability)

Robots have no internet access at competitions, so models ship as conda
packages on the self-hosted `conda-misc` channel, following the existing
`bitbots_model_*` packages.

Proposed model directory layout, mirroring the RF-DETR layout in
[rfdetr_vision.rst](src/bitbots_vision/docs/manual/rfdetr_vision.rst):

```text
<model_dir>/
  onnx/da3.onnx
  model_config.yaml
  LICENSE
```

`model_config.yaml` describes what the node needs to interpret the model
without code changes, e.g.:

- variant name and upstream checkpoint / commit,
- input width/height and expected channel order,
- output names and output kind (`relative` or `metric`),
- for metric models: the conversion constant and the reference resolution the
  focal length refers to,
- licence identifier.

Package names should encode variant, resolution, and precision (e.g.
`bitbots_model_da3_small_<res>_fp16`). Never commit ONNX files or TensorRT
engines to this repository.

---

## 6. New package `bitbots_depth`

### 6.1 Location and build

- `src/bitbots_depth/` (sibling of `src/bitbots_vision`), `ament_cmake`, C++.
- C++ instead of Python because the only GPU-capable ONNX Runtime in the
  workspace is the C++ `libonnxruntime-webgpu`. The Python `onnxruntime` is
  CPU only.
- Reuse the `bitbots_vision` patterns:
  - ONNX Runtime discovery from
    [CMakeLists.txt](src/bitbots_vision/CMakeLists.txt),
  - a processing library **without** ONNX Runtime for unit tests, plus a
    separate inference library (as `bitbots_vision` does),
  - `generate_parameter_library` for parameters,
  - model path resolution from a conda package name, as in
    [vision_parameters.yaml](src/bitbots_vision/config/vision_parameters.yaml).
- The execution-provider fallback logic from `RfdetrHandler` is copied at
  first. Extracting it into a shared helper library is a possible follow-up
  refactor, but it is out of scope here so that `bitbots_vision` stays
  untouched.
- `package.xml` dependencies: `rclcpp`, `sensor_msgs`, `cv_bridge`,
  `generate_parameter_library`, `tf2_ros` (for scale recovery, §10.2), plus
  `soccer_vision_3d_msgs` / `soccer_vision_2d_msgs` if object distances are
  computed here (§10.3). All of these already exist in the workspace.

### 6.2 Node behaviour

- Subscribes to the camera image and `camera_info` (topic names as
  parameters, defaults matching the ZED / MuJoCo topics). The subscription
  uses a sensor-data QoS and keeps only the latest frame.
- Optional rate limiting / frame skipping, so DA3 does not starve RF-DETR on
  the shared GPU (§9).
- Pre-processing: `bgra8` → RGB, resize to the model input resolution, and
  convert to a float tensor (normalisation is in the graph, §4).
- Inference through ONNX Runtime with the provider chain described in §9.
- Post-processing:
  - metric models: convert to metres using the focal length from
    `camera_info`, scaled to the model input resolution (§10.1).
  - relative models: per-frame scale recovery (§10.2).
- Publishes, with the input image header (same stamp and `frame_id`):
  - `depth/image` (`32FC1`, metres),
  - `depth/camera_info` (intrinsics scaled to the depth image resolution),
  - optional `depth/confidence` (`32FC1`),
  - optional `depth/debug_image` (colourised, only for debugging because of
    bandwidth).
- A parameter controls whether depth is published at model resolution
  (cheaper; consumers use the scaled `camera_info`) or resized to the camera
  resolution.
- Logs inference time and the active execution provider at start-up, as
  `bitbots_vision` does.

### 6.3 Tests

- gtest unit tests for the processing library: intrinsics scaling,
  metric conversion, scale-recovery estimator (synthetic data), pixel ↔ ray
  back-projection, and model config parsing.
- Inference tests that need a model package are skipped when the model is not
  installed, so `pixi run -e default test` works in CI without the `depth`
  feature.

---

## 7. Dependencies (Pixi)

> Per [AGENTS.md](AGENTS.md#dependencies), confirm the exact feature and
> environment layout with the team **before** editing
> [pixi.toml](pixi.toml) or regenerating `pixi.lock`.

`bitbots_depth` needs no new runtime libraries. ONNX Runtime, OpenCV,
`cv_bridge`, TF, and the soccer message packages are already in
`feature.ros`, so the package builds in the `default` environment and in CI.

Per D4, the DA3 model packages go into a dedicated feature so that `default`
and CI do not download large models:

```toml
[feature.depth.dependencies]
# DA3 model packages from conda-misc (variant(s) chosen after benchmarking, §11)
bitbots_model_da3_<variant> = "..."

[environments]
default = ["ros", "format"]
format = ["format"]
robot = ["ros", "format", "robot"]               # add "depth" once a model is chosen
depth = ["ros", "format", "depth"]               # dev PC / simulation testing
```

To be confirmed:

- Whether the robot gets the `depth` feature directly in `robot` or a separate
  `robot_depth` environment during the evaluation phase. Check how
  `scripts/deploy_robots.py` selects the environment.
- Whether benchmark models (several variants/resolutions) live in the feature
  or are installed only on demand.

No PyTorch, no PyPI packages, and no CUDA `system-requirements` are added.

---

## 8. Launch and configuration

### 8.1 Parameter file

Canonical defaults go in the `generate_parameter_library` definition, e.g.
`src/bitbots_depth/config/depth_parameters.yaml`. The deployed values go in
`src/bitbots_depth/config/depthparams.yaml`, following the
`bitbots_vision` split between `vision_parameters.yaml` and
`visionparams.yaml`. Document parameters by purpose, not by concrete values.

### 8.2 Launch file

`src/bitbots_depth/launch/depth.launch` (XML, like `bitbots_vision`):

```xml
<launch>
    <arg name="sim" default="false" description="true: activates simulation time" />

    <node pkg="bitbots_depth" exec="depth_node" name="bitbots_depth" output="screen">
        <param from="$(find-pkg-share bitbots_depth)/config/depthparams.yaml" />
        <param name="use_sim_time" value="$(var sim)" />
    </node>
</launch>
```

### 8.3 Wiring into the bringup

1. Add a `depth` arg (default `false` until performance is validated) to
   [bitbots_bringup/launch/vision.launch](src/bitbots_misc/bitbots_bringup/launch/vision.launch)
   and include `bitbots_depth/launch/depth.launch` when it is set, passing
   `sim`.
2. Propagate the arg through
   [vision_standalone.launch](src/bitbots_misc/bitbots_bringup/launch/vision_standalone.launch),
   [highlevel.launch](src/bitbots_misc/bitbots_bringup/launch/highlevel.launch),
   [teamplayer.launch](src/bitbots_misc/bitbots_bringup/launch/teamplayer.launch),
   and `TEAMPLAYER_ARGS` in
   [mujoco_simulation.launch.py](src/bitbots_misc/bitbots_bringup/launch/mujoco_simulation.launch.py).
3. Add `<exec_depend>bitbots_depth</exec_depend>` to
   [bitbots_bringup/package.xml](src/bitbots_misc/bitbots_bringup/package.xml).
   No condition is needed because the package builds everywhere. Only the
   model is environment-specific.
4. Simulation: the node runs in each robot's domain on the bridged camera
   topics, so the domain-bridge config needs no change unless depth should be
   visible in the main domain.

### 8.4 Monitoring / recording (optional)

- Add `depth/image` (and optionally `depth/confidence`) to `TOPICS_TO_RECORD`
  in
  [rosbag_record.launch.py](src/bitbots_misc/bitbots_bringup/launch/rosbag_record.launch.py)
  if needed. Mind the bag size (float images).
- Add the debug image to
  [monitoring.rviz](src/bitbots_misc/bitbots_bringup/config/monitoring.rviz) /
  [monitoring_foxglove.json](src/bitbots_misc/bitbots_bringup/config/monitoring_foxglove.json).

---

## 9. Execution provider per platform

| Platform | Today (`libonnxruntime-webgpu`) | Possible improvement |
|----------|---------------------------------|----------------------|
| Dev PC, NVIDIA GPU (`linux-64`) | WebGPU over Vulkan; CPU fallback | Sufficient for simulation and example images. Dev-PC timings are **not** representative of the Jetson. |
| Jetson (`linux-aarch64`) | WebGPU over Vulkan (Jetson Vulkan driver); CPU fallback | ONNX Runtime with the TensorRT (or CUDA) provider built against the Jetson's JetPack/TensorRT, shipped e.g. as an additional `conda-misc` package or via the robot setup. `bitbots_vision` would benefit too. Needs a separate team decision. |

Notes:

- RF-DETR and DA3 share one Jetson GPU. Measure both together (§11) and use
  the node's rate limiting if needed.
- If the TensorRT provider is adopted, enable its engine cache in a
  robot-local directory. Engines are device- and TensorRT-version-specific:
  build them on the robot and never commit them.
- Check WebGPU fp16 support on the Jetson before relying on fp16 exports.
  `RfdetrHandler` already handles session creation failures with a CPU
  retry.

---

## 10. Metric depth and downstream use (D2)

### 10.1 Metric model

`DA3METRIC-LARGE` (Apache 2.0) predicts metric depth given the focal length.
The node takes `fx`/`fy` from `camera_info`, scales them to the model input
resolution, and applies the conversion defined in `model_config.yaml`. The
parity check in §4 must confirm which resolution the focal length refers to.
Its cost is roughly that of DA3-Large, so it is only an option if Large fits
the Jetson budget (D3).

### 10.2 Scale recovery for relative models (Small / Base / Large)

The main-series models predict depth up to an unknown scale. The camera pose
relative to the ground plane is known from TF (this is what IPM uses), so:

1. Select pixels that are known to lie on the ground plane. Line-mask pixels
   from `bitbots_vision` (`line_mask_in_image`) are a good choice; field
   pixels outside detection boxes are an alternative.
2. Compute their expected metric depth by ray-plane intersection, using
   `camera_info` and the TF camera pose at the image stamp.
3. Estimate a robust per-frame scale (e.g. median ratio) and optionally
   smooth it over time. Reject the frame if too few ground pixels are
   available or the camera looks too far upward.
4. Verify experimentally that a pure scale is sufficient (no shift term)
   for each variant.

Both 10.1 and 10.2 should be implemented behind the `relative`/`metric`
output kind from `model_config.yaml`, so the variant can be switched by
changing the model package only.

### 10.3 Object distances (first downstream consumer)

- Synchronise `depth/image` with `balls_in_image` / `robots_in_image` by
  exact stamp (all share the input image header).
- Sample a robust depth inside each detection (e.g. median of the central
  region or the lower part of the box). Back-project with `camera_info` and
  transform into the target frame via TF.
- Publish `soccer_vision_3d_msgs`, as `soccer_ipm` does, so that the world
  model and behaviour can consume depth-based positions without new message
  types.
- To be decided: replace IPM-based positions, fuse both, or use depth only
  where IPM's ground-plane assumption fails (occluded feet, airborne ball,
  objects far from the plane). Start by publishing on separate topics and
  comparing.

### 10.4 Obstacles (later)

Convert depth + `camera_info` into a point cloud or a ground-projected
occupancy for `bitbots_navigation` path planning. `depth_image_proc` would be
a new dependency, so it needs confirmation per [AGENTS.md](AGENTS.md#dependencies).
This is out of scope for the first step.

Localisation (`bitbots_localization`) is not changed (D2).

---

## 11. Benchmark and model selection (D3)

Run for each candidate (`DA3-SMALL`, `DA3-BASE`, `DA3-LARGE-1.1`,
`DA3METRIC-LARGE`), each export resolution, and fp32/fp16:

1. **Dev PC:** functional check in MuJoCo and on rosbags. Use the
   simulation's ground-truth ball/robot positions to measure distance error
   (note the sim-to-real gap).
2. **Jetson** (only with explicit approval; no motion needed): run
   `vision_standalone.launch depth:=true` and record
   - DA3 latency and achieved rate,
   - RF-DETR rate with and without DA3,
   - GPU/CPU load and memory through `system_monitor`,
   - distance error against measured real distances on recorded scenes.
3. Choose the variant with the best accuracy that keeps the RF-DETR rate
   acceptable. Then pin its model package in the `depth` feature and record
   the results (without hard-coded numbers in code comments) in the docs.

---

## 12. Documentation

- `src/bitbots_depth/README.md` and a manual page (e.g.
  `src/bitbots_depth/docs/manual/`) describing the node, the topics, the model
  directory layout, metric vs. relative handling, the `depth` Pixi
  environment, and the export workflow (with a link to the fork).
- State each model's licence. CC BY-NC 4.0 variants are acceptable for this
  project but must not be redistributed for commercial use.
- Document the new `depth` launch arg in the bringup docs.

---

## 13. Validation checklist

1. Fork: ONNX export plus parity check pass for the candidate variants.
2. `pixi install -e depth` solves on `linux-64` and `linux-aarch64`.
3. `pixi run -e default build --packages-select bitbots_depth` and
   `pixi run -e default test --packages-select bitbots_depth` pass without
   a model installed.
4. `pixi run -e depth test --packages-select bitbots_depth` passes with the
   model installed.
5. Offline: play a rosbag with the ZED image and `camera_info` topics, run
   `depth.launch`, then check `ros2 topic hz`, stamps and frames on
   `depth/image`, and the active execution provider in the log.
6. Simulation: `ros2 launch bitbots_bringup mujoco_simulation.launch.py`
   with `depth:=true`. Verify `use_sim_time` and compare object distances
   with ground truth.
7. Jetson: benchmark as in §11 (explicit approval required).
8. `pixi run -e default build` and `pixi run -e default test` still pass for
   the whole workspace.
9. `pixi run -e default format`, then review the diff.

---

## 14. Summary of concrete changes

| File / location | Change |
|-----------------|--------|
| Fork of GerdsenAI wrapper (separate repo) | ONNX export + parity tooling; PyTorch lives only there |
| `conda-misc` channel | `bitbots_model_da3_*` packages (ONNX + `model_config.yaml` + licence) |
| `src/bitbots_depth/` (new) | C++ node, processing lib, parameters, launch, config, tests, docs |
| [pixi.toml](pixi.toml) / `pixi.lock` | New `depth` feature (model packages) + `depth` environment; later added to the robot environment (after team approval) |
| [bitbots_bringup launch files](src/bitbots_misc/bitbots_bringup/launch) | `depth` arg propagated through vision/standalone/highlevel/teamplayer/MuJoCo launches |
| [bitbots_bringup/package.xml](src/bitbots_misc/bitbots_bringup/package.xml) | `exec_depend` on `bitbots_depth` |
| [rosbag_record.launch.py](src/bitbots_misc/bitbots_bringup/launch/rosbag_record.launch.py), monitoring configs | Optional topics / visualisation |
| Robot setup (only if adopted) | ONNX Runtime build with TensorRT provider for Jetson (§9) |

---

## 15. Remaining open questions

1. Pixi layout for the robot during evaluation: add `depth` to `robot` or
   create `robot_depth`? (§7)
2. Jetson: stay with WebGPU, or build ONNX Runtime with the TensorRT provider?
   (§9)
3. Object distances: computed in `bitbots_depth` or in a separate node, and
   replace, fuse with, or complement `soccer_ipm`? (§10.3)
4. Name and owner of the fork, and who builds and uploads the `conda-misc`
   model packages. (§4, §5)
5. Export resolution(s) and whether DA3 runs on every frame or rate-limited.
   (§4, §6.2, §11)
