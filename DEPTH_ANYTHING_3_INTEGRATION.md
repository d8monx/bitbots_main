# Integrating Depth Anything 3 via the GerdsenAI ROS 2 Wrapper

This document describes what has to be done to run
[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) (DA3)
monocular depth estimation in this workspace using the
[GerdsenAI ROS 2 DA3 wrapper](https://github.com/GerdsenAI/GerdsenAI-Depth-Anything-3-ROS2-Wrapper)
(ROS package `depth_anything_3_ros2`).

It is a plan, not a finished integration. Steps that touch dependencies,
hardware, or robot configuration need team confirmation before they are
implemented (see [AGENTS.md](AGENTS.md)).

---

## 1. Starting point

### 1.1 This workspace

| Aspect | Current state | Where |
|--------|---------------|-------|
| ROS distro | ROS 2 **Jazzy** via RoboStack | [pixi.toml](pixi.toml) (`robostack-jazzy` channel, `ros-jazzy-*` deps) |
| Python | 3.12, `numpy <2` | [pixi.toml](pixi.toml) `[dependencies]`, `[feature.ros.dependencies]` |
| Env / build | Pixi environments `default`, `format`, `robot`; `colcon` via `pixi run build` / `pixi run test` | [pixi.toml](pixi.toml) `[tasks]`, `[environments]` |
| Platforms | `linux-64` and `linux-aarch64` (CI builds on both) | [pixi.toml](pixi.toml), [.github/workflows/ci.yml](.github/workflows/ci.yml) |
| RMW | `rmw_zenoh_cpp` | [pixi.toml](pixi.toml) `[feature.ros.activation.env]` |
| Camera | ZED Mini via imported `zed-ros2-wrapper`, ZED SDK stereo depth disabled (`depth_mode: 'NONE'`) | [vision.launch](src/bitbots_misc/bitbots_bringup/launch/vision.launch), [common_stereo.yaml](src/lib/zed-ros2-wrapper/zed_wrapper/config/common_stereo.yaml) |
| Image topics | `zed/zed_node/rgb/image_rect_color` (`bgra8`) and `zed/zed_node/rgb/camera_info`; the MuJoCo sim publishes the same topics | [visionparams.yaml](src/bitbots_vision/config/visionparams.yaml), [simulation.py](src/bitbots_simulation/bitbots_mujoco_sim/bitbots_mujoco_sim/simulation.py) |
| Existing NN inference | `bitbots_vision` (C++) runs RF-DETR through ONNX Runtime with a TensorRT → CUDA → WebGPU → CPU fallback chain; no PyTorch in the stack | [rfdetr_vision.rst](src/bitbots_vision/docs/manual/rfdetr_vision.rst) |
| Model distribution | NN weights are shipped as conda packages from the self-hosted `conda-misc` channel (e.g. `bitbots_model_*`) | [pixi.toml](pixi.toml) channels + "Neural network models" section |
| Imported libs | Third-party ROS packages live in `src/lib/<name>` with a `.gitrepo` file (git-subrepo / `git-include`), excluded from pre-commit and from `pixi run test` | [src/lib/zed-ros2-wrapper/.gitrepo](src/lib/zed-ros2-wrapper/.gitrepo), [.pre-commit-config.yaml](.pre-commit-config.yaml), [pixi.toml](pixi.toml) `[tasks.test]` |
| Launch hierarchy | `teamplayer.launch` → `highlevel.launch` → `bitbots_bringup/vision.launch` → `bitbots_vision/vision.launch` (+ ZED driver) | [src/bitbots_misc/bitbots_bringup/launch/](src/bitbots_misc/bitbots_bringup/launch) |
| Robot compute | Docs list Intel NUC hosts with Vulkan drivers; the code base also contains Jetson/aarch64 support (system monitor, ZED CMake fix, `linux-aarch64` platform) | [robots.rst](src/bitbots_misc/bitbots_docs/docs/manual/physical_robot/robots.rst), [setup_robot.rst](src/bitbots_misc/bitbots_docs/docs/manual/physical_robot/setup_robot.rst), [gpu.py](src/bitbots_misc/system_monitor/system_monitor/gpu.py) |

### 1.2 The GerdsenAI wrapper (upstream `main`)

- `ament_python` package `depth_anything_3_ros2`, MIT licensed, targets
  **ROS 2 Humble / Ubuntu 22.04 / Python 3.10**.
- Single node `depth_anything_3_node` (node name `depth_anything_3`):
  - Subscribes `~/image_raw` (`sensor_msgs/Image`, BEST_EFFORT) and
    `~/camera_info`.
  - Publishes `~/depth` (`32FC1`), `~/depth_colored` (`bgr8`),
    `~/confidence` (`32FC1`), `~/depth/camera_info`. Output headers reuse the
    input image header (same `frame_id` and stamp).
  - Parameters: `model_name`, `device` (`cuda`|`cpu`), `cache_dir`,
    `inference_height/width`, `input_encoding`, `keep_image_size`,
    `normalize_depth`, `publish_colored`, `publish_confidence`, `colormap`,
    `queue_size`, `processing_threads`, `log_inference_time`,
    `use_shared_memory`.
- Three inference backends in `da3_inference.py`:
  1. `DA3InferenceWrapper` – PyTorch, loads the model with
     `depth_anything_3.api.DepthAnything3.from_pretrained(...)` (Hugging Face
     download unless a local path is given).
  2. `SharedMemoryInference` / `SharedMemoryInferenceFast` – talk via
     `/dev/shm/da3` to a **separate host process**
     (`scripts/trt_inference_service_shm.py`) that runs a TensorRT engine
     (Jetson / JetPack 6 oriented, needs host `tensorrt` + `pycuda`).
- Runtime Python deps (from `requirements.txt` and DA3's `pyproject.toml`):
  `torch`, `torchvision`, `transformers`, `huggingface_hub`, `safetensors`,
  `timm`, `opencv-python`, `pillow`, `numpy<2`, `pyyaml`, `onnx`, `gsplat`,
  and the `depth-anything-3` package itself (only installable from git), which
  in turn pulls `xformers`, `open3d`, `pycolmap`, `evo`, `e3nn`, `trimesh`,
  `moviepy==1.0.3`, `plyfile`, `pillow_heif`, `omegaconf`, `einops`,
  `fastapi`, `uvicorn`, `typer`, `pre-commit`, ….
- Observed issues relevant for us (verify against the pinned commit):
  - The upstream launch file does **not** pass `use_sim_time`.
  - `launch_ros` is used but not declared in `package.xml`; `image_transport`
    is declared but unused.
  - `inference_height/width`, `input_encoding` and `processing_threads` are
    read but not used by the PyTorch backend (DA3 resizes internally, the node
    always converts to `rgb8`).
  - Default `normalize_depth: true` → output is **relative** depth in `[0, 1]`
    per frame, not metres.
  - Only `DA3-SMALL` weights are Apache-2.0; Base/Large/Giant/Metric/Nested are
    CC-BY-NC-4.0.

---

## 2. Decisions required before implementation

| # | Question | Why it matters |
|---|----------|----------------|
| D1 | **Target compute** for DA3: robot (which board – Intel NUC or Jetson?), a developer PC with an NVIDIA GPU, or simulation only? | Determines whether the PyTorch CUDA path, CPU path, or Jetson TensorRT host service is used, and which platform-specific packages are needed. On an Intel NUC without an NVIDIA GPU only the CPU backend works, which is unlikely to be real-time. |
| D2 | **What is the depth used for?** (visualisation/research, obstacle/robot distance, ball distance, localisation) | Decides whether relative depth is enough or a metric model (CC-BY-NC) / scale recovery is required, and which downstream packages must change. |
| D3 | **Model variant** and licence | `DA3-SMALL` is the only permissively licensed model and the cheapest to run. |
| D4 | **Pixi feature / environment** for the heavy PyTorch stack | Adding PyTorch & co. to `feature.ros` would bloat every developer and CI environment. A dedicated feature/environment is recommended (see §4). Needs confirmation per [AGENTS.md](AGENTS.md#dependencies). |
| D5 | **Import strategy**: vendor the wrapper in `src/lib` (git-subrepo/`git-include`, consistent with other third-party ROS packages) vs. a fork under `bit-bots/` | We need several small patches (§3.2). A fork lets us keep them and upstream them; vendoring directly from GerdsenAI means local patches in `src/lib` which [AGENTS.md](AGENTS.md#repository-boundaries) discourages. |
| D6 | Long-term backend: keep PyTorch, or export DA3 to ONNX and run it via ONNX Runtime like `bitbots_vision` | ONNX Runtime with TensorRT/WebGPU is already in the stack and works on the robot hardware; PyTorch is not. See §9. |

---

## 3. Import the wrapper package

### 3.1 Add the sources

1. Decide D5. Recommended: fork to `bit-bots/GerdsenAI-Depth-Anything-3-ROS2-Wrapper`
   (or similar) and pin a specific commit.
2. Import it as a subrepo, analogous to the existing entries:

   ```bash
   pixi run -e default git subrepo clone \
     <fork-or-upstream-url> src/lib/depth_anything_3_ros2 -b <branch>
   ```

   This creates `src/lib/depth_anything_3_ros2/.gitrepo` (compare
   [src/lib/zed-ros2-wrapper/.gitrepo](src/lib/zed-ros2-wrapper/.gitrepo)).
   Use the repository's `git-include` tooling instead if that is the preferred
   workflow for new imports.
3. Remove or ignore content that colcon / the deploy rsync should not pick up
   (Docker files, `desktop/`, demo scripts are harmless but large assets or
   generated TensorRT engines must not be committed). Check
   [.rsyncignore](.rsyncignore) if anything large must not be synced to robots.

### 3.2 Required patches to the wrapper (ideally upstreamed)

| Patch | Reason |
|-------|--------|
| Declare `launch_ros` (`exec_depend`) and drop the unused `image_transport` dependency in `package.xml` | Keep manifest and runtime imports in sync. |
| Add `use_sim_time` launch argument/parameter to `depth_anything_3.launch.py` (or just don't use the upstream launch file, see §6) | Required for MuJoCo simulation. |
| Allow `model_name` to be a **local directory** for offline use and make sure no network access happens at start-up | Robots have no internet at competitions (see §5). Verify how `DepthAnything3.from_pretrained` handles local paths. |
| Either forward `inference_height/width` to DA3 (e.g. `process_res`) or remove the parameters | Currently ineffective for the PyTorch backend; resolution is the main performance knob. |
| Make heavy imports (`torch`, `depth_anything_3`) lazy where possible | Lets `colcon test` / import checks run without the full PyTorch stack. |
| Python 3.12 / Jazzy smoke-test fixes, if any | Upstream only tests Humble / 3.10. The node uses plain `rclpy`, so few issues are expected, but it must be verified. |
| Optional: header `frame_id` / output resizing behaviour (`keep_image_size: true`) | Downstream consumers expect depth pixels aligned with the camera image and `camera_info`. |

---

## 4. Dependencies (Pixi)

> Per [AGENTS.md](AGENTS.md#dependencies), propose the exact feature and
> environment to the team and get confirmation **before** editing
> [pixi.toml](pixi.toml) or regenerating `pixi.lock`.

### 4.1 Proposed layout

Add a new feature (e.g. `depth_anything`) and a dedicated environment instead
of extending `feature.ros`, so that `default`, `robot` and CI are not forced to
download PyTorch:

```toml
[feature.depth_anything.dependencies]
# conda-forge packages, versions to be chosen consistent with neighbouring entries
pytorch = "..."
torchvision = "..."
transformers = "..."
huggingface_hub = "..."
safetensors = "..."
timm = "..."
einops = "..."
omegaconf = "..."
# + whatever `depth_anything_3` imports at inference time (see 4.3)

[feature.depth_anything.pypi-dependencies]
depth-anything-3 = { git = "https://github.com/ByteDance-Seed/Depth-Anything-3.git", rev = "<pinned sha>" }

[environments]
default = ["ros", "format"]
format = ["format"]
robot = ["ros", "format", "robot"]
depth = ["ros", "format", "depth_anything"]           # development / desktop GPU
# robot_depth = ["ros", "format", "robot", "depth_anything"]  # only if D1 says "on robot"
```

All needed core packages were found on `conda-forge` (`pytorch`,
`torchvision`, `xformers`, `open3d`, `pycolmap`, `timm`, `transformers`,
`huggingface_hub`, `safetensors`, `einops`, `omegaconf`, `e3nn`, `evo`,
`trimesh`, `moviepy`, `plyfile`, `pillow-heif`, `gsplat`). Confirm with
`pixi search <package>` for both `linux-64` and `linux-aarch64`; at the time of
writing `xformers` showed no `linux-aarch64` build.

### 4.2 Known conflicts to resolve

- **OpenCV**: DA3 and the wrapper request `opencv-python` from PyPI, while the
  workspace uses conda `opencv`/`libopencv` and `cv_bridge`. Installing the PyPI
  wheel next to the conda build must be avoided. Check whether Pixi's
  conda↔PyPI mapping satisfies `opencv-python` with the conda package; if not,
  install `depth-anything-3` without its declared dependencies (e.g. vendor a
  fork with trimmed requirements) and list the actually needed runtime deps
  explicitly.
- **numpy**: DA3 requires `numpy<2`, which matches the current constraint.
  Keep it consistent.
- **pillow**: the workspace pins `pillow <12` (see the comment in
  [pixi.toml](pixi.toml)); make sure the solve still succeeds.
- **moviepy**: DA3 pins `moviepy==1.0.3`, which is old; conda-forge's latest
  is 2.x. Prefer trimming this dependency (it is only needed for video export).
- **Heavy, inference-irrelevant deps** (`open3d`, `pycolmap`, `evo`, `e3nn`,
  `gsplat`, `fastapi`, `uvicorn`, `pre-commit`, `xformers`): only needed for
  DA3's CLI, 3D reconstruction, or Gaussian splatting. Avoid installing them on
  robots if the import path for `DepthAnything3` inference works without them
  (verify, see 4.3).
- **PyTorch build variant**: on `linux-64` choose a CUDA-enabled conda-forge
  build if a desktop NVIDIA GPU is targeted (requires `system-requirements`
  `cuda = "12"` or similar in the feature). On Jetson, conda-forge PyTorch is
  **not** a JetPack/Tegra GPU build, so CUDA inference inside the Pixi env will
  not work there (see §7).

### 4.3 Determine the minimal runtime dependency set

Before finalising the feature, install `depth-anything-3` in a scratch env and
check which modules `depth_anything_3.api` imports transitively when only
calling `from_pretrained(...)` and `inference(...)`. Keep only those.

### 4.4 CI

- Do not add the new environment to the CI build matrix unless needed; if it
  is added, cache size and build time will grow considerably.
- The wrapper builds as `ament_python` without PyTorch present, so the
  existing `pixi run -e default build` keeps working even without the new
  feature, as long as nothing imports the module at build time.
- Add `depth_anything_3_ros2` to `--packages-skip` in `[tasks.test]` in
  [pixi.toml](pixi.toml), consistent with other imported `src/lib` packages
  (its tests import PyTorch and the Jetson/shared-memory code paths).

---

## 5. Model weights (offline availability)

At start-up `DepthAnything3.from_pretrained` downloads weights from Hugging
Face. Robots must not depend on internet access.

1. Choose the model (D3). Prefer `depth-anything/DA3-SMALL` (Apache-2.0).
2. Package the weights as a conda package on the self-hosted `conda-misc`
   channel, following the existing `bitbots_model_*` packages
   (see [pixi.toml](pixi.toml) "Neural network models / data packages" and the
   `model.path` comment in
   [visionparams.yaml](src/bitbots_vision/config/visionparams.yaml)).
3. Point `model_name` to the installed local directory (wrapper patch, §3.2)
   and set `HF_HUB_OFFLINE=1` in the feature's activation env so that no
   network request is attempted.
4. If a TensorRT engine is used (§7), engines are hardware/TensorRT-version
   specific: build them on the target device at deploy time; never commit
   them.

---

## 6. Launch and configuration integration

### 6.1 Parameter file

Create a canonical parameter file, e.g.
`src/bitbots_vision/config/depth_anything_3.yaml` (or in a small new
`bitbots_depth` package if the team prefers to keep `bitbots_vision`
dependency-free):

```yaml
depth_anything_3:
  ros__parameters:
    model_name: "<local model dir or HF id>"
    device: "cuda"            # or "cpu", depending on D1
    normalize_depth: false    # keep raw model output if downstream needs (relative/metric) values
    keep_image_size: true     # align depth pixels with the camera image / camera_info
    publish_colored: false    # enable only for debugging (bandwidth)
    publish_confidence: true
    queue_size: 1
    log_inference_time: false
```

Avoid hard-coding values in documentation; describe their purpose instead.

### 6.2 Launch file

Write our own launch file instead of including the upstream one, so we
control `use_sim_time`, parameters and remaps. Follow the existing XML style,
e.g. `src/bitbots_vision/launch/depth.launch` (or in `bitbots_bringup`):

```xml
<launch>
    <arg name="sim" default="false" description="true: activates simulation time" />

    <node pkg="depth_anything_3_ros2" exec="depth_anything_3_node" name="depth_anything_3" output="screen">
        <param from="$(find-pkg-share bitbots_vision)/config/depth_anything_3.yaml" />
        <param name="use_sim_time" value="$(var sim)" />
        <remap from="~/image_raw" to="zed/zed_node/rgb/image_rect_color" />
        <remap from="~/camera_info" to="zed/zed_node/rgb/camera_info" />
    </node>
</launch>
```

Notes:

- The input is `bgra8` (ZED and MuJoCo); the node converts to `rgb8` via
  `cv_bridge`, which supports this conversion. Verify once.
- BEST_EFFORT subscriber + RELIABLE ZED publisher is QoS compatible.
- Consider placing outputs under a clear namespace (e.g. `depth/...`) via
  remaps so they don't collide with ZED topics.

### 6.3 Wire it into the bringup

1. Add an arg `depth` (default `false` until performance is validated) to
   [bitbots_bringup/launch/vision.launch](src/bitbots_misc/bitbots_bringup/launch/vision.launch)
   and include the new depth launch file when it is set, passing `sim`.
2. Propagate the arg through
   [vision_standalone.launch](src/bitbots_misc/bitbots_bringup/launch/vision_standalone.launch),
   [highlevel.launch](src/bitbots_misc/bitbots_bringup/launch/highlevel.launch),
   [teamplayer.launch](src/bitbots_misc/bitbots_bringup/launch/teamplayer.launch),
   and `TEAMPLAYER_ARGS` in
   [mujoco_simulation.launch.py](src/bitbots_misc/bitbots_bringup/launch/mujoco_simulation.launch.py).
3. Add `<exec_depend>depth_anything_3_ros2</exec_depend>` to the package that
   owns the launch file
   ([bitbots_bringup/package.xml](src/bitbots_misc/bitbots_bringup/package.xml)
   and/or [bitbots_vision/package.xml](src/bitbots_vision/package.xml)).
   Since the package only works in the new Pixi environment, consider a
   conditional dependency like the existing
   `condition="$IS_ROBOT != True"` pattern in `bitbots_bringup/package.xml`.
4. Simulation: the depth node runs inside each robot's domain and consumes the
   bridged camera topics, so the domain-bridge config in
   [mujoco_simulation.launch.py](src/bitbots_misc/bitbots_bringup/launch/mujoco_simulation.launch.py)
   needs no change unless depth should be visible in the main domain.

### 6.4 Monitoring / recording (optional)

- Add the depth (and optionally confidence) topics to
  `TOPICS_TO_RECORD` in
  [rosbag_record.launch.py](src/bitbots_misc/bitbots_bringup/launch/rosbag_record.launch.py)
  if desired; mind the bag size (images are float32).
- Add an image display to
  [monitoring.rviz](src/bitbots_misc/bitbots_bringup/config/monitoring.rviz) /
  [monitoring_foxglove.json](src/bitbots_misc/bitbots_bringup/config/monitoring_foxglove.json)
  for `depth_colored`.

---

## 7. Inference backend per platform

| Platform | Backend in wrapper | What has to be done |
|----------|-------------------|---------------------|
| Dev PC with NVIDIA GPU (`linux-64`) | PyTorch `device: cuda` | CUDA-enabled conda-forge `pytorch` in the new feature, CUDA system requirement in Pixi. Easiest path for evaluation and simulation. |
| Intel NUC (no NVIDIA GPU) | PyTorch `device: cpu` only | Works functionally but is expected to be far from real-time and competes with the RF-DETR vision and motion for CPU. Not recommended for games; consider D6 (ONNX Runtime + WebGPU/OpenVINO). |
| Jetson (`linux-aarch64`) | Shared-memory client + host TensorRT service | The wrapper's TRT service (`scripts/trt_inference_service_shm.py`, `scripts/build_tensorrt_engine.py`) runs **outside** the Pixi env with JetPack's `tensorrt` + `pycuda`. Needs: ONNX export of DA3 and engine build on the robot, a systemd/tmux service or launch `ExecuteProcess` to start it, `use_shared_memory: true`, and the PyTorch stack is then not needed on the robot at all. Check whether conda-forge `tensorrt`/`pycuda` builds for aarch64 + JetPack could replace the host install. Ansible changes may be required. |

---

## 8. Using the depth output downstream (depends on D2)

The wrapper only publishes depth images; nothing in the stack consumes them
yet. Possible consumers and the work they imply:

- **Distances to detected objects** (balls, robots, obstacles): sample depth
  at detection boxes from `bitbots_vision` and combine with `camera_info`.
  Today, 3D positions come from inverse perspective mapping (`soccer_ipm`,
  [ipm.launch](src/bitbots_misc/bitbots_ipm/launch/ipm.launch)), which assumes
  objects on the ground plane. DA3 relative depth needs metric scale – either
  a metric model (licence!) or scale fitting against IPM on ground pixels.
- **Obstacle avoidance / path planning**: convert depth + `camera_info` into a
  point cloud (e.g. `depth_image_proc` – new dependency) and feed it to the
  planner in `bitbots_navigation`.
- **Interface changes**: if new messages are needed, add them to
  [bitbots_msgs](src/bitbots_msgs) and rebuild consumers.

Keep this out of the first integration step; first get the node running
and measured.

---

## 9. Alternative worth evaluating (D6)

The robot software already runs neural networks through ONNX Runtime with
TensorRT/CUDA/WebGPU/CPU providers (`bitbots_vision`), packaged models from
the `conda-misc` channel, and no PyTorch. Exporting DA3-Small to ONNX (the
wrapper's `scripts/build_tensorrt_engine.py` already contains an ONNX export
step) and adding an ONNX Runtime backend – either upstream in the wrapper or
in our own node – would:

- avoid the large PyTorch/DA3 dependency tree on robots,
- reuse the existing execution-provider fallback chain and robot drivers,
- work on both NUC (WebGPU/Vulkan) and Jetson (TensorRT).

The GerdsenAI wrapper can still be used as-is on developer machines for
evaluation.

---

## 10. Documentation

- Add a section to the vision docs
  ([src/bitbots_vision/docs/manual/](src/bitbots_vision/docs/manual)) or the
  general manual describing the depth node, its launch arg, the Pixi
  environment, and the model package.
- Update [src/bitbots_vision/README.md](src/bitbots_vision/README.md) /
  bringup docs if a new launch argument is added.
- Note the model licence in the docs.

---

## 11. Validation checklist

1. `pixi install -e <new env>` solves on `linux-64` (and `linux-aarch64` if
   targeted).
2. `pixi run -e default build` and `pixi run -e default test` still pass
   (package skipped in tests as described in §4.4).
3. `pixi run -e <new env> build --packages-select depth_anything_3_ros2`.
4. `pixi run -e <new env> python -c "import depth_anything_3_ros2.depth_anything_3_node"`
   succeeds; model loads from the local model package with
   `HF_HUB_OFFLINE=1`.
5. Offline test: play a rosbag containing
   `zed/zed_node/rgb/image_rect_color` + `camera_info` and run the depth
   launch file; check `ros2 topic hz` on the depth topic and
   `log_inference_time: true` output.
6. Simulation: `ros2 launch bitbots_bringup mujoco_simulation.launch.py` with
   `depth:=true`; verify `use_sim_time` and stamps/frames on output.
7. Robot (only with explicit approval, no motion needed): run
   `vision_standalone.launch depth:=true`, measure CPU/GPU load with
   `system_monitor` while the regular vision runs, and confirm the RF-DETR
   vision frame rate does not degrade unacceptably.
8. `pixi run -e default format` and review the diff.

---

## 12. Summary of concrete changes

| File / location | Change |
|-----------------|--------|
| `src/lib/depth_anything_3_ros2/` (new) | Imported wrapper (+ `.gitrepo`), patched per §3.2 (preferably via fork/upstream PR) |
| [pixi.toml](pixi.toml) / `pixi.lock` | New `depth_anything` feature + environment, `HF_HUB_OFFLINE` activation env, test skip entry (after team approval) |
| `conda-misc` channel | New model weight package (DA3-Small) |
| `src/bitbots_vision/config/depth_anything_3.yaml` (new) | Canonical parameters |
| `src/bitbots_vision/launch/depth.launch` (new) | Node with `use_sim_time`, remaps to ZED topics |
| [bitbots_bringup launch files](src/bitbots_misc/bitbots_bringup/launch) | `depth` arg propagated through vision/highlevel/teamplayer/standalone/MuJoCo launches |
| `package.xml` of the launching package | `exec_depend` on `depth_anything_3_ros2` |
| [rosbag_record.launch.py](src/bitbots_misc/bitbots_bringup/launch/rosbag_record.launch.py), monitoring configs | Optional topics/visualisation |
| Vision / manual docs | Usage, environment, licence |
| Ansible / robot setup (only for Jetson TRT path) | Host TensorRT service and engine build |
