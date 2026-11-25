"""Render RGB + ground-truth depth from the Pi plus head camera in the Bit-Bots MuJoCo world.

Usage: python scripts/sim_render.py
Writes data/sim/<view>_rgb.png, <view>_depth.npy and intrinsics.json.
"""

import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
XML_DIR = ROOT.parent / "src/bitbots_simulation/bitbots_mujoco_sim/xml"
OUT = ROOT / "data/sim"

# ZED Mini config, identical to bitbots_mujoco_sim/camera.py
W, H = 672, 376
FX = FY = 362.6533508300781
CX, CY = 329.59454345703125, 182.3579864501953

# (name, head pitch in rad, ball xy, second robot xy); observer stands at (-3, 0) facing +x
VIEWS = [
    ("near", 0.6, (-2.0, 0.1), (-1.0, 0.6)),
    ("mid", 0.35, (-1.0, -0.3), (0.5, 0.8)),
    ("far", 0.15, (1.0, 0.4), (2.5, -0.7)),
]


def build_model() -> mujoco.MjModel:
    template = (XML_DIR / "kid_field.xml").read_text()
    world = template.replace("{{NUM_ROBOTS}}", "2").replace("{{OFFSET}}", "6.0").replace("{{ROBOT_TYPE}}", "piplus")
    path = XML_DIR / "generated_world_mde.xml"
    path.write_text(world)
    try:
        return mujoco.MjModel.from_xml_path(str(path))
    finally:
        path.unlink()  # keep the package directory clean


def home_qpos() -> dict[str, float]:
    robot = mujoco.MjModel.from_xml_path(str(XML_DIR / "pi_plus.xml"))
    key = mujoco.mj_name2id(robot, mujoco.mjtObj.mjOBJ_KEY, "home")
    values = {}
    for i in range(robot.njnt):
        jnt = robot.joint(i)
        if jnt.type == mujoco.mjtJoint.mjJNT_FREE:
            values["__base_z"] = float(robot.key_qpos[key, jnt.qposadr[0] + 2])
        else:
            values[jnt.name] = float(robot.key_qpos[key, jnt.qposadr[0]])
    return values


def set_robot(model, data, idx, xy, home, head_pitch=None):
    for i in range(model.njnt):
        jnt = model.joint(i)
        if not jnt.name.endswith(f"_{idx}"):
            continue
        bare = jnt.name.removeprefix("robot_").removesuffix(f"_{idx}")
        adr = jnt.qposadr[0]
        if jnt.type == mujoco.mjtJoint.mjJNT_FREE:
            data.qpos[adr : adr + 7] = [xy[0], xy[1], home["__base_z"], 1, 0, 0, 0]
        elif bare in home:
            data.qpos[adr] = home[bare]
        if head_pitch is not None and "head" in bare and "pitch" in bare:
            data.qpos[adr] = head_pitch


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = build_model()
    data = mujoco.MjData(model)
    home = home_qpos()
    ball_adr = model.joint("ball-root").qposadr[0]
    renderer = mujoco.Renderer(model, height=H, width=W)
    cam = "robot_head_camera_0"

    for name, pitch, ball_xy, other_xy in VIEWS:
        set_robot(model, data, 0, (-3.0, 0.0), home, head_pitch=pitch)
        set_robot(model, data, 1, other_xy, home)
        data.qpos[ball_adr : ball_adr + 7] = [ball_xy[0], ball_xy[1], 0.11, 1, 0, 0, 0]
        mujoco.mj_forward(model, data)

        renderer.disable_depth_rendering()
        renderer.update_scene(data, camera=cam)
        rgb = renderer.render()
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera=cam)
        depth = renderer.render()  # metric z-depth in meters

        Image.fromarray(rgb).save(OUT / f"{name}_rgb.png")
        np.save(OUT / f"{name}_depth.npy", depth.astype(np.float32))
        print(f"{name}: depth {depth.min():.2f}–{depth.max():.2f} m")

    (OUT / "intrinsics.json").write_text(json.dumps({"fx": FX, "fy": FY, "cx": CX, "cy": CY, "w": W, "h": H}))


if __name__ == "__main__":
    main()
