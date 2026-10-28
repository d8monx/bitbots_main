"""Extract camera frames, intrinsics and camera poses from a rosbag (no ROS needed).

Usage: python scripts/extract_rosbag.py data/rosbags/<bag> [--step 1.0]
       (<bag> may also be a single .mcap file inside data/rosbags/<bag>/)
Writes data/frames/<bag>/frame_XXXX.png, intrinsics.json and frames.json.
frames.json holds per frame the 4x4 pose of the camera optical frame in base_footprint
(the point on the ground below the robot), which we need for ground-plane depth.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_typestore
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "/zed/zed_node/rgb/image_rect_color"
INFO = "/zed/zed_node/rgb/camera_info"
GROUND = "base_footprint"


def to_matrix(tf) -> np.ndarray:
    """TF message transform (pose of child in parent) -> 4x4 matrix."""
    T = np.eye(4)
    q = tf.transform.rotation
    T[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    t = tf.transform.translation
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def pose_in_root(frame: str, parents: dict) -> tuple[str, np.ndarray]:
    """Walk up the TF tree; return the root frame and the pose of `frame` in it."""
    T = np.eye(4)
    while frame in parents:
        parent, T_parent_child = parents[frame]
        T = T_parent_child @ T
        frame = parent
    return frame, T


def lookup(target: str, source: str, parents: dict) -> np.ndarray:
    """Pose of `source` in `target` from the latest known transforms."""
    root_a, T_root_target = pose_in_root(target, parents)
    root_b, T_root_source = pose_in_root(source, parents)
    if root_a != root_b:
        raise KeyError(f"{target} and {source} are not connected")
    return np.linalg.inv(T_root_target) @ T_root_source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag", type=Path)
    parser.add_argument("--step", type=float, default=1.0, help="seconds between extracted frames")
    args = parser.parse_args()
    # A bag folder, or a single .mcap file (e.g. copied without metadata.yaml): name the output after the folder
    out = ROOT / "data/frames" / (args.bag.parent.name if args.bag.is_file() else args.bag.name)
    out.mkdir(parents=True, exist_ok=True)

    parents = {}  # child frame -> (parent frame, latest 4x4 pose of child in parent)
    frames, intrinsics, next_t = [], None, None
    with AnyReader([args.bag], default_typestore=get_typestore(Stores.ROS2_JAZZY)) as reader:
        conns = [c for c in reader.connections if c.topic in (IMAGE, INFO, "/tf", "/tf_static")]
        for conn, t, raw in reader.messages(connections=conns):  # in time order
            if conn.topic in ("/tf", "/tf_static"):
                for tf in reader.deserialize(raw, conn.msgtype).transforms:
                    parents[tf.child_frame_id] = (tf.header.frame_id, to_matrix(tf))
            elif conn.topic == INFO and intrinsics is None:
                info = reader.deserialize(raw, conn.msgtype)
                k = info.k
                intrinsics = {"fx": k[0], "fy": k[4], "cx": k[2], "cy": k[5], "w": info.width, "h": info.height}
            elif conn.topic == IMAGE and (next_t is None or t >= next_t):
                msg = reader.deserialize(raw, conn.msgtype)
                try:
                    T = lookup(GROUND, msg.header.frame_id, parents)
                except KeyError:
                    continue  # TF not complete yet at the very start
                next_t = t + int(args.step * 1e9)
                assert msg.encoding == "bgra8", msg.encoding
                bgra = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 4)
                name = f"frame_{len(frames):04d}.png"
                Image.fromarray(bgra[..., 2::-1]).save(out / name)  # BGRA -> RGB
                stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                frames.append({"file": name, "stamp": round(stamp, 3), "T_ground_camera": T.round(6).tolist()})

    (out / "intrinsics.json").write_text(json.dumps(intrinsics))
    (out / "frames.json").write_text(json.dumps(frames, indent=1))
    print(f"{len(frames)} frames -> {out}")


if __name__ == "__main__":
    main()
