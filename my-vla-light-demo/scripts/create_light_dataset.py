from pathlib import Path
import shutil

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


ROOT = Path.home() / "projects/smolVLA/datasets/light_switch_demo"

if ROOT.exists():
    shutil.rmtree(ROOT)

# Model LIBERO hiện tại của bạn:
# 2 images + state[8] -> action[7]
features = {
    "observation.images.image": {
        "dtype": "image",
        "shape": (3, 256, 256),
        "names": ["channels", "height", "width"],
    },
    "observation.images.image2": {
        "dtype": "image",
        "shape": (3, 256, 256),
        "names": ["channels", "height", "width"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (8,),
        "names": [f"state_{i}" for i in range(8)],
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": [
            "light",
            "unused_1",
            "unused_2",
            "unused_3",
            "unused_4",
            "unused_5",
            "unused_6",
        ],
    },
}

dataset = LeRobotDataset.create(
    repo_id="local/light_switch_demo",
    fps=10,
    root=ROOT,
    robot_type="light_demo",
    features=features,
    use_videos=False,
)

rng = np.random.default_rng(42)

on_commands = [
    "bật đèn",
    "mở đèn",
    "hãy bật đèn",
    "bật đèn lên",
    "turn on the light",
]

off_commands = [
    "tắt đèn",
    "hãy tắt đèn",
    "tắt đèn đi",
    "đóng đèn",
    "turn off the light",
]

# 20 episode, mỗi episode 60 frame.
# 10 ON + 10 OFF.
for episode in range(20):

    is_on = episode % 2 == 0

    if is_on:
        command = on_commands[(episode // 2) % len(on_commands)]
        target = 1.0
    else:
        command = off_commands[(episode // 2) % len(off_commands)]
        target = -1.0

    for frame_idx in range(60):

        # Camera dummy giống nhau cho ON/OFF.
        # Như vậy model phải học chủ yếu từ text.
        image1 = np.full((256, 256, 3), 110, dtype=np.uint8)
        image2 = np.full((256, 256, 3), 130, dtype=np.uint8)

        # Một chút pattern để image không hoàn toàn trống.
        image1[80:180, 105:150] = [170, 170, 170]
        image2[90:170, 110:145] = [160, 160, 160]

        state = rng.normal(
            loc=0.0,
            scale=0.03,
            size=(8,),
        ).astype(np.float32)

        # Noise nhỏ để các dimension còn lại không có std = 0.
        action = rng.normal(
            loc=0.0,
            scale=0.01,
            size=(7,),
        ).astype(np.float32)

        action[0] = target + rng.normal(0, 0.015)

        dataset.add_frame(
            {
                "observation.images.image": image1,
                "observation.images.image2": image2,
                "observation.state": state,
                "action": action,
                "task": command,
            }
        )

    dataset.save_episode()
    print(f"Saved episode {episode:02d}: {command} -> {target:+.0f}")

dataset.finalize()

print("\nDONE")
print("Dataset:", ROOT)
print("Episodes:", dataset.meta.total_episodes)
print("Frames:", dataset.meta.total_frames)