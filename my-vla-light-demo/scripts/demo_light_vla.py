from pathlib import Path
from datetime import datetime

import imageio.v2 as imageio
import numpy as np
import torch

from PIL import Image, ImageDraw, ImageFont
from peft import PeftConfig, PeftModel

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.processor_smolvla import (
    make_smolvla_pre_post_processors,
)


ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = ROOT / "datasets/light_switch_demo"

CHECKPOINT = (
    ROOT
    / "models/light_lora_1000"
    / "pretrained_model"
)

OUTPUT_DIR = ROOT / "demos/videos"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------
# FONT
# ----------------------------------------------------------

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

try:
    FONT_BIG = ImageFont.truetype(FONT_PATH, 56)
    FONT_MEDIUM = ImageFont.truetype(FONT_PATH, 36)
    FONT_SMALL = ImageFont.truetype(FONT_PATH, 26)
except Exception:
    FONT_BIG = ImageFont.load_default()
    FONT_MEDIUM = ImageFont.load_default()
    FONT_SMALL = ImageFont.load_default()


# ----------------------------------------------------------
# LOAD DATASET
# ----------------------------------------------------------

print("Loading dataset...")

dataset = LeRobotDataset(
    repo_id="local/light_switch_demo",
    root=DATASET_ROOT,
)

sample = dataset[0]


# ----------------------------------------------------------
# LOAD LoRA POLICY
# ----------------------------------------------------------

print("Loading LoRA adapter:")
print(CHECKPOINT)

peft_config = PeftConfig.from_pretrained(str(CHECKPOINT))

print("Base model:")
print(peft_config.base_model_name_or_path)

base_policy = SmolVLAPolicy.from_pretrained(
    peft_config.base_model_name_or_path
)

# ----------------------------------------------------------
# IMPORTANT:
# smolvla_base mặc định nhớ feature schema của robot SO100:
#   camera1, camera2, camera3 + state[6]
#
# Nhưng LoRA của chúng ta được train bằng:
#   image, image2 + state[8] -> action[7]
#
# Vì vậy phải restore đúng schema dataset ở inference.
# ----------------------------------------------------------

base_policy.config.input_features = {
    "observation.images.image": PolicyFeature(
        type=FeatureType.VISUAL,
        shape=(3, 256, 256),
    ),
    "observation.images.image2": PolicyFeature(
        type=FeatureType.VISUAL,
        shape=(3, 256, 256),
    ),
    "observation.state": PolicyFeature(
        type=FeatureType.STATE,
        shape=(8,),
    ),
}

base_policy.config.output_features = {
    "action": PolicyFeature(
        type=FeatureType.ACTION,
        shape=(7,),
    ),
}

base_policy.config.device = "cuda"

policy_config = base_policy.config

print("\nRestored demo feature schema:")
print("Input features:")
for name, feature in policy_config.input_features.items():
    print(" ", name, feature)

print("Output features:")
for name, feature in policy_config.output_features.items():
    print(" ", name, feature)

policy = PeftModel.from_pretrained(
    base_policy,
    str(CHECKPOINT),
)

policy = policy.to("cuda")
policy.eval()

policy = policy.to("cuda")
policy.eval()


# ----------------------------------------------------------
# PRE / POST PROCESSORS
# ----------------------------------------------------------

preprocessor, postprocessor = make_smolvla_pre_post_processors(
    policy_config,
    dataset_stats=dataset.meta.stats,
)


# ----------------------------------------------------------
# VLA INFERENCE
# ----------------------------------------------------------

def infer(command: str):

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Reset action queue của SmolVLA
    try:
        policy.reset()
    except Exception:
        try:
            policy.get_base_model().reset()
        except Exception:
            pass

    frame = {
        "observation.images.image":
            sample["observation.images.image"].clone(),

        "observation.images.image2":
            sample["observation.images.image2"].clone(),

        "observation.state":
            torch.zeros(8, dtype=torch.float32),

        "task":
            command,
    }

    processed = preprocessor(frame)

    with torch.inference_mode():
        action = policy.select_action(processed)

    action = postprocessor(action)

    if isinstance(action, torch.Tensor):
        action = action.squeeze(0).detach().cpu().numpy()

    action = np.asarray(action).reshape(-1)

    light_value = float(action[0])

    if light_value >= 0:
        semantic_action = "LIGHT_ON"
        light_on = True
    else:
        semantic_action = "LIGHT_OFF"
        light_on = False

    return action, light_value, semantic_action, light_on


# ----------------------------------------------------------
# DRAW LIGHT SIMULATOR
# ----------------------------------------------------------

def draw_frame(
    command,
    action,
    light_value,
    semantic_action,
    brightness,
    phase,
):

    width = 1280
    height = 720

    # background
    bg = int(25 + brightness * 90)

    img = Image.new(
        "RGB",
        (width, height),
        (bg, bg, bg + 5),
    )

    draw = ImageDraw.Draw(img)

    # Header
    draw.text(
        (60, 40),
        "SmolVLA - Light Control Demo",
        font=FONT_BIG,
        fill=(240, 240, 240),
    )

    draw.text(
        (60, 125),
        f'Natural Language: "{command}"',
        font=FONT_MEDIUM,
        fill=(230, 230, 230),
    )

    draw.text(
        (60, 185),
        f"Action[0]: {light_value:+.4f}",
        font=FONT_MEDIUM,
        fill=(230, 230, 230),
    )

    draw.text(
        (60, 240),
        f"Semantic Action: {semantic_action}",
        font=FONT_MEDIUM,
        fill=(230, 230, 230),
    )

    draw.text(
        (60, 300),
        f"Phase: {phase}",
        font=FONT_SMALL,
        fill=(200, 200, 200),
    )

    # Bulb coordinates
    cx = 920
    cy = 370

    # Light glow
    if brightness > 0:

        glow_radius = int(160 * brightness)

        if glow_radius > 0:
            draw.ellipse(
                (
                    cx - glow_radius,
                    cy - glow_radius,
                    cx + glow_radius,
                    cy + glow_radius,
                ),
                fill=(
                    int(120 + 100 * brightness),
                    int(105 + 110 * brightness),
                    int(40 + 70 * brightness),
                ),
            )

    # bulb
    bulb_color = (
        int(80 + 175 * brightness),
        int(80 + 165 * brightness),
        int(70 + 80 * brightness),
    )

    draw.ellipse(
        (cx - 90, cy - 100, cx + 90, cy + 80),
        fill=bulb_color,
        outline=(220, 220, 220),
        width=5,
    )

    # socket
    draw.rectangle(
        (cx - 50, cy + 70, cx + 50, cy + 140),
        fill=(100, 100, 100),
    )

    # rays
    if brightness > 0.4:

        for dx, dy in [
            (0, -150),
            (0, 150),
            (-150, 0),
            (150, 0),
            (-110, -110),
            (110, -110),
            (-110, 110),
            (110, 110),
        ]:

            start_x = cx + int(dx * 0.75)
            start_y = cy + int(dy * 0.75)

            end_x = cx + dx
            end_y = cy + dy

            draw.line(
                (start_x, start_y, end_x, end_y),
                fill=(255, 225, 90),
                width=7,
            )

    # action vector
    action_text = np.array2string(
        action,
        precision=2,
        suppress_small=True,
    )

    draw.text(
        (60, 580),
        f"Continuous action: {action_text}",
        font=FONT_SMALL,
        fill=(190, 220, 255),
    )

    status = "ON" if brightness > 0.5 else "OFF"

    draw.text(
        (845, 580),
        f"LIGHT: {status}",
        font=FONT_BIG,
        fill=(255, 240, 150) if status == "ON"
        else (170, 170, 170),
    )

    return np.asarray(img)


# ----------------------------------------------------------
# CREATE MP4
# ----------------------------------------------------------

def make_video(
    command,
    action,
    light_value,
    semantic_action,
    target_on,
):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_path = (
        OUTPUT_DIR
        / f"light_demo_{timestamp}.mp4"
    )

    fps = 30

    writer = imageio.get_writer(
        output_path,
        fps=fps,
        codec="libx264",
        quality=8,
    )

    # 1. BEFORE - luôn bắt đầu ở trạng thái OFF
    for _ in range(45):
        frame = draw_frame(
            command,
            action,
            light_value,
            semantic_action,
            brightness=0.0,
            phase="Observation / Before Action",
        )
        writer.append_data(frame)

    # 2. Model inference visualization
    for _ in range(30):
        frame = draw_frame(
            command,
            action,
            light_value,
            semantic_action,
            brightness=0.0,
            phase="SmolVLA -> Action",
        )
        writer.append_data(frame)

    # 3. Transition
    for i in range(45):

        ratio = i / 44.0

        if target_on:
            brightness = ratio
        else:
            # Nếu command là OFF thì video cho thấy
            # light briefly ON rồi fade OFF.
            brightness = 1.0 - ratio

        frame = draw_frame(
            command,
            action,
            light_value,
            semantic_action,
            brightness=brightness,
            phase="Executing Semantic Action",
        )

        writer.append_data(frame)

    # 4. AFTER
    final_brightness = 1.0 if target_on else 0.0

    for _ in range(60):
        frame = draw_frame(
            command,
            action,
            light_value,
            semantic_action,
            brightness=final_brightness,
            phase="Final State",
        )

        writer.append_data(frame)

    writer.close()

    return output_path


# ----------------------------------------------------------
# CLI
# ----------------------------------------------------------

print()
print("=" * 60)
print("SmolVLA Light Demo")
print("=" * 60)

while True:

    command = input(
        "\nNhập lệnh [bật đèn / tắt đèn / exit]: "
    ).strip()

    if command.lower() == "exit":
        break

    print()
    print("Running VLA inference...")

    action, light_value, semantic_action, light_on = infer(
        command
    )

    print()
    print("---------------- RESULT ----------------")
    print("TEXT            :", command)
    print("ACTION VECTOR   :", action)
    print("ACTION[0]       :", light_value)
    print("SEMANTIC ACTION :", semantic_action)
    print("----------------------------------------")

    video = make_video(
        command,
        action,
        light_value,
        semantic_action,
        light_on,
    )

    print()
    print("Video saved:")
    print(video)