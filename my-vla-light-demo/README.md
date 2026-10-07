# agriVLA — SmolVLA Light Control Demo

This demo is a small proof-of-concept showing how a Vision-Language-Action (VLA) model can map a natural-language command to a continuous action and then execute that action in a simple light simulator.

The current demo supports commands such as:

```text
"bật đèn"
"tắt đèn"
"turn on the light"
"turn off the light"
```

Example outputs:

```text
"bật đèn"
    ↓
SmolVLA
    ↓
ACTION[0] ≈ +1.03
    ↓
LIGHT_ON
```

```text
"tắt đèn"
    ↓
SmolVLA
    ↓
ACTION[0] ≈ -0.60
    ↓
LIGHT_OFF
```

The simulator then renders the corresponding light state and saves the result as an MP4 video.

---

## 1. Demo Goal

The goal of this demo is to validate a minimal end-to-end VLA pipeline:

```text
Natural-language instruction
        +
Visual observation
        +
System state
        ↓
     SmolVLA
        ↓
Continuous action vector
        ↓
Semantic actuator command
        ↓
Light simulator
        ↓
MP4 output
```

This is not yet the final agricultural control system. It is an intermediate proof-of-concept demonstrating that SmolVLA can be fine-tuned so that different language instructions produce different physical-action representations.

The same idea can later be extended to agricultural actions such as:

```text
"Water zone 1"
"Open valve 2"
"Stop irrigation"
"Water the driest zone"
```

---

## 2. Repository Structure

The light demo is located under:

```text
my-vla-light-demo/
├── datasets/
│   └── light_switch_demo/
│       ├── data/
│       │   └── chunk-000/
│       │       └── file-000.parquet
│       ├── images/
│       │   ├── observation.images.image/
│       │   └── observation.images.image2/
│       └── meta/
│           ├── episodes/
│           ├── info.json
│           ├── stats.json
│           └── tasks.parquet
│
├── models/
│   └── light_lora_1000/
│       └── pretrained_model/
│           ├── adapter_config.json
│           ├── adapter_model.safetensors
│           ├── config.json
│           ├── policy_preprocessor.json
│           ├── policy_postprocessor.json
│           ├── policy_preprocessor_step_5_normalizer_processor.safetensors
│           ├── policy_postprocessor_step_0_unnormalizer_processor.safetensors
│           └── train_config.json
│
├── demos/
│   └── videos/
│
└── scripts/
    ├── create_light_dataset.py
    └── demo_light_vla.py
```

The repository root also contains:

```text
scripts/
├── install.sh
└── run_orange_juice.sh
```

`install.sh` creates the Python virtual environment and installs the dependencies used by the project.

---

## 3. Requirements

Recommended environment:

- Ubuntu 24.04 / WSL2
- Python 3.12
- NVIDIA GPU with CUDA support
- NVIDIA driver available to WSL2
- Internet access for the first run, because the SmolVLA base model is downloaded from Hugging Face

Install required Ubuntu system packages:

```bash
sudo apt update

sudo apt install -y \
    python3-venv \
    git \
    build-essential \
    cmake \
    ffmpeg
```

These packages are installed at the Ubuntu system level, not inside the Python virtual environment.

---

## 4. Installation

From the repository root:

```bash
cd ~/Working/agriVLA
```

Run:

```bash
bash scripts/install.sh
```

The installer creates:

```text
agriVLA/.venv/
```

and installs the Python dependencies required by LeRobot, SmolVLA, PEFT/LoRA, PyTorch, and the demo.

After installation, the same `.venv` can be used for both the LIBERO demo and the light-control demo.

Check CUDA:

```bash
.venv/bin/python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')
"
```

Example:

```text
PyTorch: 2.7.1+cu126
CUDA available: True
GPU: NVIDIA T1200 Laptop GPU
```

---

## 5. Run the Light VLA Demo

From the repository root:

```bash
cd ~/Working/agriVLA
```

Run:

```bash
.venv/bin/python my-vla-light-demo/scripts/demo_light_vla.py
```

You should see:

```text
SmolVLA Light Demo

Nhập lệnh [bật đèn / tắt đèn / exit]:
```

Try:

```text
bật đèn
```

Example output:

```text
TEXT            : bật đèn
ACTION VECTOR   : [ 1.032 ... ]
ACTION[0]       : 1.032
SEMANTIC ACTION : LIGHT_ON
```

Then try:

```text
tắt đèn
```

Example:

```text
TEXT            : tắt đèn
ACTION VECTOR   : [-0.596 ... ]
ACTION[0]       : -0.596
SEMANTIC ACTION : LIGHT_OFF
```

English commands can also be tested:

```text
turn on the light
turn off the light
```

Type:

```text
exit
```

to stop the program.

---

## 6. Video Output

Every command generates a visualization video.

Videos are saved to:

```text
my-vla-light-demo/demos/videos/
```

The video shows:

```text
Natural-language command
        ↓
SmolVLA inference
        ↓
Predicted action vector
        ↓
LIGHT_ON / LIGHT_OFF
        ↓
Light state transition
```

For example:

```text
"bật đèn"
    ↓
ACTION[0] = +1.032
    ↓
LIGHT_ON
    ↓
bulb OFF → ON
```

The light itself is currently simulated. No physical relay, LED, pump, or GPIO is controlled by this demo yet.

---

# 7. What Is SmolVLA?

SmolVLA is a Vision-Language-Action model.

Conceptually:

```text
Vision
  +
Language
  +
State
  ↓
SmolVLA
  ↓
Action
```

Instead of returning a normal chatbot answer, a VLA model predicts actions that can be executed by a robot or control system.

For a robot, this may look like:

```text
Camera image
+
"Pick up the orange juice"
+
Robot joint state
        ↓
     SmolVLA
        ↓
Robot action vector
```

For the agricultural project, the long-term target is closer to:

```text
Camera image
+
Soil / weather sensor state
+
"Water zone 2"
        ↓
     SmolVLA
        ↓
Pump / valve action
```

---

## 8. `lerobot/smolvla_base`

The current light demo is fine-tuned from:

```text
lerobot/smolvla_base
```

This is the pretrained SmolVLA base model hosted on Hugging Face.

It is not stored completely inside this repository.

The local repository contains only the LoRA adapter:

```text
my-vla-light-demo/models/light_lora_1000/pretrained_model/
```

When the demo starts for the first time, the base model may be downloaded automatically from Hugging Face and cached under the user's Hugging Face cache directory.

The base model uses a SmolVLM2 vision-language backbone internally and adds an action expert for continuous action generation.

---

# 9. Why LoRA?

The demo does not fully retrain every parameter of SmolVLA.

Instead, it uses LoRA (Low-Rank Adaptation).

Conceptually:

```text
Pretrained SmolVLA weights
        +
Small trainable LoRA adapters
        ↓
Task-specific SmolVLA
```

This greatly reduces the number of trainable parameters compared with full fine-tuning.

The current adapter was trained for approximately:

```text
1000 training steps
```

and is stored in:

```text
my-vla-light-demo/models/light_lora_1000/pretrained_model/
```

Important files include:

```text
adapter_config.json
adapter_model.safetensors
```

`adapter_model.safetensors` contains the learned LoRA parameters.

---

# 10. Dataset

The custom dataset is located at:

```text
my-vla-light-demo/datasets/light_switch_demo/
```

It was generated using:

```text
my-vla-light-demo/scripts/create_light_dataset.py
```

The dataset teaches the model a simple mapping:

```text
ON instruction  → action[0] ≈ +1
OFF instruction → action[0] ≈ -1
```

Example language instructions include:

```text
bật đèn
tắt đèn
turn on the light
turn off the light
```

The dataset follows the LeRobot dataset format and contains:

- image observations
- system state
- natural-language task
- action labels

---

# 11. Visual Inputs

The custom demo uses two image inputs:

```text
observation.images.image
observation.images.image2
```

Each image is represented as:

```text
3 × 256 × 256
```

where:

```text
3   = RGB channels
256 = image height
256 = image width
```

For the current light demo, these are synthetic/dummy observations.

The model is therefore not yet making its ON/OFF decision from meaningful real-world camera information.

They are included because the experiment keeps the normal VLA input structure:

```text
image + state + language → action
```

Later, these dummy images can be replaced with actual agricultural camera frames.

---

# 12. What Is `state[8]`?

The current custom dataset contains:

```text
observation.state
shape = (8,)
```

This means the state is an 8-dimensional vector:

```text
state =
[
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7
]
```

In the current light-control proof-of-concept, these eight values are synthetic values and do not yet represent real sensors.

They exist mainly so that the custom dataset can preserve the standard VLA structure:

```text
Vision + Language + State → Action
```

In the future agricultural system, the same state vector could represent real sensor information such as:

```text
state[0] = soil moisture zone 1
state[1] = soil moisture zone 2
state[2] = temperature
state[3] = air humidity
state[4] = light intensity
state[5] = water tank level
state[6] = current pump state
state[7] = current valve state
```

For example:

```text
state =
[
    0.18,
    0.65,
    34.2,
    0.72,
    0.80,
    0.55,
    0.0,
    0.0
]
```

could represent a dry first zone with the pump currently off.

---

## 13. Why Did `smolvla_base` Originally Use `state[6]`?

The pretrained `smolvla_base` configuration is designed around robot datasets whose original observation state can have a different shape.

For example:

```text
state[6]
```

means a six-dimensional state vector.

For a robot arm, these values may correspond to actuator or joint information such as:

```text
joint positions
wrist state
gripper state
```

Our custom light dataset instead uses:

```text
state[8]
```

Therefore, during inference the demo explicitly restores the feature schema used during training:

```text
2 image inputs
+
state[8]
→
action[7]
```

This is why `demo_light_vla.py` contains a custom input/output feature configuration.

---

# 14. What Is `action[7]`?

The model outputs a seven-dimensional continuous action vector:

```text
action =
[
    a0,
    a1,
    a2,
    a3,
    a4,
    a5,
    a6
]
```

For this proof-of-concept, we define:

```text
action[0] = light control
```

The other six dimensions are currently unused.

The training target is approximately:

```text
Turn ON:
[+1, 0, 0, 0, 0, 0, 0]

Turn OFF:
[-1, 0, 0, 0, 0, 0, 0]
```

The neural network does not normally output exactly `+1` or `-1`.

For example, actual predictions can be:

```text
"bật đèn"
→ action[0] = +1.032

"tắt đèn"
→ action[0] = -0.596
```

These are predicted action values, not neural-network weights.

---

# 15. Action Vector vs Model Weights

These concepts are different.

## Model weights

Model weights are the internal learned parameters of the neural network.

They are updated during training.

With LoRA, only a relatively small set of adapter parameters is trained.

## Action vector

The action vector is the output produced by the model during inference.

Example:

```text
ACTION VECTOR:
[
  1.032,
  0.018,
 -0.003,
  0.005,
 -0.016,
  0.009,
 -0.006
]
```

The first value:

```text
1.032
```

is not a model weight.

It is the predicted value for `action[0]`.

---

# 16. From Continuous Action to Light Command

Inside `demo_light_vla.py`, the model first predicts the action:

```python
action = policy.select_action(processed)
```

The first action value is then extracted:

```python
light_value = float(action[0])
```

The current semantic mapping is:

```python
if light_value >= 0:
    semantic_action = "LIGHT_ON"
    light_on = True
else:
    semantic_action = "LIGHT_OFF"
    light_on = False
```

Therefore:

```text
action[0] = +1.032
        ↓
LIGHT_ON
```

and:

```text
action[0] = -0.596
        ↓
LIGHT_OFF
```

The semantic result is then passed to the simulator, which changes the rendered brightness and generates the MP4.

For a real embedded system, this layer could later be replaced by:

```text
LIGHT_ON
    ↓
MQTT / Serial / HTTP
    ↓
ESP32 / YOLO UNO
    ↓
GPIO HIGH
```

or for irrigation:

```text
PUMP_ON + VALVE_2_OPEN
    ↓
ESP32
    ↓
Relay / valve driver
```

---

# 17. Current Demo Pipeline

The complete current pipeline is:

```text
User enters:
"bật đèn"
        ↓
Tokenizer / preprocessing
        ↓
Dummy camera image 1
Dummy camera image 2
Dummy state[8]
Natural-language instruction
        ↓
LoRA fine-tuned SmolVLA
        ↓
Continuous action[7]
        ↓
action[0] ≈ +1.0
        ↓
Semantic mapping
        ↓
LIGHT_ON
        ↓
Light simulator
        ↓
Rendered frames
        ↓
MP4 video
```

For OFF:

```text
"tắt đèn"
        ↓
SmolVLA
        ↓
action[0] < 0
        ↓
LIGHT_OFF
        ↓
Simulator
```

---

# 18. Relationship to the Orange Juice LIBERO Demo

The original LIBERO demo uses the same general idea but with a robot simulator:

```text
Natural-language instruction
        ↓
SmolVLA
        ↓
Robot action
        ↓
LIBERO / robosuite / MuJoCo
        ↓
Robot moves
        ↓
Rendered video
```

For example:

```text
"pick up the orange juice and place it in the basket"
```

The light demo replaces the robot simulation environment with a much simpler light environment:

```text
Natural-language instruction
        ↓
SmolVLA
        ↓
Light action
        ↓
Light simulator
        ↓
Rendered video
```

This makes it easier to validate task-specific fine-tuning before integrating real agricultural hardware.

---

# 19. Recreate the Dataset

If needed, regenerate the dataset with:

```bash
.venv/bin/python \
    my-vla-light-demo/scripts/create_light_dataset.py
```

Note that recreating the dataset is not required to run the included pretrained LoRA demo.

---

# 20. Current Limitations

This is intentionally a minimal proof-of-concept.

Current limitations include:

1. The images are synthetic rather than real camera observations.
2. `state[8]` contains synthetic values rather than real sensor measurements.
3. Only `action[0]` currently has semantic meaning.
4. The light is simulated rather than controlled through real hardware.
5. The model has been fine-tuned on a small command set.
6. The LoRA checkpoint is intended for demonstration rather than production control.
7. The final actuator mapping is implemented in Python rather than on an embedded controller.

Therefore, the current result should be described as:

> A proof-of-concept showing that a LoRA-fine-tuned SmolVLA policy can map natural-language instructions to continuous task-specific actions and execute those actions in a simulated environment.

It should not yet be described as a complete autonomous smart-irrigation controller.

---

# 21. Next Steps Toward agriVLA

The next development stage is to replace the synthetic inputs and outputs with real agricultural data.

Target architecture:

```text
Field camera
        +
Soil moisture
Temperature
Humidity
Water level
Current actuator states
        +
Natural-language instruction
        ↓
      SmolVLA
        ↓
Continuous agricultural action
        ↓
Semantic control layer
        ↓
ESP32 / YOLO UNO
        ↓
Pump / valves
```

Example:

```text
Camera:
Zone 1 plants

State:
soil_zone_1 = 18%
soil_zone_2 = 62%
temperature = 34°C
pump = OFF

Instruction:
"Water the driest zone"

        ↓

SmolVLA

        ↓

Action

        ↓

PUMP_ON
VALVE_ZONE_1_OPEN
```

That will turn the current light-control proof-of-concept into an actual agricultural VLA control pipeline.

---

# 22. Quick Start

For a new machine:

```bash
git clone <repository-url>
cd agriVLA
```

Switch to the required branch if necessary:

```bash
git switch dev
```

Install Ubuntu dependencies:

```bash
sudo apt update

sudo apt install -y \
    python3-venv \
    git \
    build-essential \
    cmake \
    ffmpeg
```

Install the project environment:

```bash
bash scripts/install.sh
```

Run the light demo:

```bash
.venv/bin/python \
    my-vla-light-demo/scripts/demo_light_vla.py
```

Then test:

```text
bật đèn
tắt đèn
turn on the light
turn off the light
```

Generated videos are written to:

```text
my-vla-light-demo/demos/videos/
```

---

## Summary

The current demo validates the following idea:

```text
Language
+
Vision
+
State
↓
SmolVLA
↓
Continuous Action
↓
Actuator Command
```

Today:

```text
"bật đèn"
→ SmolVLA
→ action[0] > 0
→ LIGHT_ON
→ simulated light
```

Target agricultural system:

```text
"tưới vùng đang khô"
→ SmolVLA
→ pump / valve action
→ embedded controller
→ real irrigation hardware
```
