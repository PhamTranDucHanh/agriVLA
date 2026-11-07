# Bước 6 — Model, fresh LoRA và inference/evaluator

Đây là phần chuẩn bị CPU; không có optimizer step, backward hoặc chứng nhận VRAM 4 GB. GPU smoke/save/reload/resume thuộc bước 7. Hướng dẫn chuyển toàn bộ sang máy mới: [GPU_HANDOFF.md](GPU_HANDOFF.md).

## Dataset hiện hành sau yêu cầu bổ sung văn nói

V0 ở data/manifests/v0 và data/lerobot/v0 giữ nguyên lock c59005a5d4eea36731051f5b82a9341d9be5ae50cca4aafcad1055743ef16563. Bản dùng cho model là **wording_v1**, 3.600 train / 720 val / 720 test:

- 3.000/600/600 records kế thừa cùng ảnh, nguồn, source split, target, translation pairs và family partitions. 1.088/402/402 instructions đổi sang văn nói, có cặp dịch đúng nghĩa. Không thêm ảnh hoặc plot mới.
- WATER_ALL thêm 600/120/120 records, chia đều level 1/2/3. Lệnh chỉ định mức là **operator override**, không bị ngưỡng héo chặn. Ví dụ “Tưới cả 3 vườn mức 2 nhé” → [2,2,2], kể cả wilt [0,0,0].
- “Tắt hết.” → STOP_ALL → [0,0,0]. “Xem cả 3 vườn, vườn nào đạt ngưỡng héo thì tưới nhé.” → AUTO, vẫn giữ ngưỡng 2. “Tưới cả 3 vườn” thiếu mức không được tự suy thành level hoặc AUTO.
- Sáu intent cũ giữ policy. WORST_ONLY vẫn nói rõ mọi vùng hòa max; CAP_LEVEL vẫn là trần, không phải ép tưới đúng mức; EXCLUDE_ZONE không mở vùng bị loại. WATER_ALL có 6/2/2 train/unseen-val/unseen-test families mới. Không đưa structured intent/teacher vào features.
- Provenance gốc chỉ upstream v0; revision_metadata và dataset_lock v1 pin phần sửa wording/override. Mọi triplet đã được QA/review ở v0; QA mới chứng minh không đổi ảnh/target cũ, không nới chống leakage.

Đổi wording/policy được người dùng yêu cầu trong lúc làm bước 6; vì vậy cần convert bản mới và tính **train stats mới**. Không dùng stats v0 với dataset wording_v1. WATER_ALL/STOP có thể giải bằng text, nên phải xem metrics riêng theo intent khi đánh giá Vision.

```bash
# Từ root repo, kiểm tra bản mới (không sửa dữ liệu):
agri-soybean-vla/.venv/bin/python -m agri_vla.cli validate-revision
agri-soybean-vla/.venv/bin/python -m agri_vla.cli browse-revision --port 8766
# Mở http://127.0.0.1:8766/ và lọc WATER_ALL hoặc tìm “tắt hết”.
```

Các lệnh tái lập chỉ dùng khi output chưa tồn tại:

```bash
AGRI_DATA_PY=agri-soybean-vla/.venv/bin/python
AGRI_TRAIN_PY=agri-soybean-vla/.venv-train/bin/python
$AGRI_DATA_PY -m agri_vla.cli compose-revision --dry-run
$AGRI_DATA_PY -m agri_vla.cli compose-revision
$AGRI_TRAIN_PY -m agri_vla.cli convert --config agri-soybean-vla/configs/lerobot_wording_v1.yaml --profile smoke
$AGRI_TRAIN_PY -m agri_vla.cli convert --config agri-soybean-vla/configs/lerobot_wording_v1.yaml --profile full
```

## Model và config

Config: configs/training_v0.yaml. Đây là model profile v0 dùng dataset **revision wording_v1**, không phải dùng lại dataset v0.

| Thành phần | Giá trị |
|---|---|
| LeRobot | 7e241bd630a3719a56157a497ce5d08f244784f1 |
| lerobot/smolvla_base | 5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6 |
| SmolVLM2-500M-Video-Instruct | 7b375e1b73b11138ff12fe22c8f2822d8fe03467 |
| Input/target | camera1/2/3 RGB, state[3] zeros, action[3] |
| Padding nội bộ | max_state_dim=max_action_dim=32, giữ tương thích pretrained weights |
| Horizon/model resize | chunk_size=n_action_steps=1; 256×256 |
| LoRA | Fresh r=8, alpha=8; modules_to_save=[] |
| Targets | Expert q/v projections và state/action/time projections theo preset upstream |
| Freeze | Toàn bộ VLM/vision; chỉ LoRA trainable |
| Batch/workers/compile | 1 / 0 / false |
| AMP | true khi launch CUDA; dùng cơ chế use_amp thực có trong upstream |
| LR/schedule | 3e-4; warmup 100; cosine decay tới 3e-5 ở step 6000 |
| Token max length | 96; 64 làm mất nội dung tiếng Việt |
| Train/eval | Chỉ full/train, eval_split=0; val/test evaluator riêng |

Không có gradient_accumulation flag tự chế. 6.000 steps batch 1 trên 3.600 records ≈1,67 lượt dataset; đây chỉ là profile đề xuất, cần chốt bằng smoke/pilot/val và thời gian máy mới.

Khác demo đèn: không nạp adapter light_lora_1000, không khôi phục schema đèn image/image2 + state[8]/action[7], không dùng stats/checkpoint đèn hoặc hardcoded path/rmtree. Schema nông nghiệp được gắn **trước khi tạo model**, base load strict, còn padding 32 chiều giữ nguyên. Mỗi camera đầu vào factory là [B,1,3,256,256], state/action [B,1,3]; SmolVLA lấy observation cuối của history dài 1.

## Chuẩn bị và artifact

```bash
AGRI_TRAIN_PY=agri-soybean-vla/.venv-train/bin/python
$AGRI_TRAIN_PY -m agri_vla.cli prepare-training --dry-run
# Cache đã được chuyển/có sẵn:
$AGRI_TRAIN_PY -m agri_vla.cli prepare-training
# Chỉ thêm --download-models khi cache chưa có; tải đúng hai revision trên.
```

Output mới: models/prepared/wording_v1. Có base_policy/config + symlink tương đối tới immutable base weights, processor JSON/safetensors dùng stats mới, tokenizer đã lưu, schema.json, train_stats.json, provenance.json, token_audit.json, trainable_parameters.json, resolved_train_config.json, initial_adapter và bundle_lock.json. Không ghi đè output cũ.

CPU check thực đã PASS: hai câu VI/EN, no-grad loss finite, actual encoder256 và prediction finite[1,3]. Saved checkpoint fixture được fresh-process reload thành công. Raw untrained example lệnh mức2 cho quantized[1,2,2], khác target policy[2,2,2]; chỉ là kiểm pipeline, chưa chứng minh model thực hiện đúng lệnh.

initial_adapter là **adapter chưa train** để kiểm save/load, không có optimizer/scheduler/RNG state và không dùng resume. Launcher thật tạo fresh LoRA từ base_policy với seed 42. Model loader kiểm checksum/revisions/schema/train stats, rebase local backbone/tokenizer paths, đọc saved processor rồi mới inference.

STATE=IDENTITY vì giả định mọi van ban đầu đóng; không chia state cho std=0. ACTION=MEAN_STD và inverse từ **full train của wording_v1**. Không dùng stats held-out hoặc quantiles upstream gộp singleton episodes. Teacher/intent/case/provenance/generation_metadata/revision_metadata chỉ nằm trong sidecar, không là model features.

## Kiểm CPU, inference và evaluator

```bash
$AGRI_TRAIN_PY -m agri_vla.cli check-model-inputs --output agri-soybean-vla/reports/model-inputs-new-check.json
$AGRI_TRAIN_PY -m agri_vla.cli infer --split val --limit 4 --output agri-soybean-vla/outputs/infer/new-diagnostic
$AGRI_TRAIN_PY -m agri_vla.cli evaluate --split test --checkpoint /path/to/pretrained_model --device cuda --output agri-soybean-vla/outputs/eval/new-test
# Chỉ in resolved trainer config; không optimizer:
$AGRI_TRAIN_PY -m agri_vla.cli train-upstream --steps 100 --output agri-soybean-vla/outputs/train/new-smoke --dry-run
```

Mọi path output trong ví dụ phải chưa tồn tại. Không truyền checkpoint nghĩa là dùng initial_adapter **chưa train**; predictions chỉ xác nhận pipeline, không phải chất lượng model. --limit tạo báo cáo partial, không được gọi là kết quả toàn test.

Inference reset policy queue và processor cho mỗi record độc lập, allowlist chỉ camera/state/task, inverse normalization rồi bắt buộc finite [1,3]. Rule lượng tử: **clip [0,3] rồi floor(x+0.5)**; đúng nửa làm tròn lên. JSON giữ cả continuous/clipped/quantized action và cờ out_of_range; không che lỗi NaN/Inf bằng clipping. Không override prediction theo teacher/intent.

Evaluator chỉ nhận held-out val/test roots, ghi predictions.jsonl và metrics.json: zone accuracy, exact-vector accuracy, continuous MAE, out-of-range, false-watering ở target OFF, STOP accuracy; phân nhóm language/intent/wording. Fixture metrics có expected độc lập, không suy accuracy từ loss. Đây chưa phải benchmark, baseline, ablation hoặc chọn checkpoint của bước 9.

Wrapper gọi trainer upstream nguyên gốc, không viết lại loop/optimizer. Một hook save nhỏ thêm provenance/schema/train stats/tokenizer và checksum vào từng checkpoint **sau** khi upstream lưu adapter/config/processors và training_state. Loader kiểm cả phần này. Save/reload/resume thực sau optimizer vẫn cần bước 7.

## Infer trực tiếp ba ảnh + câu lệnh

Ngoài --split/--limit, infer có --images (đúng thứ tự zone1/2/3) và --instruction. Ví dụ sau phải thay ba path ảnh bằng files RGB thật, output là JSON file mới:

```bash
$AGRI_TRAIN_PY -m agri_vla.cli infer --images /path/zone1.jpg /path/zone2.jpg /path/zone3.jpg --instruction 'Tưới cả 3 vườn mức 2 nhé.' --output agri-soybean-vla/reports/my-scene-prediction.json
```

State vẫn là giả định zeros; transform giống converter. Đầu vào vượt96tokens bị từ chối. Các câu mẫu WATER_ALL thiếu mức bị từ chối; không có LLM gán target. Model chưa train không được kỳ vọng dự đoán đúng proxy/override. Checkpoint agri_provenance ghi **nguồn khởi tạo** (optimizer_steps=0 lúc chuẩn bị); actual step/optimizer/scheduler/RNG cho resume phải đọc training_state upstream, không đọc trường khởi tạo đó làm số step checkpoint.
