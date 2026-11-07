# JSONL → LeRobot v3 — bước 5 (v0 lịch sử)

**Hiện hành:** wording_v1 3.600/720/720, xem MODEL_TRAINING.md. CLI convert/verify-lerobot mặc định dùng configs/lerobot_wording_v1.yaml; các lệnh bên dưới ghi rõ config v0 để tái lập bản gốc, không thay thế stats/schema của bản mới.

Converter chạy trên CPU, dùng source LeRobot 0.6.1 commit `7e241bd630a3719a56157a497ce5d08f244784f1`. Không tải SmolVLA base weights, không tạo LoRA và không train ở bước này.

## Môi trường tách riêng

- `.venv`: data/browser/tests data, giữ nguyên và không có Torch/LeRobot.
- `.venv-train`: dependencies train/conversion, không sửa môi trường demo cũ.
- `.vendor/lerobot`: checkout riêng của module, HEAD đúng commit; không sửa source upstream.

```bash
cd /home/hanhpham-duc/agriVLA
bash agri-soybean-vla/scripts/setup.sh --mode train
AGRI_TRAIN_PY=agri-soybean-vla/.venv-train/bin/python
$AGRI_TRAIN_PY -m pip check
$AGRI_TRAIN_PY -m agri_vla.cli preflight --mode train
```

Đã xác minh trên Python 3.12.16 / Linux x86_64: Torch 2.7.1+cu126, torchvision 0.22.1+cu126, TorchCodec 0.5, transformers 5.5.4, PEFT 0.21.2, datasets 4.8.5, pyarrow 26.0.1, numpy 2.2.6, PyAV 15.1.0, accelerate 1.15.0. Full resolved versions: requirements-train.lock.txt; core constraints: requirements-train.constraints.txt. `pip check` và imports Dataset/SmolVLA/PEFT đã PASS.

TorchCodec được pin cùng dòng Torch 2.7 theo [ma trận tương thích chính thức](https://github.com/meta-pytorch/torchcodec#compatibility-with-torch-versions), tránh resolver lấy native ABI cho Torch mới hơn. Export này image-only, backend pyav được chọn tường minh; không dùng video decoding. CUDA hiện không khả dụng trên môi trường kiểm tra, là WARN ở preflight bước 5; chỉ smoke GPU bước 7 mới xác nhận máy 4 GB.

## Input bất biến và config chuyển đổi

Đọc `data/manifests/v0/{train,val,test}.jsonl` đã QA; dataset lock SHA256 phải là:

```text
c59005a5d4eea36731051f5b82a9341d9be5ae50cca4aafcad1055743ef16563
```

Converter gọi lại validator hiện tại: manifest/config/schema/model-contract/implementation hashes và upstream locks, nguồn/ảnh/teacher/policy/pairs/splits được kiểm trước export. Paths ảnh tương đối với **module/base_dir**, không phải thư mục JSONL.

`configs/lerobot_v0.yaml` là config mới riêng: không sửa `configs/dataset_v0.yaml` vì file scene config đã được dataset lock pin. Config mới ghi source lock, output dir, seed, transform, fps và smoke count; không tái sinh scene hoặc chia lại episodes cho eval.

## Mỗi record được lưu thế nào

Một record (một scene + instruction của một ngôn ngữ) → một singleton episode có đúng một frame. Hai bản dịch trở thành hai episode khác nhau nhưng vẫn liên kết trong metadata. `fps=1` và timestamp=0 trong mỗi episode là quy ước serialization; không giả lập chuỗi 50 frame và không ngụ ý ảnh gốc chụp 1 Hz.

| Feature gửi writer | Dtype/shape | Nội dung |
|---|---|---|
| observation.images.camera1/2/3 | image, metadata CHW [3,256,256] | Lần lượt zone1/zone2/zone3 |
| observation.state | float32[3] | [0,0,0], giả định mọi van ban đầu đóng |
| action | float32[3] | target_action proxy 0/1/2/3 |
| task | Một string UTF-8 | instruction Việt hoặc Anh của đúng record |

Ảnh RGB 640×480 được resize LANCZOS giữ tỉ lệ thành 256×192, đệm đen 32 pixel trên/dưới thành 256×256. Writer nhận uint8 HWC, loader trả float32 CHW trong [0,1]. Raw ảnh không đổi; PNG lossless được nhúng vào Parquet khi `use_videos=false`, nên không cần thư mục images độc lập để đọc dataset. Không tạo MP4.

LeRobot tự thêm timestamp/frame_index/episode_index/index/task_index. Teacher/intent/params/case/provenance/generation_metadata **không nằm trong feature schema**; chúng ở `sample_metadata.jsonl`. Mỗi dòng sidecar có episode_index, frame_index=0, sample_id, ba transformed pixel hashes và source_record đầy đủ. `export_info.json` pin mapping/selection/source/config; `export_lock.json` pin file Parquet/metadata và converter code.

## Smoke trước, full sau

```bash
AGRI_DATA_PY=agri-soybean-vla/.venv/bin/python
AGRI_TRAIN_PY=agri-soybean-vla/.venv-train/bin/python

# Dry-run không import ML, không ghi dataset:
$AGRI_DATA_PY -m agri_vla.cli convert --config agri-soybean-vla/configs/lerobot_v0.yaml --profile smoke --dry-run

# Chỉ chạy khi root output CHƯA tồn tại:
$AGRI_TRAIN_PY -m agri_vla.cli convert --config agri-soybean-vla/configs/lerobot_v0.yaml --profile smoke
$AGRI_TRAIN_PY -m agri_vla.cli verify-lerobot --config agri-soybean-vla/configs/lerobot_v0.yaml --root agri-soybean-vla/data/lerobot/v0/smoke
$AGRI_TRAIN_PY -m agri_vla.cli convert --config agri-soybean-vla/configs/lerobot_v0.yaml --profile full
$AGRI_TRAIN_PY -m agri_vla.cli verify-lerobot --config agri-soybean-vla/configs/lerobot_v0.yaml --root agri-soybean-vla/data/lerobot/v0/full
```

64 record smoke = 32 cặp Việt–Anh từ train. Selector seed=42 giữ controlled contrastive panel [0,2,4], phủ sáu intents, mọi zone/cap hợp lệ và action 0/1/2/3 ở từng chiều; không lấy 64 dòng đầu. Full conversion phải kiểm lại smoke report/file hashes/source selection trước khi ghi. Không chỉ tin một chuỗi “PASS”.

Mỗi conversion tự gọi finalize() rồi round-trip mọi record. `verify-lerobot` là lần đọc lại độc lập, mặc định chỉ in kết quả; `--output <new-report.json>` lưu report mới. Root/file đã tồn tại sẽ bị từ chối, không có overwrite/xóa dữ liệu. Nếu muốn tái lập ở chỗ khác, dùng `--output-dir` mới; full có `--smoke-root` trỏ tới smoke mới đã PASS.

## Roots và files để xem

```text
data/lerobot/v0/
  smoke/                    # 64 singleton episodes từ train
  full/
    train/                  # 3000 singleton episodes
    val/                    # 600 singleton episodes
    test/                   # 600 singleton episodes
    train_stats.json        # copy stats từ full/train, không từ val/test
    normalization_contract.json
    roundtrip_report.json
```

Mỗi root LeRobot có `data/chunk-*/file-*.parquet`, `meta/info.json`, `meta/stats.json`, `meta/tasks.parquet`, `meta/episodes/...parquet`, sidecar metadata, export info/lock và roundtrip_report.json. Các roots tách biệt, giữ đúng source split. LeRobot có thể ghi nhãn container nội bộ `splits.train` cho toàn bộ episode trong từng root; **source split thực** phải xem tên root + export_info.source_split + sidecar. Không hợp nhất roots hoặc dùng factory tự chia episodes cho eval. Bước 6 phải train chỉ trên full/train và đặt dataset.eval_split=0.

## Batch thật và QA

Với B=4, horizon=1, đọc bằng LeRobotDataset(delta_timestamps={"action":[0.0]}) và DataLoader:

| Trường | Shape / dtype |
|---|---|
| camera1/2/3 | [4,3,256,256], float32, CPU, [0,1] |
| observation.state | [4,3], float32, toàn 0 |
| action | [4,1,3], float32 |
| action_is_pad | [4,1], bool, toàn false |
| task | list 4 string UTF-8 |

SmolVLAConfig(chunk_size=1,n_action_steps=1) cho action_delta_indices=[0]. EpisodeAwareSampler drop_n_last_frames=0 phải giữ đúng toàn bộ frame, không lặp/thiếu. Đây là kiểm tra dataset/config/sampler, chưa phải policy forward/backward.

**Batch từ factory train upstream:** `resolve_delta_timestamps` còn đặt observation delta [0] cho state và ba cameras. `verify-lerobot` kiểm riêng path này: camera **[4,1,3,256,256]**, state **[4,1,3]**, action vẫn **[4,1,3]**; mọi observation/action_is_pad đều [4,1] và false. Trục thứ hai là history dài một bước, không phải episode nhiều frame. Đây là shape cần giữ đúng khi nối preprocessor/model ở bước 6; không lấy shape raw loader phía trên để suy ra model forward đã tương thích.

Round-trip kiểm **mọi** record: instruction, IDs/episode/frame/time, camera order, decoded pixels so với transform từ source, state/action/mask, feature allowlist và stats finite. Action mean/std còn đối chiếu với target_action gốc. Test RGB đỏ–xanh–lam kiểm độc lập letterbox và thứ tự camera; tests data kiểm selector/negative configs/no-ML dry-run; tests native chạy riêng trong .venv-train.

## Normalization: chỉ lấy full train

`full/train_stats.json` và `full/normalization_contract.json` là nguồn chuẩn cho model/inference/eval sau này. Mọi root LeRobot vẫn có meta/stats.json phục vụ QA, nhưng val/test stats không được dùng để normalize model.

- STATE = IDENTITY vì state hằng có std=0.
- ACTION = MEAN_STD từ **full/train**; postprocessor hoàn nguyên bằng cùng stats.
- Hàm train_statistics từ chối smoke/val/test. Native NormalizerProcessorStep/UnnormalizerProcessorStep kiểm action round-trip trên batch train/val/test với cùng train stats và state identity.
- SmolVLA mặc định không tự biết contract của module này; bước 6 phải override normalization mapping và nạp train stats rõ ràng. Không gọi loader val/test rồi dùng stats của root đó làm inference stats.

**Phát hiện stats cần giữ khi bàn giao:** upstream aggregate quantiles theo episode. Với singleton action, các q01/q10/q50/q90/q99 trong meta/stats.json bằng trung bình các giá trị singleton, không phải global quantiles. Mean/std/min/max vẫn đúng (đã so với toàn bộ targets). Không sửa vendor hoặc meta/stats đã export; [action-stats-step5-reference.json](reports/action-stats-step5-reference.json) ghi true global quantiles và đối chiếu cho QA. Contract này chỉ dùng MEAN_STD, không dùng QUANTILES.

Chưa tải weights/tokenizer, chưa tạo model/LoRA, chưa kiểm token truncation, train GPU 4 GB/save/reload/resume. Các việc đó thuộc bước 6–7. Counts/hashes/tests/report cuối xem AGRIVLA_IMPLEMENTATION_STATUS.md.
