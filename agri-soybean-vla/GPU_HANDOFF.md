# Chuyển sang máy GPU mới — bàn giao sau bước 6

Máy hiện tại không có NVIDIA GPU. Dataset/model preparation và kiểm tra CPU được thực hiện ở đây; chưa có backward, optimizer update, VRAM measurement hoặc chứng nhận fit 4 GB. Bước 7–9 phải tiếp tục trên máy GPU.

## 1. Mang gì sang máy mới

Gói bàn giao được tạo ở `agri-soybean-vla/outputs/handoff/agri-soybean-step6.tar`, cùng hai file `.sha256` và `.manifest.json`. Copy **cả ba file** bằng ổ ngoài/SCP/công cụ chuyển file bạn dùng.

Gói chứa source/config/tests/docs/context/plan/prompts/status, raw CSV/ZIP/ảnh, source reports/splits, cả manifests/LeRobot v0 và wording_v1, QA/review reports và đúng hai revision model cache. Không đưa vào Git. Gói không có `.git`, môi trường Python, vendor, pip cache, volatile .log, hoặc prepared bundle chứa đường dẫn máy cũ. Model weights đã có, nên máy mới không phải tải lại weights nếu checksum đúng; cài dependencies và clone vendor vẫn cần mạng.

Nếu sau bàn giao bạn sửa source/data/docs và cần gói mới, chọn output khác:

```bash
agri-soybean-vla/.venv/bin/python -m agri_vla.cli package-handoff --output agri-soybean-vla/outputs/handoff/new-transfer.tar --dry-run
agri-soybean-vla/.venv/bin/python -m agri_vla.cli package-handoff --output agri-soybean-vla/outputs/handoff/new-transfer.tar
```

## 2. Chuẩn bị hệ thống và giải nén

Dùng Linux x86_64 hoặc WSL2 có GPU NVIDIA được hệ thống nhìn thấy, Git, Bash, mạng cho dependencies và đủ dung lượng cho archive + giải nén + environments + checkpoints (dự trù ít nhất 30 GB trống). `nvidia-smi` phải thấy GPU/driver. Driver/WSL/container GPU được cài theo nền tảng máy; script module không cài driver.

Tại thư mục chứa ba file đã copy:

```bash
sha256sum -c agri-soybean-step6.tar.sha256
# Kỳ vọng: agri-soybean-step6.tar: OK. Nếu sai thì chuyển lại file, không tiếp tục.
AGRI_TRANSFER_DIR="$PWD"
mkdir "$HOME/agrivla-gpu"
tar -xf agri-soybean-step6.tar -C "$HOME/agrivla-gpu"
cd "$HOME/agrivla-gpu/agriVLA"
git init -b feat/compose-dataset
git branch --show-current
nvidia-smi
```

`mkdir` dùng thư mục mới; nếu đã tồn tại, chọn tên khác thay vì giải nén đè. Git init chỉ dựng branch cho preflight, không commit/push. Module không phụ thuộc tên user hoặc đường dẫn máy cũ.

## 3. Dựng lại env, không copy venv

Nếu có Python 3.12 + venv:

```bash
bash agri-soybean-vla/scripts/setup.sh --mode data
bash agri-soybean-vla/scripts/setup.sh --mode train
```

Nếu chưa có Python 3.12, dùng uv quản lý interpreter theo [hướng dẫn cài chính thức](https://docs.astral.sh/uv/getting-started/installation/). Ví dụ Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh -o /tmp/agrivla-uv-install.sh
sh /tmp/agrivla-uv-install.sh
bash agri-soybean-vla/scripts/setup.sh --mode data --managed-python --uv "$HOME/.local/bin/uv"
bash agri-soybean-vla/scripts/setup.sh --mode train --managed-python --uv "$HOME/.local/bin/uv"
```

LeRobot vendor phải đúng 7e241bd630a3719a56157a497ce5d08f244784f1 và clean. Train env cài torch 2.7.1+cu126 cùng lock đã xác minh. Dùng **scripts/setup.sh trong module**; script cài demo ở root không phải flow này. Không sửa lock/đổi Torch theo cảm tính khi gặp lỗi; đọc pip/preflight log trước.

```bash
AGRI_DATA_PY=agri-soybean-vla/.venv/bin/python
AGRI_TRAIN_PY=agri-soybean-vla/.venv-train/bin/python
$AGRI_TRAIN_PY -m pip check
$AGRI_TRAIN_PY -m agri_vla.cli preflight --mode train --require-cuda
```

Kỳ vọng pip check không có broken requirements, preflight PASS gồm CUDA. Nếu CUDA fail thì chưa chuyển sang train; kiểm tra driver/GPU pass-through/interpreter đang dùng. Không gọi CPU fallback là GPU smoke.

## 4. Xác minh bản copy, dựng model bundle theo đường dẫn mới

Giữ cùng terminal để biến AGRI_TRANSFER_DIR vẫn trỏ tới thư mục chứa archive:

```bash
$AGRI_DATA_PY -m agri_vla.cli verify-handoff --manifest "$AGRI_TRANSFER_DIR/agri-soybean-step6.tar.manifest.json"
$AGRI_DATA_PY -m agri_vla.cli validate-revision
$AGRI_TRAIN_PY -m agri_vla.cli verify-lerobot --config agri-soybean-vla/configs/lerobot_wording_v1.yaml --root agri-soybean-vla/data/lerobot/wording_v1/full
$AGRI_TRAIN_PY -m agri_vla.cli prepare-training --dry-run
$AGRI_TRAIN_PY -m agri_vla.cli prepare-training
$AGRI_TRAIN_PY -m agri_vla.cli check-model-inputs --output agri-soybean-vla/reports/model-inputs-gpu-host-cpu.json
$AGRI_TRAIN_PY -m agri_vla.cli train-upstream --steps 100 --output agri-soybean-vla/outputs/train/smoke-candidate --dry-run
```

`verify-handoff` kiểm các file đã đóng gói; setup chỉ thêm env/vendor/cache và không sửa source/data. `prepare-training` kiểm lại full round-trip CPU, dùng cache model đã khóa rồi tạo models/prepared/wording_v1 mới, đường dẫn local đúng máy mới. Nếu model cache thiếu, thêm --download-models để tải đúng revision; không dùng main/latest hoặc adapter đèn. Không cần compose/convert lại nếu QA/hash đã PASS.

Kỳ vọng: 3.600/720/720 singleton episodes, schema ba camera/state[3]/action[3], tokenizer 96 với zero truncation cả 5.040 records và saved processor, fresh LoRA chỉ 371.328 trainable parameters. Resolved trainer CUDA/AMP, batch 1, horizon 1, resize 256, train root wording_v1/full/train và eval_split=0. GPU-host CPU check vẫn không thay thế GPU smoke.

## 5. Giao model khác làm bước 7 trước run dài

Prompt có thể copy nguyên:

```text
Làm BƯỚC 7 của AgriVLA trên máy GPU mới, repo đang mở, branch feat/compose-dataset. Đọc AGENTS.md, context/plan/status, MODEL_TRAINING.md và GPU_HANDOFF.md. Bước 6 đã chuẩn bị CPU, chưa train hoặc chứng nhận 4 GB. Dataset HIỆN HÀNH là wording_v1, 3600/720/720 records, bảy intent gồm WATER_ALL(level=1/2/3) ép [level]*3; không dùng dataset/stats v0. Giữ schema/pairs/source splits/locks.
Xác minh pip check/preflight CUDA, full verifier và bundle/tokenizer96/train stats/base revisions. Dùng wrapper train-upstream/TrainPipelineConfig và upstream loop hiện có; không sửa vendor, không thêm flag không hỗ trợ. Chạy smoke 50–100 optimizer steps batch1 với prompts dài nhất và train-only subset đủ vi/en/7 intents, stats full train. Đo VRAM thật/ETA, loss/grad/weights finite, adapter thay đổi. Checkpoint có optimizer/scheduler/RNG/step và contract/tokenizer đã đóng gói.
Kiểm save → fresh-process reload → inference cùng seed/noise/tolerance → resume có optimizer step tiếp tục. Resume phải restore training state, không chỉ nạp adapter. Chuẩn bị/kiểm đường resume bằng API upstream thật; launcher hiện tại mới hỗ trợ fresh run, không tự giả đã có resume CLI. Thử overfit 32–64 train samples đối lập sau smoke nếu hợp lý. Ghi báo cáo GPU; chỉ gọi fit 4 GB khi đo trên GPU đó và có headroom. Không run dài/commit/push. Nếu OOM thì điều chỉnh profile có version/bundle mới rồi rerun, không làm sai schema/stats hoặc giảm chất lượng ảnh âm thầm.
```

Wrapper có `train-upstream --execute` để gọi trainer CUDA thực; **không chạy run dài trước khi bước 7 xác minh**. Chưa có CLI smoke-train đo VRAM/resume hoàn chỉnh ở bước 6; các lệnh dự kiến trong plan phải được implement ở bước 7, không được báo là đã có.

## 6. Sau bước 7

- Bước 8: hoàn thiện run_training.sh/gate smoke, start/resume, log/ETA/save/giữ checkpoints. Khởi đầu 6.000 steps là đề xuất, chốt bằng tốc độ đo và val. Không tự xóa output/checkpoints cũ.
- Bước 9: chọn checkpoint bằng val, rồi đánh giá full test đúng một protocol; metrics theo intent/language/seen-unseen, baselines/ablation/counterfactual. WATER_ALL/STOP có thể chỉ cần text; không dùng accuracy chung để chứng minh Vision nhận biết héo.
- Loader eval dùng checkpoint + bundle/base cache đúng revisions và train stats; không normalize theo val/test. Các lệnh evaluate cơ bản đã có trong MODEL_TRAINING.md; full validation selection/baseline/ablation vẫn thuộc bước 9.

Không có lệnh điều khiển van thật trong flow này. Action là nhãn tổng hợp/override lệnh, chưa phải lượng nước hiệu chuẩn hoặc bằng chứng hiệu quả tưới.
