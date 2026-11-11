# AgriVLA — GPU training, evaluation, interactive demo and next steps

**Cập nhật: 2026-11-11.** Tài liệu này ghi lại phần triển khai sau bước 6 cho dataset `wording_v1`: GPU smoke, sửa AMP, fine-tuning LoRA, checkpoint resume, validation/test, tiny-overfit diagnostic, giao diện Step 9, web demo ba khu vực, đóng gói checkpoint và kế hoạch xử lý model quality.

Tài liệu dành cho người tiếp nhận công việc. Dataset, schema, source split, action labels và checkpoint cũ không được ghi đè. Checkpoint step 6.000 chạy được về kỹ thuật nhưng chưa đạt baseline; không dùng để điều khiển hệ thống tưới thật.

## Bước 7 — GPU smoke và sửa AMP — 2026-11-09

Smoke ban đầu hoàn tất 100 steps và lưu checkpoint nhưng log có `grdn:nan` ở các cửa sổ step 10 và 30. Nguyên nhân là FP16 `GradScaler` mặc định bắt đầu ở 65.536, gây overflow và bỏ qua optimizer update. Wrapper dự án hiện tạo `Accelerator` với FP16, initial scale 1.024 và growth interval 2.000. Sau mỗi update, wrapper dừng ngay nếu loss/gradient không hữu hạn hoặc AMP bỏ qua optimizer step. Checkout vendor LeRobot không bị sửa.

Stable smoke mới hoàn tất 100/100 steps: mọi loss/gradient hữu hạn, adapter thay đổi và finite, checkpoint có optimizer/scheduler/RNG/step. PyTorch báo peak allocated 1,48 GB; đây không phải tổng VRAM từ `nvidia-smi`. Lệnh tái lập với output mới:

~~~bash
AGRI_TRAIN_PY=agri-soybean-vla/.venv-train/bin/python
$AGRI_TRAIN_PY -m agri_vla.cli train-upstream \
  --profile 4gb --steps 100 \
  --output agri-soybean-vla/outputs/train/step7-smoke-stable-new \
  --execute
~~~

### Tiny overfit với LR 1e-4

Pipeline tiny-overfit chọn cố định 56 train records dưới dạng 28 cặp VI/EN hoàn chỉnh, cân bằng 8 records cho mỗi intent. Pipeline truyền danh sách episode trực tiếp cho `LeRobotDataset`; nó không sao chép hoặc sửa dataset. Run dùng LoRA fresh, LR `1e-4`, 1.000 optimizer steps và lưu checkpoint mỗi 100 steps:

~~~bash
$AGRI_TRAIN_PY -m agri_vla.cli train-upstream \
  --profile 4gb --steps 1000 \
  --tiny-overfit-count 56 --learning-rate 1e-4 \
  --output agri-soybean-vla/outputs/train/my-tiny-overfit-lr1e4 \
  --execute

$AGRI_TRAIN_PY -m agri_vla.cli evaluate-tiny-overfit \
  --checkpoint agri-soybean-vla/outputs/train/my-tiny-overfit-lr1e4/checkpoints/001000/pretrained_model \
  --count 56 --device cuda \
  --output agri-soybean-vla/outputs/eval/my-tiny-overfit-lr1e4-step1000
~~~

Run ngày 2026-11-09 hoàn tất kỹ thuật 1.000/1.000 updates, loss trung bình giảm từ 3,091 ở 100 steps đầu xuống 1,410 ở 100 steps cuối, không có loss/gradient không hữu hạn hoặc AMP skipped step. Tuy vậy, phép kiểm tra khả năng ghi nhớ chính 56 train records đã **FAIL**:

| Metric | Kết quả | Gate |
|---|---:|---:|
| Exact-action accuracy | 0,00% | >= 90% |
| Zone accuracy | 16,67% | >= 95% |
| Continuous MAE | 1,2371 | <= 0,25 |
| STOP exact accuracy | 0,00% | diagnostic |

Model chủ yếu collapse về action `[1,2,1]` (40/56 records), dù subset có 20 target vectors. Artifacts local nằm tại `outputs/train/step7-tiny-overfit-56-lr1e4-20261109` và `outputs/eval/step7-tiny-overfit-56-lr1e4-step1000-20261109`. `metrics.json` có `status: pass` chỉ có nghĩa evaluator chạy xong và artifact hợp lệ; kết luận chất lượng nằm trong `tiny_overfit_result.json` với `passed: false`.

Vì tiny-overfit không đạt gate, sweep checkpoint 500–6.000 trên validation subset **không được chạy**. Lệnh `sweep-checkpoints` đã được bổ sung và luôn tái sử dụng cùng một subset validation song ngữ xác định bởi seed 42, nhưng chỉ nên chạy sau khi một tiny-overfit mới đạt gate:

~~~bash
$AGRI_TRAIN_PY -m agri_vla.cli sweep-checkpoints \
  --checkpoint /path/to/checkpoints/000500/pretrained_model \
  --checkpoint /path/to/checkpoints/001000/pretrained_model \
  --checkpoint /path/to/checkpoints/001500/pretrained_model \
  --checkpoint /path/to/checkpoints/002000/pretrained_model \
  --checkpoint /path/to/checkpoints/002500/pretrained_model \
  --checkpoint /path/to/checkpoints/003000/pretrained_model \
  --checkpoint /path/to/checkpoints/003500/pretrained_model \
  --checkpoint /path/to/checkpoints/004000/pretrained_model \
  --checkpoint /path/to/checkpoints/004500/pretrained_model \
  --checkpoint /path/to/checkpoints/005000/pretrained_model \
  --checkpoint /path/to/checkpoints/005500/pretrained_model \
  --checkpoint /path/to/checkpoints/006000/pretrained_model \
  --count 56 --device cuda \
  --output agri-soybean-vla/outputs/eval/my-fixed-val-sweep
~~~

Tiny failure loại trừ nguyên nhân generalization hoặc split vì evaluator chấm lại đúng 56 records đã dùng để train. Những phần đã được kiểm chứng gồm camera order, resize/letterbox, action normalization, checkpoint reload, finite loss/gradient và accepted optimizer steps. Vấn đề còn lại nằm trong khả năng ghi nhớ của cấu hình tối ưu/model: run chỉ có 17,86 lượt qua tiny subset; scheduler hạ LR `1e-4` xuống `1e-5` trong 1.000 steps; LoRA r=8 chỉ train 371.328 parameters trong action/language expert trong khi VLM bị freeze; subset vẫn chứa 20 target vectors và hai ngôn ngữ. Loss flow-matching giảm không đồng nghĩa action sample sau diffusion đã khớp nhãn rời rạc.

Thứ tự sửa nên giữ test split đóng:

1. Chạy một diagnostic nhỏ hơn trên train, giữ LR cố định và tăng số lượt lặp; log loss cùng prediction của chính train subset tại từng checkpoint.
2. Nếu vẫn không ghi nhớ, kiểm gradient/update theo từng LoRA target và so sánh normalized action label với normalized prediction trước khi sampling.
3. Sau khi chứng minh optimizer/data path đúng, thử cấu hình có sức chứa cao hơn như LoRA rank lớn hơn hoặc mở thêm action-expert projection. Mỗi thay đổi phải nằm trong config/run mới, không sửa action definition, dataset lock hoặc nhãn.
4. Chỉ khi tiny gate đạt mới dùng validation subset để chọn checkpoint/hyperparameter; test chỉ chạy một lần cho cấu hình cuối.

## Bước 8 — Fine-tuning LoRA 6.000 steps — 2026-11-09

Run thật dùng batch 1, warmup 100, cosine decay 6.000→3e-5, save mỗi 500 steps. Khi scaler tăng tới 4.096, guard phát hiện overflow khoảng step 4.397 và dừng trước checkpoint kế tiếp; lineage đó được bỏ sau checkpoint 4.000. Resume launcher được bổ sung để kiểm checksum/contract rồi khôi phục adapter, optimizer, scheduler, RNG, global step và data-order offset. Với adapter-only checkpoint, wrapper nạp pinned base trước rồi gắn PEFT adapter trainable; save hook cũng luôn ghi policy `config.json` trước khi gắn AgriVLA contract.

Lineage checkpoint được chấp nhận là 0→4.000, resume 4.000→4.500, resume 4.500→6.000. Tổng accepted updates là đúng 6.000, tương đương 1,67 lượt qua 3.600 train records. 600 cửa sổ log cho thấy mean loss 100 steps đầu 2,5265 và 100 steps cuối 1,1491; max logged pre-clip gradient norm 10,777, final LR 3e-5, adapter cuối finite. Loss train giảm không thay thế held-out evaluation.

Lệnh bắt đầu fresh run và resume (mọi output phải chưa tồn tại):

~~~bash
$AGRI_TRAIN_PY -m agri_vla.cli train-upstream \
  --profile 4gb --steps 6000 \
  --output agri-soybean-vla/outputs/train/my-fresh-run \
  --execute

$AGRI_TRAIN_PY -m agri_vla.cli train-upstream \
  --profile 4gb --steps 6000 \
  --resume-from agri-soybean-vla/outputs/train/my-fresh-run/checkpoints/004000 \
  --output agri-soybean-vla/outputs/train/my-resumed-run \
  --execute
~~~

Logs của lần chạy thật nằm trong `outputs/logs/step8-lora-6000-t1200-20261109*.log`. Checkpoint cuối đã fresh-process reload trên CUDA và dự đoán finite action `[3]`. Kết quả chất lượng held-out được ghi ở phần bước 9 ngay sau đây.

## Bước 9 — full validation/test và cách chạy thử

Checkpoint 6.000 là checkpoint cuối của lineage Step 8 đã được chấp nhận trước khi mở test. Evaluator dùng seed 42 cho từng sample, reset action queue, dùng processor và ACTION stats của full train, rồi inverse-normalize → kiểm finite → clip `[0,3]` → `floor(x+0.5)`. Cả validation và test đều chạy đủ 720 records, không dùng `--limit`:

| Split / model | Records | Exact-action | Zone accuracy | Continuous MAE | Out-of-range components | STOP exact |
|---|---:|---:|---:|---:|---:|---:|
| Validation, LoRA step 6.000 | 720 | 1,25% | 32,55% | 1,1413 | 441 / 2.160 | 0% |
| Test, LoRA step 6.000 | 720 | 1,81% | 32,50% | 1,2034 | 461 / 2.160 | 0% |
| Validation, always OFF / train-majority | 720 | 29,44% | 53,89% | 0,9463 | 0 | 100% |
| Test, always OFF / train-majority | 720 | 28,89% | 53,89% | 0,9519 | 0 | 100% |

Train-majority vector là `[0,0,0]`, nên hai baseline trong bảng trùng nhau. Checkpoint hiện tại kém baseline ở cả ba metric chính. Loss train giảm và inference chạy được chỉ chứng minh pipeline kỹ thuật hoạt động; chúng không chứng minh model đã học policy Vision + Language → Action. Không dùng checkpoint này để tự động điều khiển tưới thật. Breakdown VI/EN, seen/unseen và từng intent nằm trong `metrics.json`; STOP exact bằng 0 và lượng output ngoài miền cao là hai lỗi đáng chú ý cần phân tích trước lần fine-tune kế tiếp.

Artifacts local (bị Git ignore) gồm:

- `outputs/eval/step9-final-val-full-20261109/{metrics.json,predictions.jsonl}` — 720 sample ID duy nhất, predictions SHA256 `27ab9a7fe22812005f1b5c20e48e7376e2c394b446e615a215d09ee9f6ec4395`.
- `outputs/eval/step9-final-test-full-20261109/{metrics.json,predictions.jsonl}` — 720 sample ID duy nhất, predictions SHA256 `cb9082ef2689bb4f792b1400b0e87a4969b0e5aef2cbf5d7f70eb3d847023694`.
- `outputs/eval/step9-baselines-20261109/metrics.json` — always-OFF và train-majority, chỉ chọn baseline từ train targets.

Lệnh tái lập full evaluation dưới đây yêu cầu CUDA và mỗi `--output` phải là đường dẫn mới:

~~~bash
cd ~/Working/agriVLA
AGRI_TRAIN_PY=agri-soybean-vla/.venv-train/bin/python
CHECKPOINT=agri-soybean-vla/outputs/train/step8-lora-6000-t1200-20261109-resume4500-v3/checkpoints/006000/pretrained_model

$AGRI_TRAIN_PY -m agri_vla.cli evaluate \
  --split val --checkpoint "$CHECKPOINT" --device cuda \
  --output agri-soybean-vla/outputs/eval/my-full-val

$AGRI_TRAIN_PY -m agri_vla.cli evaluate \
  --split test --checkpoint "$CHECKPOINT" --device cuda \
  --output agri-soybean-vla/outputs/eval/my-full-test
~~~

Để xem model hoạt động nhanh trên vài held-out records, dùng `infer` với `--limit`; đây chỉ là diagnostic subset:

~~~bash
$AGRI_TRAIN_PY -m agri_vla.cli infer \
  --split val --limit 4 \
  --checkpoint "$CHECKPOINT" --device cuda \
  --output agri-soybean-vla/outputs/infer/my-val-4

cat agri-soybean-vla/outputs/infer/my-val-4/metrics.json
head -n 4 agri-soybean-vla/outputs/infer/my-val-4/predictions.jsonl
~~~

Để thử ba ảnh RGB của riêng bạn, truyền ảnh theo đúng thứ tự zone 1/2/3 và một instruction Việt hoặc Anh. File output cho biết `continuous_action`, `clipped_action`, `quantized_action` và `out_of_range`:

~~~bash
$AGRI_TRAIN_PY -m agri_vla.cli infer \
  --checkpoint "$CHECKPOINT" --device cuda \
  --images /path/zone1.jpg /path/zone2.jpg /path/zone3.jpg \
  --instruction 'Xem cả 3 vườn, vườn nào đạt ngưỡng héo thì tưới nhé.' \
  --output agri-soybean-vla/outputs/infer/my-three-zone-result.json

cat agri-soybean-vla/outputs/infer/my-three-zone-result.json
~~~

Lệnh ba ảnh đã được kiểm bằng ảnh thật trong dataset và tạo `outputs/infer/step9-demo-three-images-20261109.json`. Model trả continuous `[1,180; 3,449; 1,138]`, quantized `[1,3,1]`, trong khi target mẫu là `[0,1,3]`; đây là failure case, không phải demo thành công về accuracy. State đầu vào vẫn là ba van đóng `[0,0,0]`, và action chỉ được ghi ra JSON, chưa gửi tới phần cứng.

### Giao diện HTML/CSS xem kết quả Step 9

Viewer Step 9 đọc trực tiếp hai `metrics.json`, hai `predictions.jsonl` và baseline local. Source giao diện nằm tại:

- HTML: `src/agri_vla/web/step9.html`
- CSS: `src/agri_vla/web/step9.css`
- JavaScript: `src/agri_vla/web/step9.js`
- Launcher: `scripts/view_step9.sh`

Không mở `index.html` bằng `file://`, vì trình duyệt sẽ chặn `fetch()` artifacts JSON/JSONL. Từ repo root, chạy:

~~~bash
bash agri-soybean-vla/scripts/view_step9.sh
~~~

Sau đó mở **http://127.0.0.1:8780/src/agri_vla/web/step9.html** trong trình duyệt Windows. Có thể đổi port, ví dụ `bash agri-soybean-vla/scripts/view_step9.sh 8781`. Viewer hiển thị so sánh model/baseline, breakdown theo intent và tối đa 100 prediction sau khi lọc split/ngôn ngữ/đúng-sai/out-of-range; artifacts gốc vẫn đủ 720 records mỗi split. Dừng server bằng Ctrl+C.

### Web thử model với ba ảnh và instruction

Web demo tải checkpoint step 6.000 một lần trên GPU, nhận đúng ba ảnh theo thứ tự khu vực 1/2/3 và instruction tiếng Việt hoặc tiếng Anh. Backend áp dụng cùng phép RGB letterbox 256×256, tokenizer, processor, action inverse-normalization và quantization như evaluator. Chạy từ repo root:

~~~bash
agri-soybean-vla/.venv-train/bin/python -m agri_vla.cli demo-web \
  --checkpoint agri-soybean-vla/outputs/train/step8-lora-6000-t1200-20261109-resume4500-v3/checkpoints/006000/pretrained_model \
  --device cuda --port 8000
~~~

Mở **http://127.0.0.1:8000** trong trình duyệt Windows và chờ trạng thái “Model sẵn sàng”. Chọn ba ảnh, có thể chọn mức héo 0–4 để ghi chú, nhập instruction rồi bấm “Chạy SmolVLA”. Giao diện hiển thị continuous action, quantized action và OFF/LOW/MEDIUM/HIGH cho từng van. Mức héo do người dùng chọn chỉ để đối chiếu trên UI và không đi vào model; model thực nhận ba ảnh, instruction và valve state `[0,0,0]`, đúng contract training. Web chỉ lắng nghe `127.0.0.1`, không điều khiển phần cứng và luôn cảnh báo checkpoint hiện tại chưa đạt baseline.

Source web:

- Backend: `src/agri_vla/demo_server.py`
- HTML: `src/agri_vla/web/demo.html`
- CSS: `src/agri_vla/web/demo.css`
- JavaScript: `src/agri_vla/web/demo.js`

### Tiếp tục trên repository đã hoàn thành bước 1–6

Tài liệu này được chuyển lại cho thành viên đã xây dựng bước 1–6 và đã có dataset/model artifacts theo cấu trúc dự án. Không cần gửi lại toàn bộ `data/`, `.cache/models/`, reports hoặc đóng gói lại các bước cũ. Source code được đồng bộ qua Git; phần duy nhất cần gửi ngoài Git là checkpoint LoRA đã train.

Người nhận cần giữ hoặc khôi phục các artifacts cũ tại đúng vị trí:

~~~text
agri-soybean-vla/
├── data/lerobot/wording_v1/full/
├── .cache/models/
├── models/prepared/wording_v1/
└── outputs/
~~~

SmolVLA được fine-tune bằng LoRA nên training không tạo thêm một bản đầy đủ của base model. Kết quả học chính nằm trong:

~~~text
outputs/train/step8-lora-6000-t1200-20261109-resume4500-v3/
└── checkpoints/006000/
    ├── pretrained_model/adapter_model.safetensors  # LoRA weights đã học
    └── training_state/                             # optimizer/scheduler/RNG/step
~~~

Khi inference, loader ghép pinned SmolVLA base trong `.cache/models/` với LoRA adapter trong `pretrained_model/`. Vì người nhận đã có artifacts bước 1–6, không cần gửi lại base weights. Để họ vừa chạy demo vừa có thể resume training, gửi **toàn bộ checkpoint `006000`**, không chỉ riêng `adapter_model.safetensors` hoặc `pretrained_model/`.

Full checkpoint đã được đóng gói thành hai file để gửi qua ổ ngoài, SCP hoặc dịch vụ lưu file:

~~~text
agri-soybean-step6000-full-checkpoint.zip
agri-soybean-step6000-full-checkpoint.zip.sha256
~~~

ZIP chứa đường dẫn tương đối đầy đủ. Người nhận có thể lưu hai file ở bất kỳ thư mục tạm nào; sau khi kiểm checksum, giải nén **từ repo root** để checkpoint đi đúng vào cấu trúc hiện có. Ví dụ nếu hai file được đặt trong `agri-soybean-vla/outputs/`:

~~~bash
cd ~/Working/agriVLA/agri-soybean-vla/outputs
sha256sum -c agri-soybean-step6000-full-checkpoint.zip.sha256

cd ~/Working/agriVLA
CHECKPOINT=agri-soybean-vla/outputs/train/step8-lora-6000-t1200-20261109-resume4500-v3/checkpoints/006000/pretrained_model
test ! -e "$CHECKPOINT"   # Không ghi đè checkpoint đã có.
python3 -m zipfile -e \
  agri-soybean-vla/outputs/agri-soybean-step6000-full-checkpoint.zip \
  .

test -f "$CHECKPOINT/adapter_model.safetensors"
test -f "$CHECKPOINT/agri_checkpoint_lock.json"
test -f "$(dirname "$CHECKPOINT")/training_state/optimizer_state.safetensors"
test -f "$(dirname "$CHECKPOINT")/training_state/training_step.json"
~~~

Sau giải nén, cấu trúc cần đúng như sau:

~~~text
agri-soybean-vla/outputs/train/
└── step8-lora-6000-t1200-20261109-resume4500-v3/
    └── checkpoints/
        └── 006000/
            ├── pretrained_model/
                ├── adapter_model.safetensors
                ├── adapter_config.json
                ├── config.json
                ├── tokenizer/
                ├── policy_preprocessor.json
                ├── policy_postprocessor.json
                ├── agri_train_stats.json
                ├── agri_schema.json
                ├── agri_provenance.json
                └── agri_checkpoint_lock.json
            └── training_state/
                ├── optimizer_state.safetensors
                ├── optimizer_param_groups.json
                ├── scheduler_state.json
                ├── rng_state.safetensors
                └── training_step.json
~~~

Nếu `models/prepared/wording_v1/` vẫn còn và contract/checksum hợp lệ thì không cần chạy lại bước 1–6. Nếu chỉ thiếu prepared bundle nhưng `data/`, reports và `.cache/models/` còn đủ, tạo lại bundle bằng:

~~~bash
cd ~/Working/agriVLA
bash agri-soybean-vla/scripts/setup.sh --mode train
AGRI_TRAIN_PY=agri-soybean-vla/.venv-train/bin/python
$AGRI_TRAIN_PY -m agri_vla.cli preflight --mode train --require-cuda
$AGRI_TRAIN_PY -m agri_vla.cli prepare-training
~~~

Không chạy lại `compose`, không convert lại dataset và không thay split/schema/labels. `prepare-training` chỉ dùng khi bundle local bị thiếu; lệnh này không fine-tune lại model.

Chạy web demo với checkpoint vừa giải nén:

~~~bash
cd ~/Working/agriVLA
AGRI_TRAIN_PY=agri-soybean-vla/.venv-train/bin/python
CHECKPOINT=agri-soybean-vla/outputs/train/step8-lora-6000-t1200-20261109-resume4500-v3/checkpoints/006000/pretrained_model

$AGRI_TRAIN_PY -m agri_vla.cli demo-web \
  --checkpoint "$CHECKPOINT" \
  --device cuda \
  --port 8000
~~~

Mở **http://127.0.0.1:8000** và chờ “Model sẵn sàng”. Nếu port 8000 bận, dùng `--port 8001`. Checkpoint này phục vụ quan sát integration và failure cases; chất lượng hiện tại chưa đạt baseline nên không dùng để điều khiển tưới thật.

Nếu cần tiếp tục đúng optimizer lineage từ step 6.000, `--resume-from` nhận thư mục checkpoint cha, không nhận `pretrained_model/`:

~~~bash
$AGRI_TRAIN_PY -m agri_vla.cli train-upstream \
  --profile 4gb \
  --steps 7000 \
  --resume-from agri-soybean-vla/outputs/train/step8-lora-6000-t1200-20261109-resume4500-v3/checkpoints/006000 \
  --output agri-soybean-vla/outputs/train/my-resume-from-6000 \
  --execute
~~~

Đây chỉ là khả năng khôi phục kỹ thuật. Do checkpoint 6.000 chưa đạt quality gate, các thí nghiệm tiny-overfit, LR hoặc LoRA capacity mới nên bắt đầu ở output mới theo kế hoạch bên dưới thay vì mặc định tiếp tục lineage kém chất lượng.

### Những gì đã triển khai và đã kiểm chứng

- Dataset wording_v1 giữ nguyên 3.600/720/720 records, schema ba ảnh + state `[3]` → action `[3]`, labels và splits không đổi.
- Stable AMP wrapper dùng FP16 initial scale 1.024, growth interval 2.000 và dừng khi loss/gradient không finite hoặc optimizer bỏ step.
- LoRA run đã save/reload/resume tới đúng 6.000 accepted optimizer steps; checkpoint inference cuối chứa adapter, config, tokenizer, processors, train stats và provenance/checksum contract.
- Full validation/test và always-OFF baseline đã chạy. Checkpoint 6.000 chạy được về kỹ thuật nhưng thấp hơn baseline về exact-action, zone accuracy và MAE.
- Tiny-overfit pipeline chọn deterministic bilingual train subset, không sửa dataset, hỗ trợ LR riêng, lưu mỗi 100 steps và có quality gate độc lập.
- Tiny run LR `1e-4`, 1.000 steps hoàn tất không NaN/AMP skip nhưng FAIL memorization gate; output collapse chủ yếu về `[1,2,1]`.
- Fixed-validation checkpoint sweep đã được code và test, nhưng chưa chạy vì tiny gate chưa đạt.
- Web demo loopback tải model một lần trên CUDA, nhận đúng ba ảnh + instruction, letterbox 256×256, trả continuous/clipped/quantized action và ba trạng thái van. Ghi chú mức héo trên UI không bị đưa vào model.
- Inference HTTP thật đã chạy qua checkpoint 6.000 và trả JSON hợp lệ; đây là kiểm chứng integration, không phải chứng minh accuracy.
- Software validation gần nhất: 34 training/web tests pass; full data suite 535 pass và 3 integration tests skip đúng môi trường.

### Các bước tiếp theo để sửa model quality

Không thay nhãn, action definition, dataset lock hoặc test split để làm đẹp accuracy. Làm tuần tự và dừng ở gate thất bại:

1. **Tách lỗi optimization khỏi lỗi capacity.** Thêm tiny preset mới dùng 32 records đối lập, LR cố định `1e-4`, không cosine decay sớm, save/evaluate train subset mỗi 250–500 steps và cho tối đa khoảng 5.000 steps. Giữ seed, ảnh, processor và action stats như hiện tại.
2. **Theo dõi normalized action trước sampling.** Với mỗi checkpoint, lưu target normalized, sampled normalized, continuous và quantized action. Báo histogram prediction/target, số vector dự đoán duy nhất và per-intent train accuracy. Điều này xác định collapse xuất hiện ở loss/flow head hay ở sampling/postprocessing.
3. **Audit gradient và update.** Ghi gradient norm và delta-weight theo từng nhóm `lm_expert`, `state_proj`, `action_in_proj`, `action_out_proj` và time MLP. Xác nhận tất cả LoRA target dự kiến nhận gradient hữu hạn và thực sự đổi sau optimizer step.
4. **Gate A — memorization.** Chỉ coi data/optimizer path đạt khi tiny train exact-action ≥90%, zone accuracy ≥95% và MAE ≤0,25. Nếu chưa đạt, chưa chạy validation sweep và chưa train dài.
5. **Tăng capacity có kiểm soát nếu Gate A vẫn fail.** Tạo config/run mới để so sánh r=16 trước, rồi mới cân nhắc r=32 hoặc mở thêm projection/action expert. Giữ VLM/vision frozen ở thí nghiệm đầu để chỉ thay một yếu tố. Không ghi đè run r=8.
6. **Gate B — fixed validation.** Khi Gate A pass, chạy các checkpoint trên cùng một fixed validation subset; chọn bằng exact-action trước, zone accuracy sau, MAE để phá hòa. Không dùng test để chọn checkpoint, LR, rank hoặc số steps.
7. **Full validation và ablation tối thiểu.** Chạy full validation cho cấu hình thắng; báo exact-action, zone accuracy, MAE, breakdown theo intent/language và always-OFF baseline. Chỉ thêm vision/language ablation khi cần chứng minh model thật sự dùng cả hai modality.
8. **Final test một lần.** Sau khi khóa checkpoint và protocol bằng validation, chạy full test đúng một lần. Chỉ cân nhắc điều khiển phần cứng khi model vượt baseline, STOP không còn 0% và safety/off-target errors đạt ngưỡng nhóm chấp nhận.

Web hiện tại vẫn hữu ích để quan sát failure cases trong các thí nghiệm trên. Nó phải tiếp tục hiển thị quality warning cho đến khi checkpoint mới vượt các gate đã định.

