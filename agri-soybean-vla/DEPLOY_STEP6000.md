# Dùng checkpoint Soybean step 6000 trên máy này

Ngày kiểm tra: 2026-11-12. Repo `/home/hanhpham/agriVLA`, nhánh `feat/compose-dataset`.

## 1. Kết luận cần hiểu trước

Đã nhận checkpoint LoRA train tới step 6000, có thể nạp để thử inference và demo web. Chất lượng hiện tại chưa đạt baseline. Tài liệu `README_TRAINING_DEMO_PROGRESS.md` của lần chạy trước ghi test exact-action 1,81%, zone accuracy 32,50%, STOP exact 0%; baseline luôn OFF đạt exact-action 28,89%. Đây là số liệu được ghi trong tài liệu bàn giao, không phải full test chạy lại trong lượt kiểm tra này.

Audit tiếp theo trên máy này đã kiểm56mẫu thuộc train: exact0/56, zone36,90%, STOP0/8 với seed42. Loader/checkpoint/normalization/factory inputs được đối chiếu; output rất nhạy seed. Xem **MODEL_AUDIT_STEP6000.md** để phân biệt kết quả đã đo với giả thuyết nguyên nhân; không xem checkpoint là model đạt chất lượng vì demo lên READY.

ZIP chỉ chứa checkpoint và training state; không chứa metrics/predictions/logs bước 9 hoặc base weights. Vì vậy không thể chỉ từ ZIP xác nhận mọi tiêu chí đánh giá bước 9 đã PASS. Model hiện phù hợp quan sát inference/failure cases; web/API trả action, chưa điều khiển van hay bơm thật.

## 2. File nào là model được train?

Checkpoint đã đặt vào:

```text
agri-soybean-vla/outputs/train/
  step8-lora-6000-t1200-20261109-resume4500-v3/
    checkpoints/006000/
      pretrained_model/
        adapter_model.safetensors       # Trọng số LoRA đã học, 371.328 parameters
        adapter_config.json             # LoRA r=8, alpha=8, các lớp được gắn adapter
        config.json                     # Schema 3 ảnh/state[3]/action[3], horizon 1
        tokenizer/                      # Tokenizer dùng lúc train, max length 96
        policy_preprocessor.json        # Pipeline đầu vào
        policy_preprocessor_*.safetensors
        policy_postprocessor.json       # Pipeline hoàn nguyên action
        policy_postprocessor_*.safetensors
        agri_train_stats.json           # Stats full TRAIN của wording_v1
        agri_schema.json
        agri_provenance.json
        agri_checkpoint_lock.json       # SHA256 của các file checkpoint
        train_config.json
      training_state/
        optimizer_state.safetensors
        optimizer_param_groups.json
        scheduler_state.json
        rng_state.safetensors
        training_step.json              # step=6000
```

`adapter_model.safetensors` khoảng 1,5 MB vì chỉ lưu phần LoRA đã học. Mô hình inference hoàn chỉnh là **base SmolVLA + LoRA step6000 + config/tokenizer/processors/stats**. Không thể chỉ copy mỗi adapter rồi dùng như một model độc lập. `training_state/` dùng cho resume training; inference không cần đọc optimizer.

`agri_provenance.json` ghi `optimizer_steps=0` và `gpu_fit_verified=false` vì đó là thông tin lúc khởi tạo bundle bước 6, được mang theo checkpoint. Step đã lưu thực tế đọc ở `training_state/training_step.json`, scheduler và optimizer: cả ba đều ghi 6000. Đường dẫn `/home/huanminh/...` trong config là lịch sử máy train; loader của repo nạp pinned base và rebase backbone/tokenizer sang máy hiện tại, không sửa checkpoint gốc.

Hai model nền đã khóa:

- `lerobot/smolvla_base` revision `5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6`.
- `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` revision `7b375e1b73b11138ff12fe22c8f2822d8fe03467`.

Chúng nằm trong `.cache/models/`, được khôi phục từ gói bước 6. Bundle theo đường dẫn máy này nằm ở `models/prepared/wording_v1/`. `initial_adapter` trong bundle là adapter chưa train để kiểm pipeline; luôn truyền `--checkpoint` step6000 khi chạy model đã train.

## 3. Mở web để dùng model

Chạy trong terminal WSL/Linux có GPU access, từ repo root:

```bash
cd /home/hanhpham/agriVLA
AGRI_TRAIN_PY=agri-soybean-vla/.venv-train/bin/python
CHECKPOINT=agri-soybean-vla/outputs/train/step8-lora-6000-t1200-20261109-resume4500-v3/checkpoints/006000/pretrained_model

$AGRI_TRAIN_PY -m agri_vla.cli demo-web \
  --checkpoint "$CHECKPOINT" --device cuda --port 8000
```

Mở `http://127.0.0.1:8000` trong trình duyệt, giữ terminal chạy và đợi trạng thái READY. Upload ba ảnh theo thứ tự zone1/zone2/zone3, nhập lệnh Việt hoặc Anh rồi bấm chạy. Ba ảnh mẫu nằm trong `output/DACN-DATN/example-inputs/`. Ghi chú mức héo trên UI chỉ để đối chiếu, không đi vào model.

Nếu port 8000 bận, đổi `--port 8001`. Ctrl+C để dừng. Khi dùng SSH/VS Code Remote, forward port từ máy chạy model về máy mở trình duyệt. Web bind localhost và không tự public ra Internet. Tránh mở hai process model cùng lúc trên GPU 4 GB.

## 4. Chạy một lệnh và nhận JSON

Chạy khi đã dừng process web để giải phóng GPU; nếu dùng terminal mới, khai báo lại các biến ở phần 3:

```bash
EXAMPLES=agri-soybean-vla/output/DACN-DATN/example-inputs
RESULT="agri-soybean-vla/output/prediction_$(date +%Y%m%d_%H%M%S).json"

$AGRI_TRAIN_PY -m agri_vla.cli infer \
  --checkpoint "$CHECKPOINT" --device cuda \
  --images "$EXAMPLES/zone1.jpg" "$EXAMPLES/zone2.jpg" "$EXAMPLES/zone3.jpg" \
  --instruction 'Xem cả 3 vườn, vườn nào đạt ngưỡng héo thì tưới nhé.' \
  --output "$RESULT"

cat "$RESULT"
```

Thay ba path ảnh và instruction để dùng ảnh của bạn. Mỗi file output phải là file mới; CLI từ chối ghi đè. Không truyền checkpoint sẽ nạp initial adapter chưa train. `--device cpu` có thể dùng để chẩn đoán khi không có CUDA nhưng chậm hơn.

Ba câu dễ thử trên cùng ảnh:

- `Xem cả 3 vườn, vườn nào đạt ngưỡng héo thì tưới nhé.`: AUTO, target proxy của bộ ảnh mẫu là `[0,1,3]`.
- `Tắt hết.`: STOP_ALL, target `[0,0,0]`.
- `Tưới cả 3 vườn mức 2 nhé.`: WATER_ALL, target `[2,2,2]` bất kể wilt.

Các target ở đây là quy tắc tham chiếu để đối chiếu. Model tự dự đoán; code không thay output theo teacher/intent để làm cho đúng. Câu WATER_ALL thiếu mức bị từ chối; input vượt 96 tokens bị từ chối thay vì cắt mất nội dung.

## 5. Đọc output và gọi từ ứng dụng khác

Luồng inference: ba ảnh RGB → letterbox 256×256 + lệnh + state `[0,0,0]` → SmolVLA/LoRA → action liên tục → inverse normalization bằng stats train → clip `[0,3]` → làm tròn `floor(x+0.5)`.

- `continuous_action`: ba giá trị model thực dự đoán sau inverse normalization.
- `clipped_action`: giá trị sau giới hạn miền 0–3.
- `quantized_action`: ba số nguyên `[zone1,zone2,zone3]`.
- `out_of_range`: chiều nào đã vượt miền trước clipping; cần đọc cùng output.

Mức 0/1/2/3 tương ứng OFF/LOW/MEDIUM/HIGH của policy tổng hợp. Chúng chưa phải lít nước hoặc số giây mở van. State zeros chỉ giả định ba van đầu vào đang đóng. Model không tự cập nhật trạng thái cảm biến/phần cứng.

Web demo đã là một HTTP API nạp model một lần:

```bash
curl http://127.0.0.1:8000/api/status
```

Gọi `POST /api/infer` với JSON chứa đúng ba field: `instruction` (string), `images` (3 data URLs base64), `wilt_notes` (3 giá trị null hoặc 0–4, chỉ ghi chú UI). Ví dụ chạy từ repo root trong terminal khác khi web đang READY:

```bash
agri-soybean-vla/.venv/bin/python - <<'PY'
import base64, json
from pathlib import Path
from urllib.request import Request, urlopen

root = Path('agri-soybean-vla/output/DACN-DATN/example-inputs')
images = ['data:image/jpeg;base64,' + base64.b64encode((root/f'zone{i}.jpg').read_bytes()).decode()
          for i in (1, 2, 3)]
payload = {'instruction': 'Tắt hết.', 'images': images, 'wilt_notes': [None, None, None]}
req = Request('http://127.0.0.1:8000/api/infer',
              data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
with urlopen(req, timeout=300) as response:
    print(json.dumps(json.load(response), ensure_ascii=False, indent=2))
PY
```

## 6. Báo cáo nào có ở máy này?

ZIP và toàn bộ checkpoint đã kiểm SHA256; các tensor adapter/optimizer hữu hạn, optimizer step=6000. Gói bước 6 được kiểm toàn bộ 2.375 file trước khi khôi phục runtime. Một cache copy dở được giữ ở `reports/checkpoint-step6000/incomplete-cache/`, ngoài cache model đang dùng.

Receipt của lượt kiểm tra này ở `reports/checkpoint-step6000/`: `checkpoint-audit.json`, `runtime-restore.json`, `preflight.json`, `validate-revision.json`, `prepare-training.log` và các kết quả inference/API được lưu tại đây. Các artifact local đều Git ignored; ZIP gốc được giữ nguyên ở `output/DACN-DATN/`.

Đã chạy thật trên GPU T1200: bundle preparation PASS; full round-trip 3600/720/720 records PASS; tokenizer96 zero truncation 5040 records; CLI infer PASS; web READY, HTML/CSS/JS và ba request API đều HTTP200. CLI và API cho cùng continuous prediction trên cùng ảnh/lệnh/seed (max absolute error=0, tolerance=1e-5). Không retrain hoặc chạy lại full test720.

Ba phép thử trên cùng bộ ảnh mẫu (teacher wilt `[0,2,4]`):

| Lệnh | Model dự đoán | Target proxy | Đúng toàn vector? |
|---|---|---|---|
| AUTO | `[1,3,1]` | `[0,1,3]` | Không |
| Tắt hết. | `[0,2,0]` | `[0,0,0]` | Không |
| Tưới cả 3 vườn mức 2 nhé. | `[3,3,2]` | `[2,2,2]` | Không |

Raw prediction đầu tiên `[1.27332735,3.39073348,0.87161642]`; zone2 vượt miền trước clipping. Chi tiết lưu ở `reports/checkpoint-step6000/infer-auto.json` và `web-api-check.json`. Đây là ba diagnostic examples, không phải một benchmark mới. `status=pass` của receipt xác nhận pipeline chạy đúng kỹ thuật; không có nghĩa prediction đúng target. Trường `training_status=not_verified` trong output infer là nhãn generic của evaluator, không phải kiểm tra optimizer state; receipt `checkpoint-audit.json` đã kiểm step6000 riêng.

Viewer `scripts/view_step9.sh` chỉ xem báo cáo cũ, không chạy model. ZIP này không có các `outputs/eval/step9-.../metrics.json` và `predictions.jsonl` mà viewer yêu cầu, nên muốn xem đủ giao diện kết quả bước 9 cần lấy thêm các artifact đó từ máy bạn đã train. Không cần chúng để chạy web demo hoặc infer ba ảnh.
