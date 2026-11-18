# Chạy biến thể K/V và chuẩn bị full TRAIN

Đọc kết luận GPU trong [LATE_VLM_GPU_2026-11-16.md](LATE_VLM_GPU_2026-11-16.md).
Lệnh đã đóng gói không đồng nghĩa checkpoint đã đạt điều kiện full train.
`configs/late_vlm_full_ready.json` chỉ được tạo bởi `qualify` sau khi mọi gate đạt;
không tự viết file này để bỏ qua kết quả VALID. Không chạy full trong lượt kiểm chứng nhỏ.

## Dataset và bài toán thật

- TRAIN: `data/lerobot/wording_v1/full/train/`, **3.600 records**.
- VALID: `data/lerobot/wording_v1/full/val/`, **720 records**, dùng chọn checkpoint.
- TEST: `data/lerobot/wording_v1/full/test/`, **720 records**, chỉ đo cuối sau khi đóng băng lựa chọn.
- File `sample_metadata.jsonl` trong mỗi split ánh xạ record → câu, ba đường dẫn ảnh, target và provenance.
  Manifest gốc ở `data/manifests/wording_v1/{train,val,test}.jsonl`.
  Ảnh gốc được tham chiếu từ `data/raw/zenodo_8256382/images/`.
- Full runner dùng LeRobot export đã có, `episodes=null`, không export/copy ảnh lại,
  không trộn `v0` và **chưa dùng manifest counterfactual**.
- Mỗi input: ảnh zone1/2/3 đúng thứ tự camera1/2/3, RGB letterbox256×256,
  câu VI/EN, state `[0,0,0]`, tokenizer96.
- Target: vector ba số nguyên0/1/2/3 = OFF/LOW/MEDIUM/HIGH.
  Saved processor dùng action mean/std tính trên full TRAIN; không tính lại theo subset.
  Teacher wilt/intent/target dùng giám sát hoặc kiểm nhãn, không đi vào input inference.
- Output model là action continuous; native postprocessor đưa về thang gốc,
  clip0..3 và làm tròn `floor(x+.5)` để hiện ba mức tưới.
  Chưa có đơn vị lít nước hay kết nối van thật. Không ghi đè output bằng rule STOP/WATER_ALL.

File config mang tên lịch sử `training_v0.yaml`, nhưng `dataset_root` hiện hành
trỏ WORDING_V1. Full typed bridge lấy hyperparameters từ `late_vlm_kv_v1.json`.

## Xem kế hoạch, không train

Từ repo root:

```bash
cd /home/hanhpham/agriVLA
bash agri-soybean-vla/scripts/full_late_vlm.sh --steps 6000
```

Lệnh in đường dataset, batch, số lượt qua data, outputs và `execution_ready`.
CPU config resolution thực đã kiểm trong
`reports/latekv-gpu-20261116/full-resolved-config-final-plan.json`;
đó là kiểm config, không phải full backward/GPU proof.

## Lệnh full khi có recipe đủ bằng chứng

```bash
cd /home/hanhpham/agriVLA
bash agri-soybean-vla/scripts/full_late_vlm.sh \
  --recipe agri-soybean-vla/configs/late_vlm_full_ready.json \
  --steps 6000 --patience 3 --max-new-gib 1 \
  --output agri-soybean-vla/outputs/train/full-latekv-run1 \
  --execute --final-test
```

Hiện recipe chưa đạt thì lệnh **dừng trước training**; đây không phải lỗi CUDA.
`--final-test` cho phép workflow tương lai đánh giá TEST đúng một checkpoint đã
chọn bằng VALID, không dùng TEST để chọn model hoặc sửa tham số.
Lượt thử nhỏ hiện tại không mở TEST.

Runner dùng adapter của checkpoint nhỏ đã đủ bằng chứng làm weights khởi tạo,
**reset optimizer/scheduler/RNG/step** cho full TRAIN. Sau đó mới native full-state
resume giữa các phase. Batch4, LR1e-4, warmup20 rồi constant, endpoint_t1,
native Euler1, r8/alpha8 và scope đã kiểm được giữ nguyên.

Trình tự: full smoke100 → 1000 → 2000… tối đa6000 cumulative updates.
Từ1000, mỗi phase đánh giá đầy đủ VALID720 ở process riêng để giải phóng GPU.
Chọn best theo exact action, rồi zone accuracy, rồi MAE thấp hơn.
Nếu VALID không cải thiện ba phase liên tiếp thì dừng sớm, giữ best.
Chỉ giữ best+latest thuộc run mới; không xóa checkpoint nguồn/model cũ.
Nếu bị gián đoạn, chạy lại đúng lệnh và thêm `--resume`, giữ nguyên output,
recipe, steps, budget và patience. Không đổi code/config giữa run.

6000 là **ngân sách trần**, không phải cam kết convergence hay accuracy.
TRAIN3600/batch4 =900updates/epoch;6000≈6,67epochs.
Pilot280/300updates≈4,29epochs, còn full300 chỉ≈0,33epoch.
Full có độ phủ ảnh/scene cao hơn; có thể tổng quát hóa tốt hơn hoặc làm lộ underfit,
domain gap, forgetting. Không ngoại suy accuracy từ TRAIN60 hoặc VALID112.

## Output có ý nghĩa gì

Giả sử `RUN=agri-soybean-vla/outputs/train/full-latekv-run1`:

| Path | Ý nghĩa |
|---|---|
| `$RUN/*-phase-*/checkpoints/NNNNNN/pretrained_model/adapter_model.safetensors` | **Weights LoRA đã train**, gồm expert và hai K/V; không chứa toàn bộ base0.45B |
| Cả thư mục `pretrained_model/` | Adapter config, native policy config, tokenizer, saved processors, stats, schema, provenance, typed metadata và locks; giữ cùng nhau để load |
| `checkpoints/NNNNNN/training_state/` | Adam optimizer, scheduler, RNG, cumulative step và AMP scaler để resume thật; riêng weights không đủ resume |
| `$RUN/val-NNNNNN/{metrics.json,predictions.jsonl}` | Điểm VALID720 và dự đoán từng mẫu ở các mốc |
| `$RUN/progress.json` | Latest, best, lịch sử VALID, trạng thái/early-stop |
| `$RUN/final_model/` | Bản deploy của **best theo VALID**, có thể khác latest |
| `$RUN/test-final/` | Kết quả TEST cuối, sau khi chọn model |
| `$RUN/final-report.json` | Chất lượng thật, baseline, tài nguyên và HTTP smoke; `completed_quality_not_met` vẫn là chưa đạt mục tiêu |
| `$RUN/deploy.sh` | Lệnh mở web bằng typed loader tương ứng |
| `reports/full-candidate/PHASE/` | Resolved plan, shapes/tokens thật, finite checks, adapter delta, VRAM/timing/RAM và restore receipts |

`models/prepared/wording_v1/initial_adapter/` là adapter khởi tạo,
**không phải kết quả model đã train**. Không lấy file ZIP/handoff làm model runtime.
Adapter dùng chung pinned base/backbone cache; mang mỗi safetensors sang máy khác
mà thiếu config/tokenizer/processors/base thì không phải gói deploy đầy đủ.

## Mở web kiểm checkpoint nhỏ mới

Chỉ chạy khi không còn train/evaluate GPU; đây là web chẩn đoán, không tự thay
`reports/small-cases/selected.env` hay checkpoint deploy cũ:

```bash
cd /home/hanhpham/agriVLA
export HF_DATASETS_CACHE="$PWD/agri-soybean-vla/.cache/hf-datasets"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
agri-soybean-vla/.venv-train/bin/python -m agri_vla.late_vlm_full web \
  --checkpoint agri-soybean-vla/outputs/small-cases/latekv-gpu20261116-candidate-s300/checkpoints/000300/pretrained_model \
  --device cuda --port 8000
```

Mở **http://127.0.0.1:8000**, tải đủ ba ảnh theo zone, nhập nguyên câu rồi Predict.
Ghi chú mức héo trên UI không được truyền vào model. Sai prediction là chất lượng
model, không được diễn giải technical web PASS thành accuracy PASS.

Xem ảnh/câu/target bằng browser dataset có sẵn:

```bash
bash agri-soybean-vla/scripts/browse.sh --port 8765
```

Mở http://127.0.0.1:8765/?view=samples&page_size=48.
Sau full tương lai thành công, mở web bằng `bash "$RUN/deploy.sh"`.

## Disk và tài nguyên

Dùng chung cache/ảnh, không tải lại model hoặc đóng ZIP.
Checkpoint candidate đủ training state khoảng24MiB, weights khoảng6,8MiB;
best+latest khoảng48MiB, final_model khoảng10MiB. Cần tính thêm reports,
restore receipts, checkpoint tạm và cache growth; hướng dẫn `--max-new-gib 1`
dừng khi vượt budget tại các điểm kiểm, **không là hard quota của Windows/VHDX**.
Dự phòng thêm2–5GiB ổ thật để chịu cache/temp/swap, không suy dung lượng từ900GiB
free của filesystem WSL. Lúc kiểm16/11: C: còn khoảng80GiB, D:43GiB.
Số đo VRAM/time mới và ETA ngoại suy nằm trong báo cáo GPU; chưa đo full run.

Đóng GPU demo của chính mình bằng Ctrl+C trước training. Không kill process lạ.
Nếu OOM, dừng và kiểm workload/process/memory; không tự bỏ camera/split,
không bịa accumulation hoặc4-bit flags để vượt bài kiểm.
