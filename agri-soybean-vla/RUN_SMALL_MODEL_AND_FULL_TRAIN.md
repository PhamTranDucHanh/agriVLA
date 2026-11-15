# Chạy model tập nhỏ và chuẩn bị full train — 2026-11-15

Đọc kết luận hiện hành trong SMALL_CASES_EXPERIMENTS.md. Model tập nhỏ là chẩn đoán trên TRAIN, chưa phải model đã được chứng nhận chất lượng full dataset. Chỉ khi báo cáo TRAIN60 đạt 100% tại từng noise seed và fresh reload khớp mới qua gate của runner full mới. Không dùng con số overfit này làm test accuracy.

## Model và cách dùng

Từ root repo:

```bash
cd /home/hanhpham/agriVLA
AGRI_PY="$PWD/agri-soybean-vla/.venv-train/bin/python"
source agri-soybean-vla/reports/small-cases/selected.env

"$AGRI_PY" -m agri_vla.cli demo-web \
  --checkpoint "$AGRI_MODEL" --device cuda --port 8000
```

Mở http://127.0.0.1:8000. Upload ba ảnh theo đúng thứ tự zone1/2/3, nhập nguyên câu từ demo-examples hoặc bảng case_coverage.md rồi Predict. Input model: ba ảnh, instruction và state[0,0,0]. Ghi chú mức héo trên UI không được đưa vào model. Output là vector ba mức0/1/2/3 (OFF/LOW/MEDIUM/HIGH) và continuous values; đây chưa phải lít nước hay lệnh điều khiển van thật.

File weights đã train là `$AGRI_MODEL/adapter_model.safetensors`. Cần giữ toàn bộ thư mục pretrained_model (adapter_config, native config, tokenizer, processors, stats, locks) cùng pinned base/backbone cache. Adapter vài MiB không phải toàn bộ0.45B parameters: phần base frozen dùng lại từ cache. Không dùng models/prepared/wording_v1/initial_adapter vì đó là adapter chưa train.

Dataset browser: http://127.0.0.1:8765/?view=samples&page_size=48. Nếu server đã dừng:

```bash
bash agri-soybean-vla/scripts/browse.sh --port 8765
```

Bảng60mẫu/câu/target/link ảnh: reports/small-cases/case_coverage.md. Dự đoán đầy đủ từng seed: reports/small-cases/predictions.csv. Mẫu CLI ở reports/small-cases/demo-examples/run-auto.sh; script dùng đúng checkpoint trong selected.env và output mới mỗi lần.

## Kiểm tra lại checkpoint

Tắt demo bằng Ctrl+C trước khi chạy các lệnh GPU khác. Giữ output mới để không ghi đè bằng chứng.

```bash
AGRI_TAG=$(date +%Y%m%d-%H%M%S)
"$AGRI_PY" -u -m agri_vla.small_train --stage cases60 \
  --evaluate-checkpoint "$AGRI_MODEL" \
  --output "agri-soybean-vla/reports/small-cases/user-recheck-$AGRI_TAG.json"
```

Lệnh đo60TRAIN×5noise seeds, batch1; không train. Đọc gate_passed và by_seed; status gate_not_met nghĩa chưa học đúng toàn bộ các trường hợp. Exit code0 chỉ xác nhận evaluator chạy xong, không thay thế đọc quality gate. Không tùy chọn seed đẹp để báo accuracy.

## Full train: chuẩn bị, chưa chạy trong task này

Giữ WORDING_V1/train3600, val720, test720, ba camera256, tokenizer96, STATE IDENTITY/ACTION MEAN_STD và saved full-train stats. Không compose lại dataset hoặc thay normalization theo subset. File training_v0.yaml có tên lịch sử nhưng dataset_root đang là wording_v1; runner mới override hyperparameters từ YAML thí nghiệm, không tự quay lại recipe cũ.

Runner mới candidate_train.py đã kiểm cấu hình/data trên CPU; nhánh full GPU/long run chưa được thực thi trong task này. Cơ chế optimizer/native checkpoint/AMP và loader continuation đã kiểm GPU trên train nhỏ, CPU regression cả sampler3600. Vẫn phải chạy smoke full100steps trước một run dài. Chưa gọi đây là full-training proof.

Xem kế hoạch không chạy optimizer:

```bash
AGRI_RUN="agri-soybean-vla/outputs/train/full-smoke-$(date +%Y%m%d-%H%M%S)"
"$AGRI_PY" -m agri_vla.candidate_train \
  --experiment-config "$AGRI_EXPERIMENT_CONFIG" --profile "$AGRI_PROFILE" \
  --init-checkpoint "$AGRI_MODEL" --steps 100 --output "$AGRI_RUN"
```

Kiểm resolved_config: dataset.root kết thúc wording_v1/full/train; episodes=null (đủ3600); batch4; lr/scope/num_steps đúng profile; eval_split=0, không auto chia train; saved stats không đổi. full_execution_gate_passed=false nếu chưa cung cấp certificate hợp lệ.

**Chỉ khi đạt TRAIN60/fresh-reload gate**, thêm evidence và execute để chạy full smoke100:

```bash
/usr/bin/time -v "$AGRI_PY" -u -m agri_vla.candidate_train \
  --experiment-config "$AGRI_EXPERIMENT_CONFIG" --profile "$AGRI_PROFILE" \
  --init-checkpoint "$AGRI_MODEL" --evidence "$AGRI_EVIDENCE" \
  --steps 100 --output "$AGRI_RUN" --execute
```

Runner từ chối chạy nếu evidence không đạt100% từng seed. Đây là điều kiện chất lượng của yêu cầu học đúng bộ nhỏ, không phải yêu cầu xin phép mới. Lệnh này chưa phải lệnh nên chạy ngay nếu kết luận hiện hành còn FAIL. Không sửa JSON certificate để vượt gate.

Run mới dùng weights từ model tập nhỏ thuộc TRAIN, **reset optimizer/scheduler/RNG/step** cho full3600. Không mang optimizer của subset sang rồi gọi đó là resume full. Profile giữ LR/scope/sampler đã đo; không mặc định fresh-wide từ base cũng đạt chỉ vì curriculum đạt.

Nếu smoke finite và resource phù hợp, resume checkpoint100 tới bước1000 trong output mới:

```bash
AGRI_RUN_NEXT="agri-soybean-vla/outputs/train/full-phase1-$(date +%Y%m%d-%H%M%S)"
/usr/bin/time -v "$AGRI_PY" -u -m agri_vla.candidate_train \
  --experiment-config "$AGRI_EXPERIMENT_CONFIG" --profile "$AGRI_PROFILE" \
  --resume-from "$AGRI_RUN/checkpoints/000100" \
  --steps 1000 --output "$AGRI_RUN_NEXT" --execute
```

--steps1000 là mốc tích lũy, thêm900updates sau100. Resume giữ cùng dataset/batch/world1/workers0/noaugmentation/profile, native optimizer/scheduler/RNG/step, AMPscaler và sampler continuation. Kiểm restored-state.json. Không dùng lệnh train-upstream cũ để vô tình quay lại batch1/qv/Euler10 hoặc bỏ sửa loader continuation.

Sau mỗi phase, process train thoát rồi mới evaluate full VAL ở process riêng:

```bash
"$AGRI_PY" -u -m agri_vla.cli evaluate --split val --device cuda \
  --checkpoint "$AGRI_RUN_NEXT/checkpoints/001000/pretrained_model" \
  --output "agri-soybean-vla/outputs/eval/val-phase1-$(date +%Y%m%d-%H%M%S)"
```

Đọc metrics.json và predictions.jsonl: exact cảvector, zone, STOP, false-watering, từng intent/ngôn ngữ/wording. So với always-OFF và checkpoint6000 trên cùng VAL/protocol. Chọn checkpoint theo VAL; test720 chỉ đánh giá cuối khi đã đóng băng lựa chọn. Không tuning trên test. Tiếp tục1000→2000… tối đa6000 ban đầu chỉ khi VAL cho thấy có ích, dùng --resume-from checkpoint cuối và output mới mỗi phase. Constant LR/warmup20 giữ nguyên; nếu đổi schedule/LR/scope/batch thì version config mới và kiểm lại tương ứng.

## Tài nguyên và thời gian

Số đo trên T1200 Laptop4096MiB, không phải benchmark của GPU khác:

| Workload batch4 | Thời gian update thuần ước tính | Số lượt dataset |
|---|---:|---:|
|60mẫu,100steps|5–7phút|6,67|
|60mẫu,300steps|15–20phút|20|
|60mẫu,1000steps|50–67phút|66,67|
|3600mẫu,1000steps|50–67phút|1,11|
|3600mẫu,6000steps|5–7giờ|6,67|

Cộng thêm khởi tạo model, data/cache, save, đánh giá. Full chưa đo: ETA suy từ3–4s/update của tập nhỏ. Mỗi phase VAL720 tốn thêm720×latency inference +load; năm noise seeds gấp5 số predictions. Euler2 giảm chi phí inference/eval, không giảm tương ứng chi phí training forward/backward (train lấy một random timestep mỗi update).

Wide-B4 đã đo torch allocated khoảng1880MiB/reserved1916MiB; toànGPU ở run resume có peak3750/4096MiB, còn346MiB tại peak. NVML gồm driver và allocation khác, thay đổi theo máy/ứng dụng; không lấy snapshot2250MiB làm peak. Longest-prompt resource report của checkpoint hiện hành được dẫn trong selected-model.json. Full dataset giữ cùng batch/input thường không nhân VRAM theo3600/60; vẫn cần kiểm full smoke và theo dõi suốt run. Không mở hai model GPU đồng thời.

RAM process partial: peak RSS high-water khoảng3,65GiB (snapshot), RSS lúc đo2,85GiB. Full RAM chưa đo; dự phòng thêm cho dataset/cache/OS, tránh để WSL thiếu RAM hoặc swap nhiều. Guest hiện có7,6GiB RAM; đây không phải lời cam kết mọi full workload fit. VmPeak trong /proc là không gian địa chỉ ảo, không phải RAM resident. /usr/bin/time -v báo Maximum resident set size để đo run thật.

Ổ đĩa hiện có: module33GiB, gồm15GiB handoff; data4,6GiB, modelcache2,8GiB, envtrain6,1GiB, HF caches khoảng3,1GiB; không cần tải lại các phần này. Checkpoint wide đủmodel/tokenizer/processors/optimizer/RNG khoảng23,7MiB. Lưu mỗi500steps/6000steps:12checkpoints≈284MiB; mỗi100steps≈1,4GiB. Thêm report/log, receipt resume và cache mới; dự phòng thêm2–5GiB cho một full run ở workspace hiện tại là hợp lý, không phải số đo growth full. Máy lúc đo còn≈881GiB. Giữ hay xóa handoff là lựa chọn backup riêng, không tự xóa.

Theo dõi GPU ở terminal khác:

```bash
watch -n 1 /usr/lib/wsl/lib/nvidia-smi
```

Runner full mới cũng ghi Torch peak, NVML samples1Hz, RSS high-water, elapsed/step times, finite guards, adapter delta và native checkpoints ở reports/full-candidate/RUN/. Không dùng model loading/inference một lần thay cho kiểm backward/update hay VAL quality. Nếu OOM: kiểm process GPU, đóng demo của chính mình, kiểm batch/AMP/256/3camera; đổi config có version và rerun. Không kill process lạ, không bịa accumulation/4bit flags. Giữ máy không sleep khi train; checkpoint resume giúp tiếp tục nếu run gián đoạn.
