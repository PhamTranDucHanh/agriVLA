# Rà soát checkpoint step6000 — 2026-11-12

## Kết luận

Checkpoint có trọng số LoRA đã update và được nạp đúng, nhưng chưa fit được policy trên các mẫu train được kiểm. Không có cơ sở gọi model đã học thành công chỉ vì hoàn tất optimizer steps hoặc loss giảm. Trong các phép kiểm tra dưới đây chưa tìm thấy lỗi checkpoint loader, input train/infer, normalization, cache hoặc precision đủ giải thích chất lượng thấp. Chưa xác định được một nguyên nhân tối ưu duy nhất như LR, rank hoặc số bước; không đổi các tham số đó theo suy đoán.

Không có optimizer step, backward, chỉnh checkpoint hoặc tuning trên validation/test trong lượt audit. Scripts và raw receipts ở `reports/model-audit/` (local, Git ignored). Các phép thử sampling chỉ là diagnostic trên train, không thay protocol của model đang deploy.

## 1. Kết quả thực trên train

Dùng checkpoint step6000 và subset 56 train records cố định, 28 VI +28 EN, mỗi intent8records. Subset lấy bằng `tiny_overfit_selection(c,56)` hiện có, nhưng **không train một model tiny mới**. Đây là kiểm tra model full-train step6000 trên 56 ảnh/lệnh thuộc train, với seed42 như evaluator hiện hành.

| Metric | Checkpoint | Baseline luôn OFF trên cùng subset |
|---|---:|---:|
| Exact-vector accuracy | 0/56 = 0% | 12/56 = 21,43% |
| Per-zone accuracy | 62/168 = 36,90% | 90/168 = 53,57% |
| Continuous MAE | 1,10457 | — |
| STOP exact | 0/8 | 8/8 |

VI và EN đều exact0%, zone36,90%; không thể giải thích toàn bộ lỗi là do riêng tiếng Việt. Có39/168 continuous components ngoài miền trướcclip và55/90 target-OFF components bị tưới sai. 11 câu Việt trên cùng scene train wilt[0,2,4] cũng đều sai vector.

Chưa chạy toàn train3600 hoặc full test720 trong audit. 56 mẫu là panel chẩn đoán cân bằng, không đại diện ngẫu nhiên cho phân bố vận hành.

## 2. Những lỗi đã đối chiếu trực tiếp

| Phép kiểm | Kết quả và phạm vi |
|---|---|
| Tensor LoRA trên GPU so checkpoint | 74 tensors đúng tên; max absolute error=0 |
| Tắt adapter cùng ảnh/lệnh/noise | AUTO thay từ continuous[1.17966,3.44926,1.13771] sang[0.82379,1.51020,0.84963]; adapter đang ảnh hưởng output |
| Normalize→unnormalize target | Sai số max=0 trên mẫu [0,1,3] |
| Upstream train factory vs raw-image inference | Sai số max=0 cho cả3camera, state, action; task string trùng |
| Input/label mâu thuẫn | Không có cùng ordered source triplet+câu lệnh nhưng target khác trong3600records train |
| Joint forward dùng khi train vs cached denoiser | Cùng x_t/time/condition, chênh velocity đầu3dim tối đa0.005761 trên8comparisons (4times × AMP/FP32) |
| Sampler không dùng cache | AUTO/STOP/WATER_ALL2 vẫn giữ vector sai như sampler gốc trên3mẫu train |
| Inference FP32 thay FP16 | Không thay quantized vector trên3mẫu train được kiểm |
| Giữ29padding dimensions theo manifold train x_pad=t*noise_pad | Không thay quantized vector trên3mẫu train được kiểm |

Các thử cache/precision/padding không chứng minh mọi numerical bug đều bị loại trừ trên mọi đầu vào. Chúng không cho thấy đây là nguyên nhân chính của các failures được kiểm. Không sửa vendor hoặc ghi các biến thể sampling diagnostic thành mặc định.

Ví dụ sampling AUTO trên scene train:

- Gốc FP16/cache: [1.179657,3.449257,1.137714] → [1,3,1].
- Không cache: [1.179812,3.451092,1.137705] → [1,3,1].
- FP32: [1.161020,3.446613,1.133490] → [1,3,1].
- Analytic padding: [1.144887,3.456214,1.160093] → [1,3,1].
- Target: [0,1,3].

## 3. Seed nhiễu có ảnh hưởng rất mạnh

Giữ nguyên ảnh train, instruction, checkpoint và processor; chỉ đổi noise seed:

| Intent / target | Seed0 | Seed42 | Seed43 |
|---|---|---|---|
| AUTO / [0,1,3] | [0,1,0] | [1,3,1] | [3,3,3] |
| STOP / [0,0,0] | [0,0,0] | [0,2,0] | [1,2,1] |
| WATER_ALL2 / [2,2,2] | [1,0,0] | [2,3,2] | [3,3,3] |

STOP đúng ở4/7seeds đã thử trên một scene nhưng sai ởseed42. Vì vậy STOP0% trong một protocol seed42 không có nghĩa mọi seed luôn sai; đồng thời chọn lại seed để ra ví dụ đẹp không chứng minh đã sửa model. Model chưa ổn định theo điều kiện đầu vào, còn phụ thuộc mạnh vào noise. Không đổi seed mặc định hoặc chọn seed tốt nhất theo test.

## 4. Vì sao loss giảm nhưng vẫn sai mẫu train?

SmolVLA dùng flow matching. Training học velocity từ action đã trộn noise, không trực tiếp tối ưu exact-vector accuracy của output sau10bước sampling và lượng tử hóa. Hoàn tất update/finite loss chỉ kiểm hoạt động số học; model vẫn có thể chưa học đủ để khử nhiễu và suy action đúng từ ảnh/lệnh.

6000steps ×batch1 /3600records chỉ tương đương1,67lượt dữ liệu. Con số này là lý do cần kiểm hội tụ, không tự chứng minh thiếu steps là nguyên nhân duy nhất. Frozen VLM/vision và LoRA r8 chỉ371328trainable parameters cũng là giới hạn cấu hình cần kiểm có đối chứng, không đủ để kết luận cứ tăng rank sẽ sửa được.

Tài liệu `README_TRAINING_DEMO_PROGRESS.md` ghi một run tiny56/LR1e-4/1000steps riêng cũngFAIL; checkpoint/log run đó không có trong ZIP được nhận nên audit không tái xác minh run tiny cũ. Việc tiếp tục xem checkpoint full-train là model chất lượng đạt khi tiny gate chưa PASS là lỗi tiêu chí nghiệm thu. Full-train checkpoint có thể chạy được về kỹ thuật nhưng chưa đạt học policy.

Một correction về baseline: uniform random bốn mức độc lập có expected exact-vector1/64=1,5625% và zone25%. Test exact1,81% trong tài liệu gần mức uniform random, không phải thấp hơn nó; nhưng thấp hơn rõ rệt baseline always-OFF28,89% do target phân bố lệch. Số test này vẫn là documented result từ máy bạn, không phải audit vừa chạy lại.

## 5. Bước debug có thể xác định nguyên nhân sâu hơn

Chưa bắt đầu các thí nghiệm này trong lượt audit:

1. Tạo run diagnostic mới với4–8record train dễ và đối lập, gồm STOP_ALL/WATER_ALL mức1/2/3. Dùng schema/stats gốc, kiểm nhiều seed và predictions sau sampling, không chỉ loss. Không ghi đè checkpoint6000.
2. Kiểm gradient/update theo từng nhóm action projections và expert q/v; so direct denoising error với free-running sampled actions. Chứng minh ghi nhớ một bài nhỏ trước khi đưa Vision/125tổ hợp vào.
3. Chỉ thay một yếu tố (LR/schedule/budget/capacity) mỗi run, dừng nếu không đạt memorization gate. Không tự coi tăng thêm6000steps hoặc rank là cách sửa đã được xác minh.
4. Khi tiny gate đạt mới chọn bằng validation; không tuning trên test. Không đổi labels/normalization, hardcode STOP/WATER_ALL hoặc chọn seed tốt để che lỗi model.

## 6. File model và deployment

Trọng số đã học:

```text
/home/hanhpham/agriVLA/agri-soybean-vla/outputs/train/step8-lora-6000-t1200-20261109-resume4500-v3/checkpoints/006000/pretrained_model/adapter_model.safetensors
```

Base weights cần đi cùng:

```text
agri-soybean-vla/.cache/models/lerobot/smolvla_base/5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6/model.safetensors
agri-soybean-vla/.cache/models/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/7b375e1b73b11138ff12fe22c8f2822d8fe03467/model.safetensors
```

Config/tokenizer/processors/stats đi cùng adapter trong pretrained_model; prepared bundle tại models/prepared/wording_v1. `initial_adapter` không phải model đã train. `training_state/` chứa resume state, không dùng để tính inference. Hướng dẫn chạyweb/CLI/API đầy đủ: **DEPLOY_STEP6000.md**. Demo trả JSON action và mức3vùng, không điều khiển phần cứng.

Raw receipts: `reports/model-audit/checkpoint-diagnosis.json`, `sampling-diagnosis.json`, `audit.log`, `sampling.log` và hai scripts diagnostic đi kèm. Mỗi report có `optimizer_steps_performed=0`, checkpoint hash không đổi. Dataset browser localhost8765 độc lập với model audit và vẫn chỉ đọc dữ liệu.
