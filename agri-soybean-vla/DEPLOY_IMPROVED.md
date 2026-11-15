# Dùng model cải thiện trên web — 2026-11-15

Đã tạm dừng thí nghiệm theo yêu cầu. Web hiện dùng model TRAIN60/LoRA expert mở rộng, LR1e-4, checkpoint300. Chưa phải model full dataset đạt chất lượng. Không có training chạy nền; chỉ web inference và dataset browser.

## Mở web và xem kết quả ngay

1. Mở http://127.0.0.1:8000. Nếu giữ tab cũ, Ctrl+Shift+R để tải JS/HTML mới.
2. Đợi “Model sẵn sàng”. Dòng “Checkpoint đang dùng” phải chứa cases60-r8-wide-b4-n2-lr1e4-resume100-v6/checkpoints/000300/pretrained_model.
3. Trong “Thử model mới với mẫu có đáp án”, chọn AUTO. Ba ảnh và câu được nạp sẵn, target[0,1,3].
4. Bấm “Chạy SmolVLA”. Xem ba thẻ van, continuous action, quantized action và dòng so target. AUTO hiện đã kiểm API thật ra[0,1,3].
5. Thử các mục WORST_ONLY, ZONE_ONLY, EXCLUDE_ZONE, CAP_LEVEL, STOP_ALL, WATER_ALL. Bảy mẫu VI này đã chạy đúng qua API. Mục cuối “MẪU CÒN SAI · EXCLUDE_ZONE · en” được giữ để thấy giới hạn: target[0,0,3], model ra[0,1,3].
6. Muốn thử tự do, upload đủ ba ảnh đúng thứ tự zone1/2/3 rồi nhập câu. Khi đổi ảnh/câu, web bỏ target cũ, không tự chấm ảnh mới bằng đáp án của preset. Chú thích héo chỉ để đối chiếu, không vào model.

| Preset | Target | Web API thực đo |
|---|---|---|
| AUTO |[0,1,3]|[0,1,3]|
| WORST_ONLY |[0,0,3]|[0,0,3]|
| ZONE_ONLY2 |[0,1,0]|[0,1,0]|
| EXCLUDE_ZONE2, VI |[0,0,3]|[0,0,3]|
| CAP_LEVEL1 |[0,1,1]|[0,1,1]|
| STOP_ALL |[0,0,0]|[0,0,0]|
| WATER_ALL2 |[2,2,2]|[2,2,2]|
| EXCLUDE_ZONE2, EN còn sai |[0,0,3]|[0,1,3]|

HTTP inference khoảng0,73s/lần sau warm-up; lần đầu1,71s trong phép kiểm. Đây là thời gian request local của7examples, không phải benchmark mọi ảnh hoặc tốc độ training.

Model chỉ nhận ba ảnh, instruction và state[0,0,0]. Backend request chỉ có instruction/images/wilt_notes; target preset không được gửi vào API inference, wilt_notes không đưa vào tensor model. Dòng ĐÚNG/SAI do frontend so output thật với target. Không hardcode STOP/WATER_ALL hoặc ghi đè prediction theo teacher. OFF0/LOW1/MEDIUM2/HIGH3 là mức tổng hợp, chưa điều khiển van thật.

## Nếu cần tự chạy lại server

Từ root repo, trong terminal có CUDA. Nếu web8000 đang hoạt động thì dùng luôn; không mở thêm model GPU thứhai. Dừng server bạn đang chạy bằng Ctrl+C trước khi restart.

```bash
cd /home/hanhpham/agriVLA
source agri-soybean-vla/reports/small-cases/selected.env
agri-soybean-vla/.venv-train/bin/python -u -m agri_vla.cli demo-web \
  --checkpoint "$AGRI_MODEL" --device cuda --port 8000
```

Giữ terminal chạy và mở http://127.0.0.1:8000. Không bỏ --checkpoint: mặc định cũ vẫn là checkpoint6000. Không sửa default để nhầm model khác trong các run lịch sử.

Kiểm tra backend bằng terminal khác:

```bash
curl -s http://127.0.0.1:8000/api/status
```

Cần thấy status=ready, device=cuda và đúng đường dẫn300. Endpoint /api/examples cung cấp8mẫu để browser nạp. Nếu danh sách trống, kiểm file reports/small-cases/demo-examples/web-examples.json; file này được tạo từ dataset thật và giữ ngoài Git.

Kiểm thêm bằng CLI, output mới mỗi lần:

```bash
bash agri-soybean-vla/reports/small-cases/demo-examples/run-auto.sh
```

Ảnh JPEG mẫu nằm tại reports/small-cases/demo-examples/zone1.jpg, zone2.jpg, zone3.jpg; câu và target nằm input.json. Dataset browser đầy đủ: http://127.0.0.1:8765/?view=samples&page_size=48. Nếu dừng, chạy `bash agri-soybean-vla/scripts/browse.sh --port 8765`.

## File model chính xác

```text
/home/hanhpham/agriVLA/agri-soybean-vla/outputs/small-cases/cases60-r8-wide-b4-n2-lr1e4-resume100-v6/checkpoints/000300/pretrained_model/adapter_model.safetensors
```

Cần cả thư mục pretrained_model (adapter_config/config/tokenizer/processors/stats/locks) và pinned base/backbone cache, không chỉ copy một file weights. Đây là adapter đã optimizer train thật. initial_adapter trong prepared bundle là chưa train. Checkpoint300 là step của phaseLR1e-4, khởi tạo weights từ wideLR3e-4/step600; không phải300steps từ base hoàn toàn mới. Native optimizer của phase này đã nối100→300, không reset khi resume.

## Đã cải thiện gì và còn thiếu gì

Giữ WORDING_V1, full-train stats, ba camera256, tokenizer96 và saved processor. Đã đối chiếu3600targets bằng số học và10800camera tensors qua native loader, không tìm thấy lỗi ghép ảnh/câu/nhãn. Chọn panel8/32/60 phủ7intents, VI/EN, đổi lệnh cùng ảnh, hoán vị ảnh, tie, dưới ngưỡng và override mức1/2/3. Panel32 là subset đánh giá của60, không phải một run32 độc lập.

Thử r8 batch1/2/4, full action/time heads, native Euler steps. Micro8 batch4/Euler2 đạt40/40predictions trên5noise seeds; full heads không giúp nên không chọn. Panel60 narrowLoRA r8 vẫn chỉ43,33%/55% ở seed0/42. Mở LoRA trong expert từ q/v lên q/k/v/o/gate/up/down (VLM/vision frozen),1.756.800trainable parameters, rồi hạLR3e-4→1e-4. Kết quả screen cuối80%/98,33%, zone92,22%/99,44%, STOP100%. Nhiều thay đổi và warm-start thêmupdates, chưa có đối chứng cô lập để quy toàn bộ gain cho riêng một yếu tố.

7/7presetsVI đúng không đồng nghĩa model đạt toàn60 hoặc generalization. Còn một câu EN sai ở seed42, và12/60mẫu sai ở seed0. Mẫu EN EXCLUDE_ZONE2 cho continuous vùng2≈0,508, vượt ranh giới0,5 nên round thành1 thay vì0: lỗi prediction/noise thật, không sửa bằng đổi target hoặc ép output.

Đã kiểm GPU native resume thật của wideLR3e-4: adapter/optimizer/RNG maxdifference0 với checkpoint liên tục. LR1e-4 resume101 cũng sameepisodes/RNG/loss. **Theo yêu cầu tạm dừng**, fresh-process full60/five-seed reload, longest-prompt backward của bản wide mới và proof99→100LR1e-4 chưa hoàn tất; chỉ model loading +8requests web đã chạy thật. Không gọi full-ready hoặc đảm bảo accuracy full tăng.

Chưa train toàn3600, chưa tuning VAL/TEST, không commit/push. Full training recipe và tài nguyên xem RUN_SMALL_MODEL_AND_FULL_TRAIN.md; recipe full đang có quality gate chưa đạt, không nên chạy chính thức ngay. Cơ sở tiếp theo là làm model ổn định theo noise/nhánh lệnh còn lỗi và validation trên ảnh/câu mới; chỉ tăng số steps không đảm bảo giải quyết được.
