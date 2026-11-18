# Full TRAIN theo yêu cầu người dùng — 16/11/2026

**Sửa lỗi report sau phase100:** full-control-research-run1 đã hoàn tất100updates
và save checkpoint99/100; sau đó gặp `DataProcessorPipeline is not JSON serializable`
khi ghi result.json. Đã loại pre/post runtime khỏi JSON, giữ nguyên training math.
Native GPU proof riêng99→100 khớp234adapter/702optimizer/8RNG tensors maxerror0;
fresh GPU reload/inference checkpoint100PASS;19testsPASS. Plan đã backup/migrate
đúng một hash của bản sửa, không thay checkpoint metadata/states/hyperparameters.
Raw result.json hỏng được giữ làm evidence. Receipts: reports/full-report-fix-20261116/.
Tiếp tục **cùng output** bằng lệnh bên dưới và thêm `--resume`; runner sẽ bỏ qua
train100 đã xong, chạy `val-000100`, sau đó native100→1000. Không xóa output
hoặc reset optimizer. Full source hash cũ chỉ được chấp nhận nếu đúng predecessor
đã audit và bản runner-source.py lưu tại phase khớp hash đó; mọi drift khác vẫn bị chặn.

Người dùng chọn tự chạy full trong terminal sau khi đã xem kết quả VALID còn thấp.
Route này là **thí nghiệm chưa đủ chất lượng**, không xuất qualified recipe hoặc hạ
readiness thresholds. Standard `full_late_vlm.sh` vẫn giữ quality gates. Không chạy
training/GPU/TEST khi chuẩn bị lệnh; đã kiểm CPU plan/native config và16tests.

```bash
cd /home/hanhpham/agriVLA
bash agri-soybean-vla/scripts/full_research.sh \
  --output agri-soybean-vla/outputs/train/full-control-research-run1 \
  --steps 6000 --patience 3 --max-new-gib 1 \
  --research-run --execute
```

Dùng control300 expert-wide endpoint vì K/V chưa chứng minh lợi ích. Source:
`outputs/small-cases/latekv-gpu20261116-control-s300/checkpoints/000300/pretrained_model`.
Copy weights, reset optimizer/scheduler/RNG cho full TRAIN; không resume từ
checkpoint6000 cũ. Batch4/LR1e-4/warmup20constant/r8alpha8/Euler1, frozen vision/VLM.
Saved processor/tokenizer96/full-TRAIN stats và ba camera256 giữ nguyên.

Input lấy từ `data/lerobot/wording_v1/full/train/` đủ3600records, episodes=null.
Mỗi mẫu gồm ba ảnh, instruction VI/EN, state000, target ba mức0..3. Metadata trong
`sample_metadata.jsonl` và ảnh gốc dưới `data/raw/zenodo_8256382/images/`.
Không dùng counterfactual manifest hoặc teacher metadata làm inference features.

Trước train kiểm longest-prompt memory0updates; sau đó100→1000→2000… tối đa6000
updates. Mỗi mốc đo VAL720, giữ best+latest, dừng khi không cải thiện3mốc.
**Không mở TEST**. Giữ `research_run=true`, `full_execution_gate_passed=false` trong
native plan, `readiness_gate_passed=false` trong final report. Cờ research phải được
truyền rõ; checkpoints research chỉ resume trong cùng chế độ research.

Output trong thư mục RUN đã chỉ định:

- `*/checkpoints/NNNNNN/pretrained_model/`: adapter_model.safetensors đã train,
  tokenizer/processors/config/stats/typed metadata; training_state ở thư mục cạnh
  chứa native optimizer/scheduler/RNG/step/AMP.
- `val-NNNNNN/{metrics.json,predictions.jsonl}`: fullVALID720, seed42/B1.
- `progress.json`: latest/best/learning history và trạng thái gián đoạn.
- `final_model/`: copy model best theo VALID, không mặc định latest.
- `final-report.json`: bestVALID metrics và trạng thái research chưa qualified.
- `deploy.sh`: mở web loopback8000 bằng typed loader, dùng cache base/backbone cũ.
- `logs/`: train/memory/VAL logs; thêm receipts trong `reports/full-candidate/PHASE`.

Nếu gián đoạn, chạy lại nguyên lệnh và thêm `--resume`; cùng output,steps,patience,
budget và source code. Runner khôi phục latest complete checkpoint của phase dở.
Không sửa code/config trong lúc run. Không chạy hai instance trên cùng output.

Sau hoàn tất:

```bash
bash agri-soybean-vla/outputs/train/full-control-research-run1/deploy.sh
```

Mở http://127.0.0.1:8000. Input ba ảnh+câu+zerostate, output continuous và ba mức
OFF/LOW/MEDIUM/HIGH; chưa gửi lệnh tới phần cứng. Không giả định VAL tốt hơn
pilot, không lấy software tests hoặc web chạy được làm accuracy.

Dự trù6000updates≈6,67epochs,6–8giờ gồmVAL, để dư8–10giờ; chưa đo full workload.
Pilot peak GPU≈2,3GiB/4GiB và RAMprocess≈3,7GiB. Full còn cần smoke thật trong
chính lệnh. Dùng chung cache, best+latest khoảng48MiB, final_model≈10MiB;
budgetartifact1GiB kiểm giữa phase, không là hard quotaVHDX. Dư2–5GiB ổ Windows
cho cache/temp/swap. Dừng GPU demo của mình bằngCtrl+C trước chạy; không kill
process lạ hoặc bỏcamera khi OOM.
