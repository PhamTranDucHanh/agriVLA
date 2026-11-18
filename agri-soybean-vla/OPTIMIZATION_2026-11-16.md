# Dò cấu hình theo TRAIN → VAL, ngày 16/11/2026

**Kết luận mới nhất sau matched GPU K/V:** [LATE_VLM_GPU_2026-11-16.md](LATE_VLM_GPU_2026-11-16.md). Hai arm300updates: TRAIN28038,21% cảhai, meanVALID112 control8,57%/candidate8,75%; chưa đạt improvement hoặc full readiness. GPU/resume/freshreload/HTTPsameB1 đã kiểm thật, không full/TEST. Hướng dẫn lệnh hiện hành cho variant ở [FULL_LATE_VLM_GUIDE.md](FULL_LATE_VLM_GUIDE.md). Các “đang chạy” dưới đây là lịch sử; không suy LR3e-4step300 có accuracy nếu thiếu raw checkpoint/evaluation.

Đã kiểm đối chứng và augmentation tới300updates; LR3e-4 ở100updates tốt hơn LR1e-4 cùng100steps, đang kiểm LR3e-4 tới300. Branch thực tế: `feat/improve-accuracy` (giữ nguyên branch đang mở, không checkout/commit/push). Chưa có cấu hình đạt điều kiện full train; chưa chạy full mới. Không xem accuracy trên tập nhỏ là accuracy ngoài mẫu. Các số dưới đây thay cho trạng thái tạm dừng ngày 15/11; những báo cáo cũ được giữ lại làm lịch sử.

## Vì sao đổi cách kiểm

Endpoint LoRA trên 60 TRAIN gần như học thuộc: một lần dự đoán đúng 299/300 lượt kiểm, gồm 60 mẫu × 5 noise seeds (0,1,2,3,42). Lấy trung bình hai lần dự đoán độc lập (MC2) đạt 300/300; fresh process, ảnh gốc và native `select_action` khớp reference, max error 0. Tuy nhiên, trên cùng VAL112 cân bằng, single chỉ đúng 7/112 (6,25%), MC2 chỉ 6/112 (5,36%), STOP đều 0/16. Always-OFF đạt 36/112 (32,14%). MC2 mất khoảng 1,24 giây/mẫu, gần gấp đôi single. Vì vậy không chọn MC2 để full train hoặc quảng bá nó là model tốt.

Đây là khoảng cách khái quát lớn. Sai số làm tròn một mẫu TRAIN không còn là mục tiêu đáng tối ưu. Không tiếp tục SGD trên 60 mẫu để săn 100%. Chưa chứng minh được một nguyên nhân duy nhất cho khoảng cách này.

## Dữ liệu và kỹ thuật

- Dataset gốc **WORDING_V1**, `data/lerobot/wording_v1/full/{train,val,test}`: 3600/720/720. Giữ nguyên split/ảnh/target/manifest lock. Tên `training_v0.yaml` và repo_id `soybean_v0_full_train` là tên lịch sử; root,5040records toàn bộ, QA và các hash đều là WORDING_V1, không nạp data v0.
- Tập nhỏ mới lấy **280 record chỉ từ TRAIN**, gồm 140 cặp VI/EN, 40 record mỗi intent, 115 scene, 164/168 câu TRAIN khác nhau. Giữ các trường hợp đối lập cũ: cùng ảnh STOP và WATER_ALL mức1/2/3; AUTO/ngưỡng; WORST/tie; ZONE1/2/3; CAP1/2; EXCLUDE1/2/3; hoán vị vùng. Tham số CAP/zone/level được bao phủ; tần suất từng tham số chưa hoàn toàn đều.
- VAL112 cố định: 16 record mỗi intent, 56 VI/56 EN và 56 seen/56 unseen wording. Selector v2 round-robin theo intent → wording → target, giữ cặp dịch. Không đổi panel theo kết quả. Đây là partial VAL, khác phân bố toàn VAL720.
- Model vẫn nhận ba camera RGB256, câu lệnh, state[0,0,0]; output ba giá trị liên tục rồi clip0..3, `floor(x+.5)`. Teacher/intent/target không là input inference.
- Saved processor/tokenizer96, STATE IDENTITY và ACTION MEAN_STD từ full TRAIN. Stats SHA `b06849340aa520815152fd63921b3602ab9278f7c3fd242e600c6a33a086c78f`.
- Wide expert LoRA r8/alpha8: q/k/v/o/gate/up/down và các projection nhỏ, 1.756.800 trainable parameters; VLM/vision frozen. Batch4, LR1e-4, warmup20 rồi constant, AMP FP16 và FP32 master weights; không có accumulation/4-bit tự chế.
- Endpoint là **thí nghiệm đổi objective có version**, không phải mặc định SmolVLA: native `x_t=t*noise+(1-t)*action`, target velocity=`noise-action`. Khi t=1, một Euler step trả `noise-v`; loss velocity đúng bằng loss bình phương của action đó với target normalized. Điều này khớp bài toán singleton xác định, giảm lệch giữa loss và một-step inference; chưa chứng minh khái quát tốt hơn flow-matching gốc.

Config v7: endpoint; v8: MC2 (không chọn); v9: mở rộng pilot280, quay về single inference; v10: cùng profile thêm `override_image_mix_v1` cho TRAIN. Không sửa config/receipt cũ. Pilot dùng weights của endpoint600, **reset optimizer** khi chuyển60→280. Resume100→300 giữ native optimizer/scheduler/RNG/step, AMP và thứ tự loader. Không lấy số step của curriculum cộng vào rồi gọi đó là epoch full train.

Đã kiểm CPU tensor thật: TRAIN280 và VAL112, tổng1176ảnh, task/action/state đúng metadata, ảnh khớp transform từ ảnh gốc với max pixel error0. Saved model loader cũng kiểm full-train stats. Báo cáo: `reports/optimization-20261116/native-pilot-inputs.json`.

## Kết quả pilot

| Cấu hình | TRAIN exact / zone | VAL112 exact / zone | STOP VAL |
|---|---:|---:|---:|
|60mẫu endpoint600, single, seed42|100% /100% trên60|6,25% /30,95%|0/16|
|Cùng weights, MC2, seed42|100% /100% trên60|5,36% /29,46%|0/16|
|280mẫu, LR1e-4,100updates, seed42|27,14% /47,50% trên280|6,25% /28,27%|0/16|
|280mẫu, LR1e-4,300updates, seed42|38,21% /61,67% trên280|11,61% /36,31%|3/16|
|Cùng init/config100updates, thêm override-image-mix v1|23,57% /44,76% trên280|5,36% /27,68%|0/16|
|Override-image-mix v1,300updates, seed42|33,57% /57,98% trên280|13,39% /37,50%|3/16|
|Không augmentation, LR3e-4,100updates, seed42|28,57% /55,83% trên280|10,71% /38,99%|5/16|
|Không augmentation, LR3e-4,300updates|Đang kiểm|Đang kiểm|Đang kiểm|
|Always-OFF trên VAL112|Không phải model học|32,14% /63,10%|16/16|

TRAIN280 mới thấp nghĩa vẫn thiếu fit ở phạm vi này; đồng thời checkpoint từ60 đã bộc lộ học thuộc. Hai hiện tượng có thể cùng tồn tại. Lượt LR3e-4 giữ nguyên mọi yếu tố khác so LR1e-4, không augmentation. Ở100steps, exactVAL12/112 so7/112, zone38,99% so28,27%, STOP5/16 so0/16; false-watering142/212 so186/212. Tiếp tục tới300 để so cùng ngân sách; chưa coi100steps đã đủ fit hoặc readyfull.

Nguồn: `reports/optimization-20261116/deploy-reload-v2.json`, `pilot280-100-evaluation.json`, `pilot280-300-evaluation.json`; native train receipts ở `reports/small-cases/pilot280-*/`. Bản `deploy-reload.json` đầu tiên ghi nhận lỗi hook MC2 không đi qua `select_action`; đã sửa hook chung và kiểm lại bằng bản v2. Không xóa báo cáo lỗi đó.

## Một yếu tố tiếp theo: phá tương quan ảnh–lệnh ở override

TRAIN có1440scene;1336scene chỉ một case lệnh. Điều này cho phép đường tắt ảnh→target nhưng chưa chứng minh đó là nguyên nhân duy nhất. `train-scene-structure.json` ghi thống kê và kiểm đủ900STOP/WATER_ALL TRAINrecords đều thỏa target độc lập với ảnh.

Thí nghiệm v10 chỉ đổi ảnh ở STOP/WATER_ALL: mỗi hàng đó nhận **cả triplet ba camera từ một hàng TRAIN khác trong chính batch4**, task/state/action giữ nguyên. STOP luôn000, WATER_ALLlevel luôn[level]*3 bất kể héo, nên phép ghép này đúng policy. Năm intent phụ thuộc ảnh không đổi. Donor xác định bằng hash(step,episode), không rút thêm RNG; log lưu từng cặp episode và pixel có thực sự đổi không. Không thêm intent vào features. Inference/web không trộn ảnh hoặc sửa prediction theo luật.

So sánh công bằng với pilot100 v9: cùng init endpoint600, cùng TRAIN280/batch4/seed42/optimizer mới/warmup20/LR1e-4, chỉ thêm augmentation. Đánh giá vẫn ảnh gốc của TRAIN280 và đúng VAL112 cũ. Mốc100 chưa cải thiện (6/112 so7/112); mốc300 đạt15/112 so13/112, STOP vẫn3/16. Chênh hai mẫu chưa đủ kết luận model tốt hơn, cả hai còn kém baseline; không chọn augmentation cho full chỉ từ số này. Raw data/stats/lock không đổi; recipe/version transform được lưu trong checkpoint. Đối chứng và augmentation100 có cùng thứ tự episode cả100updates, cùng RNG ở các bước được ghi1/99/100/101/300 trong cả300updates; receipts `mix-control-fairness.json` và `mix300-control-fairness.json`. Code: `src/agri_vla/augmentation.py`, hook ở small/native-full trainer và resume proof. Không tự động chọn nó cho full run.

## Chốt trước một run dài

`configs/readiness_balanced_v1.json` được khai báo **trước khi đọc VAL của pilot280**. Nó yêu cầu TRAIN60 từng seed exact≥95%, zone≥98%, STOP100%; VAL112 exact cao hơn OFF ít nhất5 điểm phần trăm, zone cao hơn ít nhất2 điểm, STOP100%. Đây là mức sàng lọc trước train dài, không phải mục tiêu chất lượng cuối. Gate strict100% của các config cũ vẫn giữ nguyên; muốn dùng balanced phải truyền policy rõ ràng. Không hạ ngưỡng sau khi biết kết quả.

`candidate_train.py`/`full_run.py` nay bắt buộc có bằng chứng VAL đúng weights, đúng selector, đúng sample/target/seed và tính lại metrics từ predictions; summary tự ghi “pass” không mở được gate. Chưa có `configs/full_ready.json` hợp lệ, vì vậy full-run đang khóa. Chưa có full100 GPU smoke cho cấu hình cuối hoặc chứng nhận full-run mới.

Workflow chuẩn bị sẵn: `scripts/full_run.sh`, default chỉ in plan; khi có recipe đạt gate, một lệnh `bash agri-soybean-vla/scripts/full_run.sh --execute --steps 6000 --max-new-gib 1` sẽ train theo phase, đánh giá VAL720 để chọn best và dừng sớm nếu3phase không cải thiện; chỉ sau khi chọn xong mới TEST720 một protocol, copy `final_model/`, kiểm HTTP web thật và tạo `deploy.sh`. Chỉ giữ best+latest checkpoint thuộc chính run đó; không xóa model/source cũ. Orchestration có unit tests, **chưa được xem là full GPU end-to-end proof**.

Không dùng TEST để dò ở lượt này. Nếu nhìn TEST rồi quay lại chỉnh, nó thành dữ liệu phát triển; muốn kết luận ngoài mẫu phải có holdout khác. VAL hiện còn đủ để chẩn đoán, nên chưa cần làm vậy.

## Bằng chứng kỹ thuật mới (khác với chất lượng)

Trên profile v10 thật: checkpoint99→fresh process→native resume một update tới100 so nhánh liên tục100 **PASS max error0**:234adapter/702optimizer/8RNG tensors, sample order/loss và scheduler/step/AMP đều khớp. Bước này có trộn ảnh thực, không phải fixture. `reports/small-cases/resume-mix99to100-20261116/resume_report.json`.

`reports/optimization-20261116/mix-memory.json`: backward4prompt TRAIN dài nhất74tokens, thêm backward batch có STOP augmentation +3prompt74; đủ3encoder inputs[4,3,256,256], finite gradients và weights không đổi (0optimizer updates). Torch peak allocated1886,65MiB/reserved1896MiB. Sau đó inference đúng prompt VAL dài nhất81tokens, allocated955,57MiB, vẫn tokenizer capacity96 và đủ3camera. NVML1299MiB cuối probe là **snapshot sau inference**, không phải peak training; lấy peak toàn GPU từ log train1Hz. Không kiểm mới TEST.

## Tự kiểm lại

Từ root repo, dừng demo GPU bằng Ctrl+C trước khi chạy một job GPU khác. Dùng output mới để không ghi đè raw receipts:

```bash
export HF_DATASETS_CACHE="$PWD/agri-soybean-vla/.cache/hf-datasets"
AGRI_PY="$PWD/agri-soybean-vla/.venv-train/bin/python"
AGRI_TAG=$(date +%Y%m%d-%H%M%S)
AGRI_MODEL="$PWD/agri-soybean-vla/outputs/small-cases/pilot280-resume100-20261116/checkpoints/000300/pretrained_model"

# Đối chiếu pixel/task/action từ native data với ảnh/record gốc, CPU:
"$AGRI_PY" -m agri_vla.pilot --audit-inputs \
  --output "agri-soybean-vla/reports/optimization-20261116/user-input-audit-$AGRI_TAG.json"

# Frozen model, TRAIN280 + VAL112; không train và không TEST:
"$AGRI_PY" -u -m agri_vla.pilot --checkpoint "$AGRI_MODEL" \
  --output "agri-soybean-vla/reports/optimization-20261116/user-pilot-$AGRI_TAG.json"

# Xem trực quan checkpoint chẩn đoán trên web:
"$AGRI_PY" -m agri_vla.cli demo-web --checkpoint "$AGRI_MODEL" --device cuda --port 8000
```

Mở http://127.0.0.1:8000. Web nhận ba ảnh theo thứ tự zone1/2/3 và câu lệnh, trả ba mức tưới. Web chạy được không đồng nghĩa dự đoán đúng. `selected.env` cũ chưa được đổi sang model chưa đạt VAL. File weights của model trong lệnh là `$AGRI_MODEL/adapter_model.safetensors`; phải giữ cả thư mục processor/tokenizer/config/locks và base/backbone cache đã pin.

## Chi phí, đĩa và giới hạn

Pilot100 đối chứng đo update mean2,787s. Lượt augmentation200updates tiếp theo: mean2,789s, dataload≈0,169s/update; wholephase680,84s gồm setup/screen/save. Torch updatepeak allocated1891,83MiB/reserved1916MiB; tổng GPU NVML peak quan sát2238/4096MiB, còn1858MiB ở thời điểm đó. ProcessHWM3,66GiB, RSS2,93GiB, swap0; full3600 chưa đo RAM. NVML lấy1Hz có thể bỏ qua spike ngắn; Torch peak reset mỗi update. Ứng dụng khác có thể làm headroom giảm. Tập nhiều record hơn không nhân VRAM theo kích thước dataset nếu batch/input giữ nguyên; vẫn cần full smoke với loader3600 trước run dài.

Với batch4:100updates280≈1,43epoch;300≈4,29epoch;6000updates3600≈6,67epoch (900steps/epoch). Update thuần6000×2,787s≈4,65giờ. VAL112 hiện≈0,695s/mẫu, tức VAL720≈8,3phút;6lượt VAL+1TEST khoảng58phút, cộng init/data/save/thermal load. **Dự trù6–8giờ cho6000**, suy từ pilot, chưa phải thời gian full đã đo. Một kiểm TRAIN280+VAL112 hiện≈4phút ngoài khởi tạo;100updates nhỏ cộng kiểm thường≈10–13phút.

Không export lại ảnh/dataset, không tải lại base, dùng chung `.cache/hf-datasets`. Một native wide checkpoint khoảng24MiB (adapter≈6,8MiB, còn lại tokenizer/processors/optimizer/RNG). Bản deploy chỉ copy `pretrained_model`, dùng lại base cache; không gộp lại0,45B weights thành bản sao lớn.

Runner có budget mới mặc định1GiB tính output/report và tăng trưởng HF dataset cache, giữ best+latest, không nén archive/backup mới. Chốt dung lượng/thời gian full vẫn cần smoke thật. Budget không giới hạn Windows/WSL swap, VHDX hoặc phần mềm khác. `df` trong WSL báo899GiB trống là dung lượng ảo; lúc kiểm ổ Windows C còn≈79GiB, D≈43GiB. Các file đã xóa không bảo đảm VHDX tự co. Không tự xóa backup/model cũ.
