# Kết quả GPU: LoRA K/V block15, chưa đủ điều kiện full train

Branch giữ nguyên `feat/improve-accuracy`. Hoàn tất hai nhánh ×300 optimizer updates,
hai proof resume ×1 update riêng và frozen evaluation. **Không full train, không
TEST, không commit/push.** Model deploy cũ và mọi thay đổi có trước được giữ nguyên.

**Kết luận:** K/V học và chạy được trên GPU4GiB ở workload pilot đã đo, nhưng
không cải thiện VALID đủ ý nghĩa. Không chọn candidate để full train; không tạo
`configs/late_vlm_full_ready.json`, không hạ readiness gate.

## Đã làm gì và vì sao

Tiếp tục đúng một thay đổi từ Prompt4: thêm r8/alpha8 LoRA vào `k_proj`, `v_proj`
của VLM block15 thực sự được SmolVLA dùng. Thêm20.480 trainable parameters;
control1.756.800, candidate1.777.280. Base/vision vẫn frozen. Giữ endpoint loss,
native Euler1, batch4/LR1e-4, warmup20/constant, tokenizer96, ba RGB256 cameras,
saved processor và full-TRAIN normalization. Không thay head, thêm loss khác,
đổi split/nhãn/ảnh hoặc train counterfactual đồng thời.

Hai nhánh cùng copy234 expert adapter tensors từ endpoint60 checkpoint600
(source SHA `083aac7c127bc431eb34c761255730973f871097fe36a7383677e775405d5147`),
cùng optimizer mới và RNG barrier42. Candidate thêm4 tensors, B khởi tạo0.
Source warm là reference cố định để đối chứng, không phải init đã thắng Prompt3.
Kiểm receipts xác nhận **cùng episodes và RNG ở đủ300updates**, không tái dùng
control lịch sử khi chưa chứng minh điều kiện khớp.

Protocol/hashes/panels/acceptance được chốt trước GPU và VALID mới:
`reports/latekv-gpu-20261116/{protocol.json,validation-screen-protocol.json}`.
Smoke2→native resume100→native resume300 giữ tổng300updates mỗi nhánh.
Không cộng các mốc thành402. Hai proof99→100 dùng output riêng, không thay
weights main run. Không thêm thí nghiệm train sau khi thấy VALID thấp.

## Dữ liệu và độ phủ

Original WORDING_V1 giữ nguyên3600/720/720. Pilot chỉ lấy TRAIN280:115scenes,
287ảnh nguồn/8groups,40records mỗi intent, VI140/EN140. Có đủ STOP/AUTO/WORST,
ZONE1/2/3, EXCLUDE1/2/3, CAP1/2, WATER_ALL1/2/3. CAP1/2 là14/26records;
WATER_ALL1/2/3 là16/12/12: cân bằng intent không có nghĩa mọi tham số đều bằng nhau.
Tính lại280targets bằng `revision.revision_action` đúng WATER_ALL override:
**0mismatch**. Không dùng policyv0 thiếu override.

VALID112 cố định:25scenes,60ảnh/1sourcegroup khác TRAIN,16records mỗi intent,
VI56/EN56. Năm noise seeds0/1/2/3/42 vẫn là cùng112mẫu, không phải560mẫu held-out
độc lập. VALID là panel phát triển đã được xem nhiều lần; chưa đo full VAL720
hoặc TEST trong lượt này. Xem `coverage.json`, `train280-parameter-coverage.json`,
`train280-policy-recheck.json` trong report directory.

## Kết quả thật

TRAIN đo seed42/B4; VALID đo B1, cùng native API/processor và noise protocol.
Exact yêu cầu đúng cả ba vùng. Các tỷ lệ mean là trung bình năm noise seeds.

| Metric | Control expert-wide | Candidate thêm K/V15 |
|---|---:|---:|
|VALID112 exact ở100, seed42|7/112=6,25%|7/112=6,25%|
|TRAIN280 exact ở300, seed42|107/280=38,21%|107/280=38,21%|
|Old TRAIN60 exact, seed42|55/60=91,67%|55/60=91,67%|
|New TRAIN220 exact, seed42|52/220=23,64%|52/220=23,64%|
|VALID112 exact ở300, seed42|13/112=11,61%|13/112=11,61%|
|VALID mean exact / macro-intent|8,57%|8,75%|
|VALID mean zone accuracy|35,36%|35,48%|
|VALID STOP exact|16,25%|16,25%|
|False-watering / target-OFF components|71,79%|71,70%|
|VALID continuous MAE|0,85516|0,85430|

Exact counts theo seeds0/1/2/3/42: control8/12/11/4/13; candidate8/12/11/5/13,
mỗi số trên112. Candidate chỉ thêm **một lần đúng trong560lượt**, mean tăng
**0,18điểm phần trăm**, zone tăng0,12điểm%. Không coi đó là cải thiện đáng tin
để chi ngân sách full. Candidate chỉ thắng exact1/5seeds. Paired scene bootstrap
10000draws,25clusters: khoảng95% của chênh mean exact **[0;+0,64]điểm%**;
lower bound không dương. Các scene còn dùng chung ảnh nên khoảng này chỉ có
ý nghĩa exploratory/conditional, không đo uncertainty trên source độc lập hay
training seeds. Chỉ training seed42 được chạy.

Tiêu chí đã chốt: mean exact≥+5điểm%, zone≥+2điểm%, thắng≥4/5noise seeds,
scene interval lower>0, không regress STOP/false-watering/new220 và old60 drop≤10điểm%.
Candidate giữ được các guard sau nhưng **trượt bốn tiêu chí improvement đầu**.
Raw quyết định: `decision.json`, không thay threshold sau kết quả.

Trên cùng VALID112, always-OFF đạt32,14%exact/63,10%zone. Gate full hiện hành
còn yêu cầu seed42 hơn OFF ít nhất5/2điểm% và STOP100%; candidate không đạt cả
ba kiểm này. Không dùng rule STOP/WATER_ALL ghi đè output để tính lại điểm.

Candidate VALID mean theo intent: AUTO0%, EXCLUDE0%, CAP11,25%, STOP16,25%,
WATER_ALL27,50%, WORST3,75%, ZONE2,50%. VI6,43%/EN11,07%; seen12,50%/unseen5%.
Giá trị từng mẫu, ảnh, câu, target, continuous và quantized predictions:
`validation-examples.csv`; `control-final.json`, `candidate-final.json` giữ raw
TRAIN280 và toàn bộ năm noise seeds. Learning curve loss mỗi20updates và mốc100/300
có trong `decision.json`/raw updates. Không dùng loss giảm thay cho accuracy.

## Điều gì được chứng minh, điều gì chưa

- Forward/backward/optimizer update thật: loss/gradient finite, không AMP skip,
 saved adapter/Adam tensors đều finite; optimizer steps đều300. Control234adapter/
 702optimizer tensors, candidate238/714.
- Candidate K/V.B có gradient finite/nonzero ở **đủ300updates**. Last phase
 adapter squared-delta3,79276, trong đó K/V0,11531. Thay đổi có đi vào action;
 không phải thêm module rồi không được gọi.
- Native99→100 trong process mới khôi phục optimizer/scheduler/RNG/step/AMP và
 prepared-loader continuation; cùng batch/RNG/loss với nhánh liên tục.
 Sau một update, adapter/optimizer/RNG **maxerror0**, scheduler/step/AMPexact,
 ở cả hai nhánh. Tensor tolerance1e-6 absolute/1e-5 relative; loss1e-5/1e-4.
- Trước save300→fresh-process reload300→inference TRAIN60 cùng B4/seed42:
 maxcontinuouserror0 cảhai, quantized exact; declared atol/rtol1e-3.
 CPU scope fixture ởPrompt4 không được dùng thay trained resume proof.
- Input allowlist vẫn ba camera+câu+zerostate, không teacher/intent/target.
 WORDING_V1 locks, saved tokenizer/processor/fullTRAINstats và frozen code hashes
 không thay đổi. Static metadata conservative flags không được sửa để thay cho
 raw GPU receipts.

**Diễn giải:** model giữ khả năng trên nhóm60 cũ khá cao, nhưng chưa fit220mẫu
TRAIN mới; đây không chỉ là overfit của một model đã fit toàn TRAIN. STOP cũng
yếu dù không cần đọc mức héo, nên lỗi không chỉ do thị giác. K/V15 ít tham số
đã được học, nhưng thay đổi này chưa giải quyết đủ conditioning/decoding dưới
budget hiện tại. Chưa chứng minh representation đủ, chưa chứng minh native
flow head là nguyên nhân duy nhất, chưa chứng minh K/V không bao giờ có ích.

Bước hợp lý tiếp theo là frozen probe đọc intent/parameters và wilt từ representation
trước khi chọn thêm kỹ thuật. Nếu intent/parameters chưa đọc được, đối chứng
supervision phụ cho intent/parameters trước; nếu representation đọc được tốt nhưng
action vẫn sai/noisy, mới đủ căn cứ thử head xác định theo nhánhB. Đây là đề xuất,
**chưa triển khai/train thêm trong lượt này**. Không tăng rank/LR/steps đồng loạt.

## GPU, RAM, thời gian, disk

GPU thực NVIDIA T1200 Laptop4096MiB, driver610.60.

| Workload | Torch allocated peak | Torch reserved peak | NVML total peak quan sát |
|---|---:|---:|---:|
|Control300, B4|1880,39MiB|1916MiB|2300/4096MiB|
|Candidate300, B4|1883,04MiB|1896MiB|2281/4096MiB|
|Candidate longest TRAIN, native Adam loaded|1869,72MiB|1896MiB|không đo peak riêng|

Longest full-TRAIN prompt74tokens, B4 dùng bốn prompt74tokens; ba camera đều
4×3×256×256 vào vision encoder. Longest VAL prompt81tokens đã inference thật;
tokenizer capacity96. Longest probe backwardfinite, weights giữ nguyên,0updates.
NVML1352MiB cuối probe là snapshot sau inference, **không phải backward peak**.
Training telemetry1Hz cho peak quan sát; có thể bỏ spike ngắn. Candidate headroom
quan sát1815MiB, control1796MiB. Reserved thấp hơn ở candidate không chứng minh
tiết kiệm memory: allocator/cache khác nhau. Không dùng số này hứa full4GB đã pass.

Candidate mean update2,94565s, data0,18734s → **3,13299s/update cộng data**.
Control update2,92942s. Đây là run tuần tự trên laptop, không benchmark thermal
được kiểm soát. Process RAM HWM≈3,68GiB; candidate có swap snapshot tới≈132MiB.
WSL có≈7,6GiB RAM/2GiBswap. Full RAM chưa đo; không lấy VmPeak làm RAM thật.

Original full3600/batch4 =900updates/epoch. Pilot280×300updates≈4,29epochs;
full300 chỉ0,33epoch;≈3857fullupdates mới bằng số lượt qua dữ liệu của pilot.
6000fullupdates≈6,67epochs; phần update+data ngoại suy **5,22giờ**. Một VAL720
B1 ước≈8,68phút từ VAL112 thật;6VAL+TESTcuối thêm≈61phút. Có startup/save/cache/
thermal: **dự trù6–8giờ, để dư lịch8–10giờ**, chưa đo full và không dự đoán accuracy.
Train full có1258ảnh/1440scenes so287/115 ởpilot, nhưng vẫn cùng8TRAINgroups;
VALID thuộc group khác. Intent distribution full cũng khác pilot cân bằng.
Thêm dữ liệu có thể giúp hoặc làm lộ underfit/domain gap; phải xem full VALID.

Không download, export/copy ảnh hay tạo ZIP mới. Candidate weights6,81MiB;
cả deploy folder10,27MiB; checkpoint đủ training state23,91MiB.
Đã hash/tombstone rồi dọn **chỉ intermediate checkpoints/restore duplicates của
lượt gpu20261116**, giải phóng271,75MiB. Giữ control300/candidate300 là best+latest
theo VALIDseed42 trong cohort; giữ toàn bộ reports. Artifact còn≈60MiB,
HFdatasetcache tăng0byte, dưới budget1GiB. Giữ toàn bộ model cũ.

Full tương lai dùng chung cache, best+latest≈48MiB và final_model≈10MiB; với logs/
reports/temp dự trù0,1–0,3GiB logical nếu cache không tăng. Budgetguard1GiB có
điểm kiểm, không là hard quota VHDX. Để dư2–5GiB ổ thật cho cache/temp/swap.
C: lúc kiểm còn≈80GiB, D:≈43GiB; không lấy900GiB WSL free làm dung lượng Windows.

## Web, model và lệnh full

Model candidate **đã train nhưng chưa đủ chất lượng**, không thay selected.env:

```
outputs/small-cases/latekv-gpu20261116-candidate-s300/checkpoints/000300/pretrained_model/
  adapter_model.safetensors
```

Adapter SHA `3ee0c612d6385a9565435f1a1aa6d6c4df67b61bf63d955f559f32d9b0820d13`.
Giữ toàn bộ pretrained_model và shared pinned base/backbone, không dùng initial_adapter.

HTTP smoke thật: webtypedloader→ba ảnhTRAIN+câu→`[0,1,3]`, đúng target mẫu.
Input không gửi target/teacher. So HTTPB1 với frozen benchmarkB4 ban đầu có
maxcontinuouserror0,003305 > absolute1e-3, quantized giống nhau; **giữ receipt
mismatch**, không nới tolerance. Kiểm bổ sung nativeB1 cùngseed42: cả ba pixel
tensors khớp0, continuous với HTTP khớp0, quantizedexact. Khác biệt B1/B4 có thể
làm đổi output gần ngưỡng; không suy một mẫu web đúng thành general accuracy.
Web server của phép kiểm đã dừng. Receipts `web-check/`.

Đã bổ sung typed bridge `late_vlm_full.py` và `scripts/full_late_vlm.sh`, sửa route
web thực sự gọi `evaluation.load_model`, và sửa Path conversion khi full runner
đọc evidence lúc resume. Không sửa vendor.24testsPASS; CPU resolve full config
đúng TRAIN3600/episodes=null/batch4/LR1e-4/tokenizer96. Tests/CPU plan **không phải
full GPU end-to-end proof**. Lệnh execute được kiểm là dừng trước training khi
thiếu qualified recipe.

Xem [FULL_LATE_VLM_GUIDE.md](FULL_LATE_VLM_GUIDE.md): lệnh mở web, dataset browser,
plan và một lệnh full có gate, dataset được lấy từ đâu, từng output/save/resume/
selection/deploy có nghĩa gì. `qualify` chỉ xuất recipe nếu relative comparison,
TRAIN60/B1/five-seed/fresh reload, fixedVALID, memory và trainedresume đều đạt.
Hiện không thể xuất recipe vì VALID fail. Không tự điền JSON để mở khóa.

## Tái lập lượt nhỏ

Chọn TAG mới, không chạy đè `gpu20261116`. Script và cách chạy memory/resume/five-seed
được mô tả trong [LATE_VLM_KV_V1.md](LATE_VLM_KV_V1.md). Thứ tự:

```bash
bash agri-soybean-vla/scripts/late_vlm_v1.sh smoke NEW_TAG
bash agri-soybean-vla/scripts/late_vlm_v1.sh 100 NEW_TAG
# Longest-prompt memory + genuine99→100 proof cảhai; frozenVALID112seed42.
bash agri-soybean-vla/scripts/late_vlm_v1.sh 300 NEW_TAG
# late_vlm_evaluate cảhai: --seeds 0 1 2 3 42
```

Receipts gốc ở `reports/latekv-gpu-20261116/` và
`reports/small-cases/latekv-gpu20261116-*`.
Checkpoint2/99/100/proof duplicates đã dọn sau kiểm; hashes/metadata và raw
reports còn trong `checkpoint-retention.json`. Muốn replay one-updateproof từ
checkpoint đã dọn phải tái lập cohort ởTAG mới, không coi fixture là proof.
`smoke_report.json`, `decision.json`, `tests-final.xml` ghi kết luận máy đọc được.
