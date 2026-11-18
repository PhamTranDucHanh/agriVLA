# Prompt3 — đối chứng initialization, chưa chọn cấu hình full train

Ngày16/11/2026, branch`feat/improve-accuracy`. Đã chạy **một run mới, đúng300optimizer updates**; tái sử dụng B cũ sau khi chứng minh điều kiện khớp. Không chạy TEST/full train hoặc commit/push. Không đổi dataset/model deploy/readiness. Hai slot run còn lại không dùng: điều kiện chọn init trước khi đối chứng dữ liệu **chưa đạt**.

Kết luận: warm-start giúp fit TRAIN ở ngân sách300updates mới; chưa có bằng chứng đủ mạnh về lợi ích VALID theo protocol đã chốt. Không gọi fresh init thất bại ở mọi ngân sách, không kết luận representation là nguyên nhân duy nhất, không dùng kết quả pilot để dự đoán accuracy full.

## Protocol đóng băng

Config`configs/initialization_ablation_v1.json`; receipts`reports/initialization-ablation-v1-20261116/`. `protocol-lock.json` pin hashes/panels trước chạy A. B đã có kết quả trước khi định nghĩa protocol; A chưa có kết quả để điều chỉnh ngưỡng.

-A: fresh expert-wide LoRA r8/alpha8 trên pinned pretrained base.
-B: weights-only từ endpoint60 checkpoint600, SHA083aac7c127b…; optimizer mới ở bước0, sau100 native resume tới300. B có lịch sử học TRAIN60 trước đó; so300updates mới không đồng nghĩa cùng tổng compute lịch sử.
-Cùng280original TRAINrecords,115scene,287ảnh nguồn/8groups; endpoint_t1, native Euler1, LR1e-4, warmup20/constant, batch4, seed42, frozenVLM/vision;1756800trainable params. Ba cameras256, tokenizer96, saved processor và fullTRAINstats b0684934… không đổi.
-TRAIN280, old60/new220 dùng noise42 để so lịch sử. VALID112 cố định selectorv2 dùng0/1/2/3/42; B1. `evaluation-scope.json` ghi rõ phạm vi này trước A300evaluation để tiết kiệm lặp TRAIN nhiều noise; không đổi panels/ngưỡng.
-Chọn init chỉ khi meanVALIDexact hơn≥5điểm%, zone≥2điểm%, hơn ở≥4/5seed, conditional scene-bootstrap95%lower>0, không làm STOP/falsewatering xấu đi, qua các TRAIN280/old60/new220 guards. Ngưỡng không bị hạ sau kết quả. Readiness full cũ giữ nguyên.

Historical B audit: native configs policy/optimizer/scheduler/dataset/seed/batch/workers/PEFT khớp.100updates đầu optimizer mới,200sau resume giữ702optimizer/8RNG tensors vàscheduler/step/AMPexact. A/B khớp **đủ300batch episodes**, RNG ở1/99/100/101/300đều bằng nhau. Khác biệt wrapper cũ/mới là telemetry và augmentation hook no-op cho profilev9. Đây là TRAIN thật, không phải checkpointfixture.

## Kết quả

### Learning curves, noise42

| Arm / update | TRAIN280 exact | TRAIN280 zone | VALID112 exact | VALID112 zone | STOP VALID |
|---|---:|---:|---:|---:|---:|
|A fresh100|18/280=6,43%|19,64%|8/112=7,14%|20,24%|0/16|
|A fresh300|26/280=9,29%|28,81%|8/112=7,14%|24,70%|4/16|
|B warm100|76/280=27,14%|47,50%|7/112=6,25%|28,27%|0/16|
|B warm300|107/280=38,21%|61,67%|13/112=11,61%|36,31%|3/16|

Ở300: old60 A4/60=6,67%, B55/60=91,67%; new220 A22/220=10%, B52/220=23,64%. A còn underfit rõ; B vẫn chưa fit phần TRAIN rộng. Không lấy điểm old60 để nói full TRAIN đã học chuẩn.100→300 A loss/MAE/zone cải thiện nhưng exactVALID không tăng: loss không thay thế sampled-action accuracy.

### VALID112 qua năm noise seeds, không có optimizer updates

| Metric trung bình | A fresh300 | B warm300 |
|---|---:|---:|
|Exact / macro-intent exact|5,89%|8,57%|
|Zone accuracy|22,50%|35,36%|
|STOP exact|15,00%|16,25%|
|False-watering / target-OFF components|91,23%|71,79%|
|Continuous MAE|0,9183|0,8552|
|Exact sample std giữa5noise seeds|1,20điểm%|3,26điểm%|

Exact theo seeds0/1/2/3/42: A6/6/8/5/8 trên112; B8/12/11/4/13 trên112. B thắng4/5seed nhưng seed3 thua A. Mỗi seed vẫn là cùng112record; **không coi560lượt là560held-out mẫu độc lập**. Macroexact bằng overall vì panel cân bằng16record/intent.

Nhóm VALID, trung bình5noise seeds:

| Nhóm | A exact | B exact |
|---|---:|---:|
|VI|4,64%|6,07%|
|EN|7,14%|11,07%|
|Seen wording|3,57%|12,14%|
|Unseen wording|8,21%|5,00%|
|AUTO|0%|0%|
|CAP|0%|10%|
|EXCLUDE|0%|0%|
|STOP|15%|16,25%|
|WATER_ALL|26,25%|27,50%|
|WORST|0%|3,75%|
|ZONE|0%|2,50%|

B hơn exactmean **2,68điểm%**, dưới ngưỡng5. Conditional paired scene-bootstrap10000draws,25scene clusters: B−A khoảng95% **[−1,67;+7,08]điểm%**, còn chứa0. Scene dùng chung một số ảnh nên khoảng này chỉ exploratory/conditional, không phải uncertainty trên nguồn độc lập. Cả hai còn thấp hơn OFF32,14%exact/63,10%zone trên cùngVALID112.

**Quyết định:** không init nào qua mọi tiêu chí. Không chạy đối chứng counterfactual, không tự tiêu hết3run, không thay đổi threshold để mở phase2. Kết quả không chứng minh counterfactual tốt/xấu, vì chưa train nó. Manifest Prompt2 vẫn là dữ liệu đã QA, chưa có chứng nhận GPU training loader cho view đó.

Training-seed variability chưa đo: cảhai dùng training seed42. Năm seed ở đây chỉ đổi Gaussian noise khi inference. Warm-start có lợi về fitting dưới budget này nhưng chưa khái quát đủ; chưa đủ dữ kiện chọn dropout/rank hoặc mở VLM. Ưu tiên tiếp theo là frozen interventions/representation probe đã đề xuất trong DIAGNOSIS trước khi tăng thêm knobs hoặc full run.

## Bằng chứng kỹ thuật và tài nguyên thật

-A có234adapter tensors,1756800params,702optimizer tensors;234optimizer-step tensors đều300. Loss/gradient norms ghi nhận đều finite, savedweights/optimizer finite; adapter squared-delta soactual initial=12,1031>0.
-Fresh-process replayVALID42 cảA/B khớp toàn112record, maxabsoluteerror0, quantizedexact; tolerance khai báo1e-3. Native predict/select_action, processor/normalization hiện hành.
-A update mean2,84784s; dataload mean0,22528s/update.300updates thuần14,24phút; run tổng18phút15giây gồm startup/screens/save.
-Torch training allocated peak1875,84MiB/reserved1896MiB; NVML peak quan sát2237/4096MiB, headroom1859MiB. NVML1Hz có thể bỏ spike ngắn. RAMprocess HWM≈3,67GiB, RSScuối≈2,81GiB, swap0.
-B reused meanupdate2,78993s; Torch allocated1880,39/reserved1916MiB; NVML2225MiB. Hai run khác thời điểm/hostbackground nên timing/memory là số đo từngrun, không phải kiểm soát công suất/thermal.
-Frozen inference mới: A100≈243,26s; A300≈241,93s; VALIDfive A≈395,93s; B≈396,33s; model load nằm ngoài timer inference. Toàn lượt nhiều thời gian đánh giá hơn một pilot thường. Không lấy toàn thời gian phiên chia300 để suy ETAfull.
-Chỉ giữ A300 là best/latest: exactVALID42 A100=A300, A300zone tốt hơn. A99/A100 thuộc run mới đã hash rồi dọn; curves/predictions/hashes còn đủ trong receipts. Mọi model cũ giữ nguyên. `checkpoint-retention.json` ghi chính xác paths/hashes đã dọn.

Chốt artifact mới khoảng **29,54MiB**, HFdatasetcache tăng0byte theo so sánh trước/sau, dưới budget1GiB. `final-verification.json` kiểm lại mọi protocolhash và checkpoint hash. Các job đã kết thúc, NVMLcuối250/4096MiB và không còn compute process trong query. Ba tests mới cho tiêu chí chọn/init và bootstrapPASS; không dùng tests đó thay bằngchứng trainingGPU thật. Toàn lượt tính từprotocolfreeze khoảng45phút lúc tổng hợpresources, gồm train/evaluation/audit; phần trainA riêng18phút15giây.

## Dự trù full: khác pilot như thế nào

Đây là phép ngoại suy cho **original WORDING_V1 TRAIN3600**, cùng profile/batch4/input. Không chạy full hoặc gọi modelready. Full có1440scene/1258ảnh, so115scene/287ảnh của panel. Thêm record tăng supervision và độ phủ, **không tạo thêm held-out source groups**: TRAIN vẫn8groups, VALID thuộc group khác theo split gốc.

`updates_per_epoch = 3600 / 4 = 900`.300updates pilot280≈4,29epochs;300updates full chỉ0,33epoch. Muốn cùng4,29lượt dữ liệu cần≈3857updates, chưa đảm bảo cùng convergence hoặc quality.

| Full updates | Lượt TRAIN3600 | Update + dataload từ pilot, chưa VALID/setup |
|---|---:|---:|
|100|0,11|5,1phút|
|300|0,33|15,4phút|
|1000|1,11|51,2phút|
|2000|2,22|1giờ42phút|
|3857|4,29|3giờ18phút|
|6000|6,67|5giờ07phút|

Tính: `steps × (2,84784 + 0,22528)s`. MộtVALID720 single-noise B1 ngoại suy từ75,39s/112≈8,08phút.6000updates với6lượtVALID thêm≈48,5phút; TESTcuối sau khi freeze nếu được phép tương lai thêm≈8phút. Thêmstartup/save/thermal/I/O: **dự trù trung tâm6–8giờ**, để dư lịch8–10giờ nếu máy nóng/chậm; đây không phải trần đã đo. Nếu mỗilượtVALID dùng5noise thì chi phí đánh giá tăng gần5lần. Hiện không mở TEST mới.

Full có thể tốt hơn vì nhiều ảnh/case/languagecontext giúp học ổn hơn và giảm ghi nhớ6scene. Nó cũng có thể tệ hơn nếu giữ300steps, LR/schedule chưa phù hợp, sampler lặp shortcut, noise chưa học triệt tiêu hoặc overfit tiếp. Warmup20/constantLR ở pilot không tự trở thành recipefull tốt. Cần lịch học/checkpoints theo số exposures và chọn bằngVALID; không sao chép blindly steps/warmup, cũng không nhân LR theo kích thước dataset.

Distribution action cũng khác: target-OFF components TRAIN280=49,52%, fullTRAIN=48,54%, VALID112=63,10%. Nhiều TRAIN hơn không tự sửa lệch phân bố đó hoặc unseen wording. Không có cơ sở suy “full sẽ đạt X%” từ các số hiện tại.

Full resource dự trù:

-**VRAM:** batch/input/model giữ nguyên nên không tăng12,86lần theo recordcount. Pilot đo2237MiBtotal/4096; cần longest-prompt và full-loader GPU smoke riêng trước run dài. Không xem pilotfit là fullproof; mở VLM/đổihead/batch/objective phải đo lại.
-**RAM:** pilot HWM3,67GiB. Có thể dự phòng khoảng4–6GiBprocess cho full, cộngOS/filecache; đây là dự phòng chưa đo. Native worker0 đọc batch, không cần giữ toàn bộ ảnh decoded. WSL RAM7,6GiB có thể chịu pressure nếu mở thêm ứng dụng; chưa có chứng nhận RAMfull.
-**Disk, cache tái sử dụng:** native checkpoint≈23,7MiB; best+latest≈47,4MiB, đang ghi bản kế có thể tạm≈71MiB; deploy adapter bundle≈10MiB. Logs/predictions/receipts thêmvài–vàichụcMiB. Dự trù0,1–0,3GiB mới nếu cache hiện tại được tái sử dụng, dành0,5–1GiBartifactbudget. Không export lại ảnh/modelbase/ZIP hoặc giữ mọicheckpoint.
-**Cache chưa tái sử dụng được:** HFArrow/cache có thể phát sinh hàngGiB; chưa có fullsmoke đo delta nên không cam kết budget1GiB cho mọi full loader. Phải audit cache và dừng nếu vượt budget, không tự tải/nhân bản dataset. VHDX củaWSL có thể không co khixóafile, nên logical artifact bytes không phải cam kết dung lượngWindows; không dựa vào900GiBảo từdf.

Nếu dùng counterfactual view17738record, không gọi nó là original3600epoch. Prepared sampler5760records/cycle tương đương1440updates/cycle ởB4;6000updates≈4,17cycles. Đó là supervision lặp trên cùng1258ảnh, không phải thêmảnh. Full ETA trên chỉ áp dụng cùng loại update, chưa đo CPUoverhead cho loadercounterfactual. Hiệnview này chưa train nên chưa chọn làm full recipe.

## Artifact và lệnh tái lập

ModelA mới để chẩn đoán, không phải modeldeploy tốt:

`outputs/small-cases/initialization-ablation-v1-20261116-A/checkpoints/000300/pretrained_model/adapter_model.safetensors`.

ModelB giữ nguyên:

`outputs/small-cases/pilot280-resume100-20261116/checkpoints/000300/pretrained_model/adapter_model.safetensors`.

Giữ wholepretrained_model+base/backbone cache để inference, không chỉcopy một weightsfile. Không đổi selected.env hoặc webmodel.

Receipts quan trọng: `protocol-lock.json`, `historical-B-audit.json`, `A-B-resolved-contract.json`, `A-B-fairness.json`, `A-training-audit.json`, `A300-five-noise-seeds.json`, `B300-five-noise-seeds.json`, `decision.json`, `learning-curves.csv`, `loss-gradient-curves.csv`, `group-metrics.csv`, `predictions-with-images.csv`, `resources.json`, `full-training-projection.json`, `smoke-report.json`.

Lệnh tái lập A vào **outputmới**, người dùng chạy riêng nếu muốn, không thuộc run đã hoàn tất:

```bash
cd /home/hanhpham/agriVLA
export HF_DATASETS_CACHE="$PWD/agri-soybean-vla/.cache/hf-datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
AGRI_TAG=$(date +%Y%m%d-%H%M%S)
agri-soybean-vla/.venv-train/bin/python -u -m agri_vla.small_train \
  --stage cases280 --evaluate-stage cases60 \
  --profile r8_wide_b4_n1_t1_lr1e4 --steps 300 \
  --experiment-config agri-soybean-vla/configs/small_cases_v9.yaml \
  --output "agri-soybean-vla/outputs/small-cases/user-init-A-$AGRI_TAG"
```

Thêm `--init-checkpoint` endpoint60/000600/pretrained_model để tái lập B từweights-only; không dùng `--resume-checkpoint` cho bước0củaB. Resume100→300 lịch sử đã được audit. Evaluator bounded mới: `python -m agri_vla.ablation evaluate --checkpoint PATH --output NEW.json --seeds 42`; confirmation dùng `python -m agri_vla.ablation_validation --reference SEED42.json --output NEW-five.json`. Dùng `.venv-train/bin/python` và HFcacheenv nhưtrên. Rawoutputs không ghi đè; A100checkpoint đã dọn theo retention nên chỉ có thể tái kiểm từreceipt hoặc run tái lập mới.

Không cung cấp lệnh full`--execute` như một config đã được chọn: initializationcriterion và qualitygates chưa đạt, chưa có fullrecipe. Workflowfull cũ vẫn có khóa bằngchứng. Giai đoạn này chỉ chốt được kết quả đối chứng và giới hạn ngoại suy.
