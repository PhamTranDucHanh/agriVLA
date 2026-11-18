# Chẩn đoán checkpoint frozen — 16/11/2026

Phạm vi: Prompt 1, branch `feat/improve-accuracy`. Chỉ đọc source, TRAIN/VAL manifests và receipts, tính lại metrics bằng CPU. **Không chạy inference mới, không backward/optimizer/train, không mở TEST, không sửa dataset/model/vendor hoặc checkpoint deploy.** Các phép đối chiếu sẵn có đủ để xác nhận những giới hạn bên dưới; chưa cần tiêu GPU cho một run bổ sung.

Kết luận chính: dữ liệu nạp đúng ở các kiểm tra hiện có, nhưng model chưa học policy trên TRAIN rộng. TRAIN60 chỉ có **6 scene**. Đối chứng TRAIN280 step300 đúng **55/60 mẫu cũ**, nhưng chỉ **52/220 mẫu bổ sung**. VALID112 đúng13/112. Đây đồng thời là ghi nhớ tập hẹp, thiếu fit trên phần TRAIN rộng và khái quát kém; không thể gán toàn bộ cho một lỗi inference hay một nguyên nhân overfit.

## Bằng chứng và phạm vi kiểm

Đọc AGENTS.md, context/status, MODEL_TRAINING.md, OPTIMIZATION_2026-11-16.md, configs v7–v10, small_train/small_cases/pilot/evaluation/inference_sampling/augmentation và demo_server. Raw receipts ưu tiên hơn những dòng status cũ còn ghi “đang chạy”.

Artifact mới ở `reports/diagnosis-20261116-final/`:

- `checkpoint-evidence.json/.csv`: đường dẫn tuyệt đối, nguồn init/resume, objective/profile, panel/scenes, seeds, processor/stats SHA, adapter SHA đầy đủ và checkpoint-lock SHA.
- `group-metrics.csv`: metrics tính lại, bao gồm intent/ngôn ngữ/wording và old60/new220.
- `target-zone-confusion.csv`: counts theo từng zone, từng target level và prediction0..3.
- `predictions-with-images.csv`: 2.784 lượt dự đoán có sẵn, gồm nhiều checkpoint/seed trên cùng sample; **không phải2.784 mẫu độc lập**.
- `error-examples.csv`: toàn bộ1.696 lượt sai; ảnh tuyệt đối, instruction, target, continuous/quantized prediction. `review-14-errors.csv`: một lỗi mỗi intent/ngôn ngữ của baseline300 VALID, chọn sample_id nhỏ nhất, không chọn theo lỗi nặng/nhẹ.
- `translation-pairs.csv`, `same-scene-different-target.csv`, `identical-instruction-validation.csv`, `noise-existing-train60.csv`: các đối chiếu không tạo prediction mới.
- `metadata-boundary.json`, `input-receipt-hashes.json`: kiểm allowlist và hash nguồn.

Script tái lập: `scripts/diagnose_frozen_receipts.py` bằng `.venv/bin/python`. Output có chế độ tạo mới, chạy lại cần đổi OUT sang folder mới. Chỉ import stdlib; không nạp model. Folder `diagnosis-20261116-incomplete` là lần đầu dừng vì AST camera constant là comprehension; folder `diagnosis-20261116` là bản trung gian chưa ghép certificate single. **Dùng folder `-final`**; giữ lịch sử, không xóa artifact cũ.

Kiểm mới: join sample_id→TRAIN/VAL manifest; target khớp, finite continuous, lượng tử hóa đúng clip0..3 rồi floor(x+.5), ảnh tham chiếu tồn tại. Cả7 model variants có adapter SHA khớp receipt, mọi file trong checkpoint lock khớp, saved stats khớp prepared full TRAIN. Receipt CPU trước đó kiểm840TRAIN+336VAL pixel tensors khớpraw maxerror0 và instruction/action/state mapping exact; lượt này đọc lại bằng chứng, không chạy lại tensor audit.

## Bảng checkpoint có prediction hợp lệ

Tất cả dùng wide expert LoRA r8/alpha8, frozen VLM/vision, TRAIN seed42, ba camera256, tokenizer96, processor đã lưu, full-TRAIN ACTION MEAN_STD/STATE IDENTITY. Objective endpoint_t1 + native Euler1 là biến thể thực nghiệm, không phải flow-matching Beta mặc định. VAL đều cùng112record, seed42, B1. TRAIN280 đo B4; capacity60 đo B1,5seeds0/1/2/3/42. Không đánh đồng khác biệt batch với causal evidence về split; fresh reload comparisons trước đó có maxerror0 trên phạm vi đã đo.

| Variant / prefix adapter SHA256 | Panel / scenes | Step riêng run | Init / continuation | TRAIN exact seed42 | VALID exact / zone |
|---|---:|---:|---|---:|---:|
| endpoint60 single /083aac7c127b |60/6|600|v7 endpoint300→native resume600; trước endpoint là weights-only từ v6 lowLR300|60/60|7/112;30,95%|
| endpoint60 MC2 /083aac7c127b |60/6|600|cùng bytes adapter single; đổi saved inference profile v8, không train mới|60/60|6/112;29,46%|
| pilot280 LR1e-4 /b0c4f1af8023 |280/115|100|endpoint600 weights-only, optimizer mới|76/280|7/112;28,27%|
| pilot280 LR1e-4 /b48d69ec6572 |280/115|300|native full-state từ100|107/280|13/112;36,31%|
| pilot280 mix LR1e-4 /2e5b3f35cbbb |280/115|100|cùng endpoint600, optimizer mới; chỉ thêm TRAIN override-image-mix|66/280|6/112;27,68%|
| pilot280 mix LR1e-4 /d37e52c2f07c |280/115|300|native full-state từ mix100|94/280|15/112;37,50%|
| pilot280 LR3e-4 /e6b33a64657f |280/115|100|cùng endpoint600, optimizer mới; không mix|80/280|12/112;38,99%|

Step là bộ đếm của từng run, không cộng curriculum rồi gọi đó là epoch full dataset. Stats SHA256 chung: `b06849340aa520815152fd63921b3602ab9278f7c3fd242e600c6a33a086c78f`. CSV/JSON có full hashes và paths; prefix trong bảng chỉ để đọc.

Nhánh LR3e-4 resume100 có `updates.jsonl` tới300, nhưng không có `result.json`, không có checkpoint con trong thư mục checkpoints và không có pilot300 evaluation tại thời điểm audit. **Không có checkpoint300 hoàn chỉnh để xếp hạng; accuracy chưa biết.** Không tự chạy lại job này. Checkpoint6000 lịch sử và v6 là nguồn bối cảnh, không đưa documented TEST cũ vào phép đánh giá mới này.

## Model đang học cái gì

### 1. Phần lớn điểm TRAIN280 đến từ 60 mẫu cũ

| Run | Old60 | New220 | Toàn280 | VALID112 |
|---|---:|---:|---:|---:|
|LR1e-4,100|58/60=96,67%|18/220=8,18%|27,14%|6,25%|
|LR1e-4,300|55/60=91,67%|52/220=23,64%|38,21%|11,61%|
|Mix LR1e-4,300|51/60=85%|43/220=19,55%|33,57%|13,39%|
|LR3e-4,100|37/60=61,67%|43/220=19,55%|28,57%|10,71%|

Đã xác nhận model chưa fit phần rộng hơn. LR cao ở100steps làm new220 tốt hơn LR thấp100 nhưng mất nhiều điểm old60; đây là tradeoff cần đo, không phải tiến bộ toàn diện. Chưa chứng minh warm-start60 là nguyên nhân: chưa có fresh-adapter control cùng budget. TRAIN280 chỉ115scene và tái sử dụng ảnh nguồn; counts record không tương đương số quan sát độc lập.

### 2. Sai cả nhóm lệnh không cần thị giác

Baseline LR1e-4 step300 VALID:

| Intent | Đúng exact /16 | Continuous MAE | False watering / target-OFF components |
|---|---:|---:|---:|
|AUTO|0|1,0488|23/28|
|CAP_LEVEL|3|0,7283|23/32|
|EXCLUDE_ZONE|0|0,9112|26/36|
|STOP_ALL|3|0,5448|23/48|
|WATER_ALL|5|0,5248|không có targetOFF|
|WORST_ONLY|1|1,0476|29/32|
|ZONE_ONLY|1|1,0596|28/36|

STOP/WATER_ALL không cần wilt để tính target nhưng vẫn sai12–13/16 mẫu. Do đó lỗi toàn bộ không thể chỉ do nhận biết héo. Chưa phân biệt được lỗi hiểu text, conditioning với ảnh lạ hay action decoder bằng các con số này. Không hardcode STOP ở deploy rồi tính thành model accuracy.

### 3. VI/EN và wording chưa ổn định

Baseline300: EN8/56=14,29%, VI5/56=8,93%; seen9/56=16,07%, unseen4/56=7,14%. Khác biệt ngôn ngữ trên panel nhỏ chưa đủ kết luận bản thân backbone yếu tiếng Việt hơn.

Trong56cặp dịch VALID, **48cặp cho vector rời rạc khác nhau**,9cặp chỉ một ngôn ngữ đúng. Các cặp có cùng ảnh/target và cùng noise seed. TRAIN280 cũng52/140cặp khác prediction. Điều này chứng minh output thiếu bất biến theo cách diễn đạt tương đương trên các mẫu đó; không chứng minh tokenizer hoặc bản dịch sai. Pixel/task audit trước đó khớp.

Trong56VALID có câu trùng nguyên văn với TRAIN280 đã đánh giá, chỉ9đúng. Với24VALID có câu trùng TRAIN60 cũ, baseline300 chỉ4đúng; endpoint600 chỉ2đúng. Vậy unseen wording không giải thích toàn bộ gap. Đây là so giữa scene khác nhau, không phải ảnh intervention được cô lập hoàn toàn.

### 4. Có lỗi phối hợp ảnh–lệnh, chưa cô lập được thị giác

Từ raw receipts baseline300,96cặp cùng scene/ngôn ngữ nhưng khác target trên VALID có20cặp trả cùng vector dự đoán. TRAIN có3/216cặp như vậy. Đây là counts các cặp đối chiếu có chia sẻ scene/record, không phải96/216 thí nghiệm độc lập. Cùng scene trả khác vector cũng chưa chứng minh hiểu đúng: nhiều prediction khác nhau nhưng đều sai.

AUTO VALID0/16 và câu seen vẫn thấp gợi ý lỗi khái quát thị giác hoặc kết hợp modality. **Chưa có wilt classifier/probe trực tiếp hoặc single-camera intervention mới**, nên không kết luận vision encoder mất thông tin héo, không kết luận thứ tự camera sai. FullTRAIN scene audit cũ:1336/1440scene chỉ một case; đây là nguy cơ shortcut, chưa phải chứng minh model đang bỏ qua text.

### 5. Lệch đầu ra về mức1, không chỉ sai sát ngưỡng

Baseline300 VALID có336zone outputs. Theo target0/1/2/3, số dự đoán đúng là60/212,44/68,10/28,8/28. Riêng212zone cầnOFF lại có116dự đoán1,35dự đoán2 và1dự đoán3: false-watering152/212=71,70%. Tổng189/336outputs là mức1; target mức1 chỉ68/336. Zone1/2/3 accuracy lần lượt46/112,33/112,43/112.

MAE0,8379 cho thấy không thể chỉ sửa ngưỡng làm tròn một chút. Chưa chứng minh đây là conditional-mean collapse của MSE: thiếu representation, conditioning sai và thiếu fit cũng có thể tạo pattern này. Saved stats đã khớp, không có bằng chứng đổi stats để chữa bias.

### 6. Sampling vẫn có nhiễu nhưng MC2 chưa chữa khái quát

Certificate single TRAIN60:299/300lượt đúng; chỉ1/60sample sai ở một seed, continuous max range theo seed tới0,7697. MC2:300/300đúng, maxrange còn0,4017; output continuous vẫn không hoàn toàn xác định. Cùng weights VAL seed42 single7/112, MC2 6/112. Vậy chỉ tăng số noise samples chưa giúp trên panel này.

**Không có sweep đa-seed VALID mới**, nên chưa định lượng variance VALID hoặc khẳng định sampling không đóng vai trò. Không chọn seed đẹp; kết quả5seeds TRAIN không thay thế chứng nhận ngoài mẫu.

## Metadata có lọt vào inference không

Đã thực thi chính thân hàm `evaluation.model_batch` trích AST với identity sentinels. Kết quả chỉ giữ `observation.images.camera1/2/3`, `observation.state`, `task`; loại `teacher_annotations`, `intent`, `target_action`, `action`, `case_id`, `wilt_notes`, `revision_metadata`. Giữ đúng object identity của các input hợp lệ.

Source review: `predict_batch` chỉ đưa `pre(model_batch(batch))` vào native select_action; panel_evaluate cũng dùng cùng allowlist; pilot lấy target/intent sau đó để tính metrics; inference reset queue/processors. Demo tạo batch mới từ ba ảnh, zero state, instruction; wilt_notes chỉ nằm trong response. Augmentation dùng intent/target TRAIN để xác định phép biến đổi hợp lệ, không thêm chúng vào features. Supervised loss có target là bình thường; target không là inference feature.

Kết luận chắc chắn trong phạm vi hàm biên và codepath đã đọc: **không thấy teacher/intent/target đi vào model inference**. Đây không phải lần capture GPU end-to-end mới trên mọi plugin/processors; không tuyên bố đã chứng minh mọi đường chạy ngoài repo.

## Mức chắc chắn và ba thí nghiệm tiếp theo

**Đã xác minh:** chưa fit TRAIN280; old60 tốt hơn new220 rất nhiều; translation invariance thấp; STOP ảnh-independent cũng fail; lệch output về1; MC2 chỉ cải thiện capacity60; stats/hash/label joins đúng trong phạm vi kiểm. **Giả thuyết chưa chứng minh:** warm-start60 gây bất lợi, visual shortcut, frozen representation thiếu tín hiệu wilt, endpoint loss không phù hợp. Không có một nguyên nhân duy nhất đã được cô lập.

1. **Frozen intervention panel (ưu tiên đầu tiên, không train):** chốt trước một panel chỉ từ TRAIN/VAL và seeds0/1/2/3/42. Cùng ảnh đổi STOP/WATER/AUTO/zone; cặp dịch; STOP/WATER đổi scene; AUTO thay đúng một camera bằng ảnh cùng split và tính target mới từ policy hiện hành. Giữ nguyên camera còn lại/câu/noise, tách synthetic diagnostic khỏi benchmark. Chọn hướng ngôn ngữ nếu các override/cặp dịch vẫn lỗi; chọn thị giác nếu override ổn nhưng AUTO không phản ứng đúng camera/độhéo. Nếu output đổi theo noise nhiều hơn theo thay đổi hợp lệ input, ưu tiên decoding/objective. Không coi “output có đổi” là đúng, phải so target và vùng cần bất biến.
2. **Đối chứng initialization:** fresh wideLoRA trên pretrained base so warm-start60, cùng TRAIN280 gốc/objective/LR/batch/sample budget/seeds, optimizer mới ở cảhai. Chọn fresh nếu cải thiện ổn định new220/VAL và không sụp old60/STOP; bỏ giả thuyết curriculum là nguyên nhân chính nếu hai nhánh tương đương trong khoảng bất định. Không dùng số tốt nhất một seed. Chưa chạy trong lượt này.
3. **Đối chứng coverage cùng scene–nhiều lệnh:** từ TRAIN tạo manifest tham chiếu ảnh cũ, target tính lại đúng WORDING_V1; cân bằng scene/source trước intent/tham số, giữ cặp dịch. So data gốc với counterfactual cùng init/budget. Chọn nếu giảm lỗi cặp dịch, cùng-scene constraints và false-watering, đồng thời cải thiện VALID; bỏ nếu chỉ fit augmentation hoặc chỉ tăng1–2record VALID. Đã có image-mix v10 nhưng nó chỉ đổi override, chưa thay thế đối chứng coverage toàn policy. Chưa chạy trong lượt này.

Chưa ưu tiên tăng rank/dropout, mở VLM hoặc đổi head phân loại ngay: thiếu causal evidence để chọn đúng thay đổi. Nếu ba chẩn đoán trên cho thấy representation/objective mới là nút thắt, mới lập biến thể kiến trúc riêng và đo VRAM/backward sau. Giữ TEST cho kết luận cuối; VALID112 đã là tập phát triển, không gọi một cải thiện nhỏ trên panel này là accuracy đồ án.
