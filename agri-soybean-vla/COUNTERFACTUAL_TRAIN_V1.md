# TRAIN đối chiếu v1 — chuẩn bị dữ liệu, chưa train

Ngày16/11/2026, branch `feat/improve-accuracy`. Thực hiện Prompt2 sau [báo cáo chẩn đoán](DIAGNOSIS_2026-11-16.md). Không train/backward/optimizer/inference model, không đổi checkpoint deploy hoặc readiness gates. WORDING_V1 gốc3600/720/720, ảnh, splits, locks, saved processor/tokenizer96 và fullTRAIN stats được giữ nguyên, kiểm SHA trước/sau.

## Kết quả và vị trí file

Config mới: `configs/counterfactual_train_v1.json`. Code CPU: `src/agri_vla/counterfactual.py`. Output: **`reports/counterfactual-train-v1/`**.

| File | Nội dung |
|---|---|
|`manifest.jsonl`|View17738records:3600original +14138counterfactual; tham chiếu ảnh cũ|
|`sampling-plan.jsonl`|2880scene visits, mỗi visit chọn một cặp VI/EN của view|
|`original-control-plan.jsonl`|Cùng2880scene/anchor theo đúng thứ tự, chỉ chọn record gốc|
|`original-coverage.json`, `view-coverage.json`|Audit scene, group, ảnh, intent/tham số, wilt/action, VI/EN và ties|
|`sampling-coverage.json`, `original-control-coverage.json`|Phân bố thực tế của hai lịch lấy mẫu|
|`source-image-audit.csv`, `image-hashes.json`|1258ảnh nguồn, teacher wilt và số lần tham chiếu; bytesSHA|
|`preview.html`|16scene/48ảnh tham chiếu với các câu và target để xem trực quan|
|`qa-report.json`, `artifact-lock.json`|QA nhãn/leakage, hashes config/code/output/protected inputs|
|`independent-verification.json`|Đọc lại manifest/schedule thật và kiểm nguồn scene/câu, target, cap|
|`preview-http-complete.json`|HTML và toàn bộ bytes của3ảnh qua HTTP khớp disk|

Tổng artifact sinh dữ liệu khoảng **19,83MiB**, thấp hơn budget200MiB. Không copy ảnh, không export LeRobot/Arrow mới, không tải model hoặc tạo checkpoint. View mới **chưa được nối vào native training loader**; không truyền path này vào launcher cũ rồi mặc định nó sẽ dùng được. Không có `full_ready.json` mới.

## Audit TRAIN gốc

-3600record,1440scene,1258ảnh nguồn,8group,42template families đã xuất hiện. VI1800/EN1800.
-1336scene chỉ một case lệnh;98scene có4case;4scene11case;2scene14case. Cùng ảnh/scene được lặp qua nhiều record, không tính các record đó là quan sát độc lập.
-Intent counts: AUTO900, WORST600, ZONE450, CAP450, EXCLUDE300, STOP300, WATER_ALL600. ZONE1/2/3 đều150; EXCLUDE1/2/3 đều100; CAP1/2 là226/224; WATER1/2/3 đều200.
-1258ảnh độc lập theo source ID có wilt0/1/2/3/4: **466/343/248/65/136**. Counts camera theo record gần đều vì cách compose, nhưng không làm mất sự thiếu ảnh nguồn wilt3.
-8group có số ảnh nguồn:10-87=256,14-198=124,15-29=90,16-60=105,3-106=159,5-180=99,5-27=284,7-130=141. Một scene có thể chứa nhiều group.
-344scene có max-wilt≥2 và đồng hạng; ban đầu chỉ66scene trong số đó có câu WORST. Có95scene toàn bộ dưới ngưỡng.
-Action components0/1/2/3 của dữ liệu gốc:5242/2022/1796/1740. Label không phải lượng nước vật lý đã hiệu chuẩn.

Counts chi tiết có trong JSON/CSV; không gọi1440scene là1440nguồn ảnh độc lập vì ảnh được tái sử dụng giữa scene.

## Cách tạo view

Giữ đủ3600record gốc bằng sample_id. Với mỗi scene TRAIN đã có:

1. Chọn STOP và một WATER_ALL có level1/2/3 luân phiên cân bằng trên scene.
2. Chọn hai yêu cầu phụ thuộc ảnh từ AUTO, WORST, ZONE1/2/3, EXCLUDE1/2/3, CAP1/2. Với mọi eligible tie scene, bảo đảm có WORST, thay một slot nếu cần.
3. Mỗi yêu cầu có cặp VI/EN. Thêm một family đồng nghĩa khác cho một slot luân phiên. Câu được sao nguyên văn từ cặp dịch **đã có trong TRAIN**, đúng intent và đúng tham số; lưu `wording_source_sample_id`.
4. Tham chiếu đúng ba ảnh/teacher của scene gốc, lưu `scene_source_sample_id`. Tính target mới theo yêu cầu mới.
5. Loại bản trùng cùng scene/ngôn ngữ/câu. Nếu cùng input có target khác thì dừng QA.

Không sinh14yêu cầu ×mọi family trên mọi scene. Không ghép camera tùy ý, crop hay đổi màu. Các record nguyên bản vốn nhiều case vẫn được giữ; view có4–14case/scene, nhiều nhất38record/scene. Mỗi scene có cả STOP và WATER để tạo target đối lập ngay cả khi tất cả cây dưới ngưỡng. Cả344eligible tie scenes nay có WORST. Mọi scene có ít nhất một yêu cầu với hai family diễn đạt khác nhau.

View có VI8869/EN8869. Record counts view chưa cân bằng intent: STOP3840, WATER4156, AUTO1622, WORST2144, ZONE2182, CAP1894, EXCLUDE1900. **Không dùng uniform shuffle toàn17738record làm sampler mặc định**: cách đó sẽ thiên về override và các scene vốn nhiều câu.

Không thêm thông tin quan sát mới: vẫn1258ảnh và8group. Việc tăng số câu/target chỉ giúp tạo supervision quan hệ, chưa chứng minh tăng VALID accuracy.

## Policy, nhãn và chống leakage

Gọi đúng **`agri_vla.revision.revision_action`**, policy`wilt_proxy_with_operator_override_v1`. Hàm này thêm WATER_ALL và kế thừa sáu policy còn lại; không gọi trực tiếp policyv0 cho WATER_ALL.

Oracle số học độc lập kiểm đủ125wilt-triplets ×14intent/parameter requests = **1750trường hợp**. WATER_ALL ép[level]*3 bất kểwilt; STOP000; AUTO=max(0,wilt−1); WORST chọn mọi vùng hòa max đủngưỡng; ZONE/CAP/EXCLUDE áp đúng tham số. Từng nhãn gốc và nhãn view được đối chiếu; không có mismatch.

QA nguồn:

-TRAINmanifest khớp hash trong datasetlock WORDING_V1; source split TRAIN manifest khớp source lock.
-Chỉ8TRAINgroups được phép, rời groupVAL/TEST trong source split lock. 1258ảnh tham chiếu khớp bytesSHA provenance của TRAIN; không chọn ảnh theo kết quả VALID.
-Câu chỉ lấy từ actual TRAIN và family thuộc partitiontrain. Câu seen có thể hợp lệ xuất hiện ở VALID theo thiết kế gốc; việc dùng câu từ TRAIN không đồng nghĩa lấy câu từ VALID.
-Không parse VAL/TESTmanifest để tạo view; chỉ hash bytes trước/sau để chứng minh giữ nguyên. Không đọc held-out predictions trong generator.
-Hash saved tokenizer/processors/schema/stats/config trong prepared bundle được kiểm với bundlelock; không đọc lại base weights lớn để tạo bản sao. Không tính normalization từ view hoặc held-out.
-Mọi instruction mới là instruction cũ của TRAIN, nên không xuất hiện chuỗi prompt mới. Token audit đã lưu của TRAIN có maxVI74/maxEN22, capacity96, khôngtruncation. Đây là suy luận từ exact-string reuse và saved-tokenizer hash, không phải một lần tokenize GPU mới.

Teacher/intent/params là metadata phục vụ sinh nhãn/QA. `model_inputs()` chỉ trả ba camera paths +zero state+task. Đây là hợp đồng cho loader tương lai, chưa phải native tensor dataset. Loader phải chỉ dùng target cho supervised loss, không đưa structured intent/wilt/case vào features. Loss/processor/pre/post phải tiếp tục dùng fullTRAIN stats hiện hành.

## Sampling cân bằng có giới hạn

Một chu kỳ chuẩn bị gồm2880scene-visits, mỗi visit là một cặp VI/EN, tổng5760record:

-Lượt đầu phủ đủ1440scene, do đó giữ đủ1258ảnh.
-1440visit tiếp ưu tiên group có camera exposure thấp rồi ảnh ít được lấy. Áp cap cứng tối đa3visit/scene và32camera appearances/ảnh nguồn trong chu kỳ. VI/EN cùng một visit dùng chung triplet nên cap32tương ứng tối đa64record-camera appearances.
-Trong scene chọn intent đang ít được lấy, rồi tham số, family/pair; không để nhiều câu của một case tự động làm case đó nặng hơn. Dùng RNG riêng cho chọn scene và chọn câu, để hai nhánh đối chứng giữ cùng scene/source sequence.
-Phân bố thực tế: AUTO/STOP/WORST/CAP mỗi411pairs; WATER/ZONE/EXCLUDE mỗi412pairs. Các mức WATER137/138pairs, ZONE137/138, EXCLUDE136/137/139, CAP205/206. VI2880/EN2880.
-Scene visit counts:663scene×1,114×2,663×3. Tất cả scene/ảnh đều có mặt. Image-camera appearances1..32.

Group-camera exposure thực tế theo pair-visits:10-87=2493,14-198=950,3-106=949,5-27=1286,5-180=634,7-130=1106,15-29=604,16-60=618. Group10-87 giảm share từ33,74%record-camera ở TRAIN gốc xuống28,85%pair-camera trong schedule, nhưng vẫn nhiều nhất. Không thể đồng thời mặc định mọi group đều nhau, phủ toàn bộ scene và giữ cap; báo counts thực tế thay vì gọi phân bố hoàn toàn đều.

Hai lịch original-control và counterfactual có **cùng thứ tự scene/anchor group/anchor image**, cùng5760record và cùng source exposure. Original-control chỉ chọn original IDs và có intent counts khác do scene gốc không có mọi lệnh. Nhánh này kiểm hiệu quả cả coverage lẫn phân bố supervision, chưa cô lập riêng ảnh hưởng của cân bằng class với thêm counterfactual.

5760record là một chu kỳ sampling được chuẩn bị, không phải dataset gốc đổi thành5760record hay đã train một epoch. Khi train sau này, loader phải replay plan và lưu vị trí plan/RNG khi resume; không chỉ shuffle17738rows. Nếu muốn đổiseed/cycle, sinh plan mới cóversion/hash và QA lại. Chưa kiểm sampler resume bằng optimizer thật.

## Kế hoạch đối chứng sau này, chưa thực thi

1. Đầu tiên giữ cấu hình và init đã chọn bằng đối chứng riêng. Chọn cùng danh sách TRAIN scenes cho hai nhánh, cùng batch/steps/noise/evaluator. Original dùng original records; counterfactual dùng original records cộng generated rows trên đúng những scene đó. Không đưa thêm scene vào riêng một nhánh.
2. Với pilot280 cũ: lấy đúng115scene từ selection280. Control chỉ dùng280original IDs; augmentation dùng các IDs đó +counterfactual rows trên115scene. Gọi `sampling_plan()` cho cảhai với cùngseed/budget/caps và assert scene/source order giốngnhau. Không dùng toàn5760schedule chuẩn bị cho fullTRAIN trong một run được gọi làpilot280.
3. Đo lại old60, new220 và fixedVALID112 ảnh gốc. Chốt metrics trước: exact/zone/macrointent/STOP/falsewatering, VI/EN/seen/unseen và invariance cặp dịch. Không chọn config chỉ bằng augmentation-training loss, không hạ readiness gate.
4. Nếu có cải thiện đáng kể trên cùng samples và nhiều noise seeds, mới xác nhận ít ứng viên trênVALID720. Giữ TEST cho kết luận cuối. Không coi chênh1–2recordVALID112 là bằng chứng vững.
5. Nếu cần cô lập riêng data-vs-class-balancing, thêm control có matched intent marginals nhưng original-only, và báo scene/source exposure thay đổi nào là bất khả tránh. Không tuyên bố đối chứng hiện tại đã tách hai hiệu ứng đó.

## Xem và tái lập

Từ rootrepo, mở web xem dữ liệu bằng môi trường data, không cầnGPU:

```bash
cd /home/hanhpham/agriVLA
agri-soybean-vla/.venv/bin/python -m http.server 8767 --bind 127.0.0.1 --directory /home/hanhpham/agriVLA
```

Mở **http://localhost:8767/agri-soybean-vla/reports/counterfactual-train-v1/preview.html**. Trang hiển thị ảnh gốc, câuVI/EN và target, **chưa có prediction model**. Ctrl+C dừng server. Không thay demo model đang chọn. Có thể mở trực tiếp fileHTML; server tiện hơn trongWSL.

Tái lập generator vào **folder mới**:

```bash
agri-soybean-vla/.venv/bin/python -m agri_vla.counterfactual \
  --config agri-soybean-vla/configs/counterfactual_train_v1.json \
  --output /tmp/agrivla-counterfactual-v1-recheck
```

Không ghi đè folder hiện có. Preview ở output ngoài module cần serve root phù hợp hoặc mởfile trực tiếp vì dùng ảnhrelative. Script tự kiểm hashes, labels, source/text và budget; không mở train launcher. Thử nghiệm146testsPASS gồm test_counterfactual và test_revision. Kiểm mới độc lập đã đọc lại toàn17738rows, replay source sequence hai plans và xác nhận48image links tồn tại; HTTP kiểm HTML+3ảnh đủbytes200PASS. Chưa chụp browser rendering hoặc kiểm model accuracy trên view này.
