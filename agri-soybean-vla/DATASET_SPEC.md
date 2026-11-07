# Hợp đồng dataset AgriVLA-Soybean-v0

**Hiện hành sau yêu cầu bước 6:** dataset revision **wording_v1**, 3.600/720/720 records; JSONL ở data/manifests/wording_v1, LeRobot ở data/lerobot/wording_v1/full/{train,val,test}. Bản v0 bên dưới được giữ làm lịch sử và không bị ghi đè. Source triplets/splits/target sáu intent cũ giữ nguyên; 1.088/402/402 instructions được thay bằng văn nói song ngữ, thêm 600/120/120 WATER_ALL records.

WATER_ALL cần level=1/2/3: “Tưới cả 3 vườn mức 2 nhé” → [2,2,2] bất kể wilt. STOP_ALL “Tắt hết.” → [0,0,0]. AUTO và nhóm theo héo vẫn giữ ngưỡng 2. Vườn trong câu lệnh là tên gọi cho zone tổng hợp, không ngụ ý ba ruộng thật đồng thời. Teacher/structured intent/params/revision_metadata chỉ là metadata; features vẫn ba RGB + state[3] và task, target action[3].

Schema v1: schemas/scene_v1.schema.json. Policy extension: configs/policy_extension_v1.yaml. Dataset lock v1: 92111dfa8086eef77b90e9ac47003c43f1847ea07a899e1df3c6232d2f977ba3; parent c59005a5d4eea36731051f5b82a9341d9be5ae50cca4aafcad1055743ef16563 giữ nguyên. Dùng stats wording_v1/full/train_stats.json (SHA256 b06849340aa520815152fd63921b3602ab9278f7c3fd242e600c6a33a086c78f), STATE IDENTITY/ACTION MEAN_STD. Xem MODEL_TRAINING.md và GPU_HANDOFF.md.

---


Phiên bản scene: agri-soybean-scene-v0. Module nằm trên nhánh feat/compose-dataset. Bước 0–5 đã kiểm chứng môi trường, nguồn/split, policy/templates, scene JSONL và conversion/round-trip LeRobot; có browser localhost bước 4.5. Các phần model/train vẫn là hợp đồng cho các bước tiếp theo.

## Bài toán và giới hạn

Một mẫu biểu diễn một quyết định trên **ba vùng tổng hợp**: ba ảnh đậu tương + một câu lệnh Việt hoặc Anh → action ba chiều. Ảnh có thể đến từ các plot/thời điểm khác nhau; không gọi chúng là ba camera đã chụp đồng thời ngoài thực tế.

Nguồn: [Soybean Zenodo 8256382](https://zenodo.org/records/8256382). Bước 1 đã đối chiếu đủ 1.892 fullID với 1.892 ảnh JPEG RGB 640×480, giải mã và kiểm tra checksum/CRC/hash. Có 1.788 ảnh đủ điều kiện nhãn 0–4 và 104 ảnh nhãn 5 ngoài miền v0; giữ nguyên raw và quarantine, không đổi thành 4. Không có ảnh thiếu/thừa/hỏng hoặc trùng chính xác. Bước 2 đã nhóm/audit near-duplicates theo heuristic đã công bố; bước 4 từ chối cả near và uncertain trong cùng scene.

Nhãn wilt là mức biểu hiện héo do chuyên gia chấm. Không phải độ ẩm đất, lít nước, thời gian mở van hay nhu cầu tưới tối ưu. Nhãn action do nhóm tạo từ policy tổng hợp, chưa được xác nhận bằng thí nghiệm tưới.

## Ba lớp dữ liệu cần phân biệt

| Lớp | Nội dung | Model sử dụng |
|---|---|---|
| Observation | Ba RGB, câu instruction, state hợp lệ | Input |
| Target | target_action [z1,z2,z3] | Nhãn để tính loss khi train |
| Teacher/provenance | wilt, intent/params, split, IDs, template/pair metadata | Sinh nhãn, QA, debug và đánh giá |

Không đưa cả record JSON vào prompt model. Converter tương lai phải chọn đúng fields. Nếu nhét điểm héo vào state hoặc intent có cấu trúc vào input, model có thể đoán target mà không sử dụng ảnh/ngôn ngữ như mục tiêu thí nghiệm.

## Input và output

| Feature | Dạng dự kiến | Ý nghĩa |
|---|---|---|
| observation.images.camera1 | RGB, metadata shape [3,256,256] | zone1 |
| observation.images.camera2 | RGB, metadata shape [3,256,256] | zone2 |
| observation.images.camera3 | RGB, metadata shape [3,256,256] | zone3 |
| observation.state | float32[3] = [0,0,0] | Giả định cả ba van ban đầu đóng |
| task | Một string UTF-8 từ instruction | Một lệnh Việt hoặc Anh |
| action | float32[3] từ target_action | Mỗi zone có mức 0–3 |

Ảnh gốc giữ nguyên. Pipeline sau này resize/pad giữ tỷ lệ và kiểm tra chất lượng; tensor shape CHW khác với ảnh uint8 HWC truyền cho writer. API, dtype/scale và chiều camera phải được kiểm chứng bằng round-trip ở bước 5.

State [0,0,0] là **giả định khởi đầu của mẫu độc lập**, không phải cảm biến được đo và không phải trạng thái ESP32 thật. STATE normalization dùng IDENTITY để tránh zero-variance và không bịa noise. Khi có state thật cần version dataset mới với semantics và calibration rõ ràng.

Action v0 có **bốn mức tổng cộng**:

| Giá trị | Tên | Ý nghĩa PoC |
|---|---|---|
| 0 | OFF | Không tưới |
| 1 | LOW | Mức tưới đề xuất thấp |
| 2 | MEDIUM | Mức tưới đề xuất trung bình |
| 3 | HIGH | Mức tưới đề xuất cao |

Van ON/OFF không tự tạo bốn tốc độ lưu lượng. Muốn chuyển 1/2/3 thành thời gian/thể tích cần hiệu chuẩn và safety guard ở giai đoạn phần cứng.

ACTION normalization dùng thống kê train; inference hoàn nguyên trước finite/range checks và lượng tử hóa. Không thay bằng stats val/test. SmolVLA sinh action liên tục, không tự trả về câu lệnh chữ. Padding nội bộ giữ tương thích pretrained weights; chưa kiểm chứng runtime ở bước 0.

## Policy tham chiếu đã implement và kiểm chứng ở bước 3

Mapping wilt→level: 0→0, 1→0, 2→1, 3→2, 4→3. Đây là heuristic đơn điệu của nhóm. Wilt 1 vẫn OFF: templates không được ngầm nói mọi mức héo đều phải tưới.

| Intent | Tham số | Quy tắc |
|---|---|---|
| AUTO | Không | Mapping cho cả ba zone |
| WORST_ONLY | Không | Nếu max wilt ≥2: giữ mọi zone hòa max; nếu không toàn 0 |
| ZONE_ONLY | zone: 1/2/3 | Chỉ giữ mức của zone được chỉ định |
| CAP_LEVEL | cap: 1/2 | Lấy min(mapped level, cap) từng zone |
| EXCLUDE_ZONE | zone: 1/2/3 | Zone bị loại bằng 0, còn lại theo mapping |
| STOP_ALL | Không | Luôn [0,0,0] |

CAP_LOW là cách gọi cap=1, không thêm intent enum thứ bảy. Code chỉ nhận intent enum đúng tên và đúng params; không parse câu tự do hay gọi LLM gán target. Cấu hình mâu thuẫn/ngoài miền, thiếu/thừa params hoặc YAML trùng key đều bị từ chối. STOP_ALL vẫn yêu cầu input teacher hợp lệ; trong tập train không thay thế nút dừng độc lập của hệ thống thật. WORST_ONLY dựa trên mức héo lá quan sát, không có dữ liệu đo đất khô nhất.

Ví dụ teacher wilt=[0,2,4]:

| Intent | Target |
|---|---|
| AUTO | [0,1,3] |
| WORST_ONLY | [0,0,3] |
| ZONE_ONLY(zone=2) | [0,1,0] |
| CAP_LEVEL(cap=1) | [0,1,1] |
| EXCLUDE_ZONE(zone=3) | [0,1,0] |
| STOP_ALL | [0,0,0] |

## Hợp đồng song ngữ

Mỗi record có language=vi hoặc en; task chỉ chứa câu của ngôn ngữ đó. Hai bản dịch tương đương dùng cùng scene, source IDs, intent/params, teacher labels và target, nhưng sample_id/template_id riêng.

Ví dụ cùng target [0,1,0]:

- Việt: Chỉ tưới vùng 2 theo mức quy định nếu đạt ngưỡng héo.
- Anh: Water only zone 2 at its prescribed level if it meets the wilt threshold.

Hai dòng minh họa đầy đủ nằm ở [scene_pair.example.jsonl](examples/scene_pair.example.jsonl). Tất cả paths và source IDs trong file này là placeholder; không có ảnh đi kèm và không phải sample Zenodo đã kiểm chứng. Schema test ở bước 0 chỉ xác nhận cấu trúc, không xác nhận ảnh tồn tại hay policy sinh đúng.

- language: vi/en.
- template_id: mẫu câu cụ thể của một ngôn ngữ.
- template_family_id: liên kết mẫu Việt–Anh tương ứng.
- translation_pair_id: liên kết hai record tương đương cho một scene/intent/wording.
- wording_partition: seen/unseen đối với template family đã dùng train.

Giữ hai bản dịch và family trong cùng source split/wording partition; unseen-val families không vào train, unseen-test families không vào train hoặc val. Hai ngôn ngữ đều có trong train; không gọi unseen wording là zero-shot language transfer.

Bước 3 có 60 family: mỗi intent 6 train + 2 unseen-val + 2 unseen-test, mỗi family một cặp vi/en. Sau điền zone/cap có 132/44/44 câu ở ba pool. Các giao family và câu chuẩn hóa đều rỗng. Renderer không nhận wilt hoặc tạo target; policy tính target từ metadata riêng. Chỉ instruction được chuyển thành model task; structured intent/params/wilt không phải features. Registry giữ cả hai bản dịch cùng pool, còn translation_pair_id gắn scene và kiểm coverage seen-family thực tế thực hiện ở bước 4. Seed cố định chỉ phục vụ tái lập; generator phải phân phối nhiều family thay vì luôn chọn cùng một mẫu.

Bước 4 đã tạo/kiểm translation_pair_id và độ phủ family trên dataset thật, cả theo từng intent/zone/cap. Train chứa đủ 66 family–params ở mỗi ngôn ngữ. Mỗi case val/test có đủ hai bản dịch × seen/unseen với cùng ảnh/teacher/target, family seen phải từng xuất hiện với params đó ở train.

Target pilot: train 3.000 record = 1.500 cặp Việt–Anh; cân bằng ngôn ngữ theo từng intent. Mỗi val/test 600 record = 150 trường hợp × 2 ngôn ngữ × seen/unseen wording. Các bản dịch/paraphrase không tăng số ảnh hoặc quan sát độc lập. Counts cuối phải theo QA thực tế.

## Manifest và schema

[scene_v0.schema.json](schemas/scene_v0.schema.json) là JSON Schema Draft 2020-12. Một dòng JSONL là một record, có các nhóm fields:

| Nhóm | Fields |
|---|---|
| Nhận diện | schema_version, sample_id, scene_id, scene_type |
| Ảnh/nguồn | images.zone1/2/3, source_image_ids[3], source_groups[3] |
| Ngôn ngữ | instruction, language, template_id, template_family_id, translation_pair_id, wording_partition |
| Teacher/target | teacher_annotations.wilt_scores[3], intent, intent_params, target_action[3] |
| Tái lập | split, seed, policy_version, label_source, label_type |
| Provenance bước 4 | case_id, provenance (config/lock/source/encoded/pixel hashes, components), generation_metadata (kind, permutation_of, permutation) |

Paths tương đối POSIX theo data root được ghi trong config; không cho absolute paths, backslash hoặc thành phần .. Schema yêu cầu ba source IDs khác nhau; source_groups có thể lặp vì một plot có nhiều thời điểm.

label_source và policy_version = wilt_proxy_policy_v0; label_type = synthetic_policy_action. Wilt chỉ 0–4; target integer chỉ 0–3. Converter sau này serialize target thành float32.

Schema kiểm tra cấu trúc/type/range/parameter shape. Các field bước 4 là optional trong schema để giữ cặp placeholder bước 0 hợp lệ, nhưng validator dataset thật bắt buộc đủ provenance/case/relation metadata. Bước 4 đã đọc/đối chiếu ảnh/checksums/nguồn, pair/quartet semantics/provenance, recomputed target, quota, coverage và cross-split leakage sau compose. Schema pass riêng không thay thế full dataset QA.

Scene_id nhận diện ordered source triplet; case_id nhận diện scene + intent/params; translation_pair_id nhận diện case + family; sample_id thêm vi/en. permutation dùng one-based: [3,1,2] nghĩa là ảnh mới zone1 lấy ảnh cũ zone3, zone2 lấy zone1, zone3 lấy zone2; params zone được remap và target hoán vị tương ứng. Mọi IDs sinh deterministic từ nội dung. Hash config, schema, model contract, implementation và manifests/semantic hashes nằm trong dataset_lock.json.

## Split, episode và kiểm chứng tiếp theo

Chia theo nhóm ảnh nguồn trước compose. Bước 2 đã khóa train tám nhóm (1.258 ảnh), val 4-228 (189), test 11-46 sau quarantine (341); không thay đổi phương án CSV vì QA/duplicate audit không có xung đột qua split. Val chỉ có bốn ảnh wilt 4 và test chỉ một plot, nên báo cáo kết quả phải thể hiện giới hạn nguồn độc lập này.

Source splits ở data/manifests/source_splits/v0/*.sources.jsonl, có thêm split, duplicate_component_id và source_split_version=soybean_source_split_v0. Các field này chỉ là provenance/QA metadata. split_lock.json pin config/source hashes, seed, policy và output manifest hashes. Không được đổi split hoặc đọc lẫn excluded.sources.jsonl khi sinh scenes.

Giao source IDs, file/pixel hashes, nhóm nguồn, camera, plot và components giữa ba split đều rỗng. Ba near pairs và 42 cặp chưa rõ theo heuristic dHash+pixel đều nằm cùng nhóm/split. Heuristic chưa bao quát mọi crop/rotation; original plot sequences được giữ trọn trong một split để tránh leakage theo thời gian. Xem README/config/report để biết thresholds và review evidence.

Dự kiến một scene+instruction là episode một frame, fps=1 chỉ là serialization. Horizon chunk_size=n_action_steps=1 là cấu hình cần smoke test, chưa được kiểm chứng ở bước 0.

Báo cáo sau train phải tách vi/en × seen/unseen wording, exact action vector/per-zone accuracy, MAE, STOP/tie/permutation và translation agreement/both-correct. Hai bản dịch có thể cùng sai; agreement không phải accuracy.

Bước 0–5 đã hoàn thành. Scene dataset ở data/manifests/v0 có 3.000/600/600 records, 1.500/150/150 cases, 1.440/90/90 ordered scenes và dùng 1.258/173/254 ảnh nguồn khác nhau. Source manifest vẫn giữ 104 excluded; chúng không vào v0. LeRobot roots ở data/lerobot/v0/full/{train,val,test} có 3.000/600/600 singleton episodes; smoke 64 ở data/lerobot/v0/smoke. Tất cả text/IDs/pixels/state/action và masks/stats đã round-trip CPU; không phải model forward/backward. Khả năng train 4 GB cần kiểm chứng bước 7.

Ảnh export resize giữ tỉ lệ LANCZOS 256×192, đệm đen 32 pixel trên/dưới thành 256×256, PNG lossless nhúng Parquet. Metadata bổ sung ở sample_metadata.jsonl/export_info.json/export_lock.json; không thêm teacher/intent/IDs vào features. Raw loader chỉ dùng action delta [0] cho camera [B,3,256,256], state [B,3], action [B,1,3]; factory train upstream còn thêm observation delta [0], camera thành [B,1,3,256,256], state [B,1,3], action giữ [B,1,3]. Cả hai path đã được kiểm bằng batch thật. Đây là history dài 1, không phải nhân episode thành nhiều frame.

Full train stats có bản sao ở full/train_stats.json; normalization_contract.json yêu cầu STATE IDENTITY, ACTION MEAN_STD dùng full train cho cả model/inference/val/test. Native normalizer/inverse đã kiểm; val/test stats chỉ phục vụ QA. Quantiles do upstream aggregate theo episode không phải true global quantiles cho singleton action; không dùng QUANTILES với các số đó, xem action-stats-step5-reference.json. Model contract JSON gốc được giữ nguyên hash do dataset lock pin; flag runtime trong file đó là lịch sử bước 0, bằng chứng CPU mới nằm ở roundtrip-step5 reports. Không sửa pinned contract để làm mất provenance.

