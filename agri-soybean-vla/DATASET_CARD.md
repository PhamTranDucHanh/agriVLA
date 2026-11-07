# AgriVLA-Soybean-v0 — Dataset card

**Hiện hành sau yêu cầu bước 6:** dataset revision **wording_v1**, 3.600/720/720 records; JSONL ở data/manifests/wording_v1, LeRobot ở data/lerobot/wording_v1/full/{train,val,test}. Bản v0 bên dưới được giữ làm lịch sử và không bị ghi đè. Source triplets/splits/target sáu intent cũ giữ nguyên; 1.088/402/402 instructions được thay bằng văn nói song ngữ, thêm 600/120/120 WATER_ALL records.

WATER_ALL cần level=1/2/3: “Tưới cả 3 vườn mức 2 nhé” → [2,2,2] bất kể wilt. STOP_ALL “Tắt hết.” → [0,0,0]. AUTO và nhóm theo héo vẫn giữ ngưỡng 2. Vườn trong câu lệnh là tên gọi cho zone tổng hợp, không ngụ ý ba ruộng thật đồng thời. Teacher/structured intent/params/revision_metadata chỉ là metadata; features vẫn ba RGB + state[3] và task, target action[3].

Schema v1: schemas/scene_v1.schema.json. Policy extension: configs/policy_extension_v1.yaml. Dataset lock v1: 92111dfa8086eef77b90e9ac47003c43f1847ea07a899e1df3c6232d2f977ba3; parent c59005a5d4eea36731051f5b82a9341d9be5ae50cca4aafcad1055743ef16563 giữ nguyên. Dùng stats wording_v1/full/train_stats.json (SHA256 b06849340aa520815152fd63921b3602ab9278f7c3fd242e600c6a33a086c78f), STATE IDENTITY/ACTION MEAN_STD. Xem MODEL_TRAINING.md và GPU_HANDOFF.md.

---


Dataset tổng hợp ba zone, song ngữ Việt–Anh; QA tự động PASS.

Ảnh nguồn: Study of Leaf Wilt in Soybean Plants, Zenodo 8256382 (https://zenodo.org/records/8256382).
Ảnh giữ nguyên bytes RGB; ba ảnh không phải quan sát đồng thời. Raw wilt 0–4 do nguồn gán, action do wilt_proxy_policy_v0 sinh.
104 ảnh annotation=5 vẫn excluded. License/phạm vi phân phối raw chưa được xác minh đầy đủ; không đóng gói raw vào Git.

| Split | Ảnh nguồn có sẵn/đã dùng | Ordered scenes | Cases | Records | All-zero records |
|---|---:|---:|---:|---:|---:|
| train | 1258/1258 | 1440 | 1500 | 3000 | 684 (22.8%) |
| val | 189/173 | 90 | 150 | 600 | 212 (35.3%) |
| test | 341/254 | 90 | 150 | 600 | 208 (34.7%) |

Một ordered scene là một bộ ba ảnh theo thứ tự zone. Một case là scene + intent/params; train có hai bản dịch, val/test có bốn record (vi/en × seen/unseen).
Các bản dịch, paraphrase, tổ hợp và ảnh cùng plot/thời điểm không phải quan sát thống kê độc lập.
Val có bốn ảnh gốc wilt 4; test chỉ một plot. Reuse/zone/action/quota/template coverage chi tiết ở qa_report.json.

Model nhận ba RGB, task=instruction và state [0,0,0] giả định mọi van ban đầu đóng; teacher wilt/intent/provenance chỉ ở metadata.
Action 0/1/2/3 là OFF/LOW/MEDIUM/HIGH proxy; ngưỡng wilt 2. Không phải lượng nước/nông học đã hiệu chuẩn, không đo đất khô nhất.
Policy mapping {0:0,1:0,2:1,3:2,4:3}; WORST_ONLY giữ mọi zone hòa max nếu max≥2, STOP luôn zeros với input hợp lệ.

Tái lập: compose theo configs/dataset_v0.yaml, seed và hai upstream locks; dataset_lock.json pin config, implementation, schema, manifests và semantic hashes.
Paths ảnh tương đối với module/base_dir, không tương đối với thư mục manifests. Source IDs tra trong source_splits/v0/*.sources.jsonl đã khóa.
Train quota AUTO/WORST/ZONE/CAP/EXCLUDE/STOP = 30/20/15/15/10/10%; eval là panel thiết kế, không phải tần suất vận hành thực.
60 family song ngữ; unseen-val/test tách train và nhau; seen coverage được kiểm trên record train thật.

QA ảnh/hash/schema/policy/pairs/leakage/permutation đã chạy. Visual review lưu riêng ở preview/visual_review.json khi người/agent thực sự xem ảnh.
Đã convert/round-trip LeRobot ở bước 5; chưa kiểm tokenizer, train model, đo GPU 4 GB hoặc chứng minh visual/language grounding/hiệu quả tưới.

## Bằng chứng bàn giao bước 4

Dataset lock SHA256: c59005a5d4eea36731051f5b82a9341d9be5ae50cca4aafcad1055743ef16563.
Dry-run và compose thật có cùng byte/semantic hashes; standalone validation giống QA trong compose. Full suite: 326 PASS.

Đã mở xem 72 record, tương ứng 24 cases, 22 ordered scenes và 55 ảnh nguồn qua contact sheets; không gọi chúng là 72 scene độc lập. Phủ vi/en, seen/unseen, sáu intents, tie/cap/exclusion/permutation/below-threshold và cùng ảnh [4,4,4] với AUTO [3,3,3] đối chiếu STOP [0,0,0].
Mở [preview HTML cuối](data/manifests/v0/review/index.html) và [visual review receipt](data/manifests/v0/review/visual_review.json). HTML được kiểm tra cấu trúc/local links, ảnh được mở bằng image viewer; chưa kiểm bằng browser renderer. Đây không phải expert re-annotation hoặc validation nông học.

[QA report](data/manifests/v0/qa_report.json), [dataset lock](data/manifests/v0/dataset_lock.json), [mẫu JSON thật đầy đủ](reports/example-step4-record.json), [verification](reports/verification-step4.json), [tests](reports/tests-step4.xml). Các artifact data/reports local bị Git ignore; cần tái lập trên máy khác từ config/seed/nguồn đã khóa.

## LeRobot v3 — đã kiểm chứng bước 5, 2026-11-06

[Smoke 64](data/lerobot/v0/smoke), [full train 3.000](data/lerobot/v0/full/train), [val 600](data/lerobot/v0/full/val), [test 600](data/lerobot/v0/full/test). Mỗi record một episode một frame, fps=1 chỉ serialization; ảnh PNG 256×256 letterbox nhúng Parquet, không có MP4. Teacher/intent/IDs/provenance chỉ ở sidecar sample_metadata.jsonl.

Đã kiểm mọi record: text, IDs, camera order/pixels sau transform, state/action/masks, stats finite. Full gồm 12.600 ảnh, sampler giữ đủ 3.000/600/600 frame. Batch factory thực có camera [4,1,3,256,256], state [4,1,3], action [4,1,3], masks [4,1] false. Chưa policy forward/backward.

[Train stats](data/lerobot/v0/full/train_stats.json) và [normalization contract](data/lerobot/v0/full/normalization_contract.json): STATE IDENTITY, ACTION MEAN_STD chỉ từ full train. Đã chạy native normalizer/inverse trên batch ba split, sai số action bằng 0 trong batch đã kiểm. Không dùng val/test stats hoặc episode-aggregated quantiles để normalize model. [Verifier độc lập toàn bộ](reports/roundtrip-step5-full.json), [hướng dẫn conversion](CONVERSION.md).
