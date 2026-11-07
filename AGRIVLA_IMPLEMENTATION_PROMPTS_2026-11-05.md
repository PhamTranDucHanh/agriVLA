# Bộ prompt triển khai từng bước — AgriVLA-Soybean-v0

Repo: /home/hanhpham-duc/agriVLA. Nhánh: feat/compose-dataset.

**Hiện hành sau bước6:** wording_v1 **3600/720/720**, bảy intents gồm WATER_ALL(level1/2/3) ép[level]*3. Model/config/CPUchecks đãPASS, chưa optimizer/GPUfit. Máy này không NVIDIA; dùng **agri-soybean-vla/GPU_HANDOFF.md** để chuyển source+data+model cache và tiếp tục prompt7 ở repo mới. Tokenizer96; stats wording_v1/full/train_stats.json SHAb06849340aa520815152fd63921b3602ab9278f7c3fd242e600c6a33a086c78f. Không dùng root/stats v0. Prompts0–5 dưới đây là lịch sử v0, không làm lại hoặc overwrite artifacts.

Copy từng khối, lần lượt 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8. Bạn bắt đầu run dài bằng lệnh bàn giao ở bước 8; sau train dùng prompt 9. Mỗi prompt dùng được cho một model mới có quyền đọc cùng workspace.

Nếu đổi máy/checkout, chuyển các tài liệu và artifact local cần thiết; Git không mang theo raw/LeRobot/checkpoints. Bước 0 sẽ đưa các tài liệu bàn giao vào allowlist Git. Bước 7 bắt buộc chạy trên máy GPU 4 GB thực tế. Không bỏ qua bước chưa đạt.

Các CLI trong plan là giao diện cần hiện thực, không phải bằng chứng code đã tồn tại. Mỗi model chỉ làm bước được giao, giữ thay đổi của người dùng và cập nhật AGRIVLA_IMPLEMENTATION_STATUS.md bằng kết quả thật. Nhiệm vụ này là pipeline Soybean; không cần chạy lại bài LIBERO trong AGENTS.md.

## Prompt 0 — Khung dự án, spec, môi trường

```text
Làm BƯỚC 0 của AgriVLA-Soybean-v0 trong /home/hanhpham-duc/agriVLA, nhánh feat/compose-dataset.

Đọc AGENTS.md, AGRIVLA_CODEX_CONTEXT_2026-11-05.md và AGRIVLA_IMPLEMENTATION_PLAN_2026-11-05.md. Kiểm tra Git/file thật trước khi sửa. Chỉ thực hiện bước 0.

Tạo module agri-soybean-vla, package/CLI có --help, DATASET_SPEC.md, setup env riêng agri-soybean-vla/.venv với mode data/train và preflight. Hiện thực/kiểm tra mode data trước; chuẩn bị mode train đúng LeRobot commit trong plan, chưa tải weights hoặc train. Các lệnh data-only không import Torch/LeRobot. Không chép hardcoded path/rmtree từ demo đèn.

Sửa allowlist .gitignore: nhận source/config/tests/docs và các file context/plan/prompts/status; vẫn bỏ qua raw/venv/vendor/cache/generated dataset/checkpoints/outputs. Tạo AGRIVLA_IMPLEMENTATION_STATUS.md ghi trạng thái thực, lệnh đã chạy, artifact và blocker từng bước. Không commit/push.

Giải thích tiếng Việt: module làm gì, state [0,0,0] là giả định gì, bốn mức action là gì, vì sao wilt/intent chỉ ở metadata. Kết thúc bằng file thay đổi, lệnh cụ thể, test thật, artifact xem được và điều kiện sang bước 1. Giữ demo/vendor, không tự làm bước 1.
```

## Prompt 1 — Tải và kiểm tra raw data

```text
Làm BƯỚC 1 tại /home/hanhpham-duc/agriVLA trên feat/compose-dataset. Đọc AGENTS.md, AGRIVLA_CODEX_CONTEXT_2026-11-05.md, AGRIVLA_IMPLEMENTATION_PLAN_2026-11-05.md và AGRIVLA_IMPLEMENTATION_STATUS.md; xác minh CLI/env bước 0 dùng được.

Implement rồi chạy fetch-source/inspect-source theo plan. Bước này được phép tải CSV và ZIP Soybean Zenodo 8256382 (~656 MB) vào data/raw bị ignore nếu chưa có; ưu tiên file local, xác minh checksum, có --dry-run. Không ghi đè/xóa raw sẵn có.

Parse header có cột rỗng. CSV kỳ vọng 1.892 fullID unique: nhãn 0/1/2/3/4 có 524/473/369/216/206 dòng; nhãn 5 có 104 dòng thuộc 11-46. Quarantine 5 với lý do, không tự relabel. Đối chiếu ZIP: paths, missing/extra/duplicate/unreadable ảnh, RGB, dimensions, checksum, timestamp. Nếu nguồn khác số liệu kỳ vọng, báo thay đổi thay vì ép kết quả khớp.

Xuất sources.jsonl, quarantine.csv, source_report.json, contact sheet và tests fixture cho các lỗi quan trọng. Giải thích số dòng CSV khác số ảnh thực sự hợp lệ; không báo 1.788 ảnh đã hợp lệ trước kiểm tra ZIP.

Cập nhật status/context. Bàn giao file, lệnh, counts/checksum, test results, artifact và blocker. Giữ demo/vendor; không commit/push, không train, không tự làm bước 2.
```

## Prompt 2 — Split theo nguồn

```text
Làm BƯỚC 2 tại /home/hanhpham-duc/agriVLA, nhánh feat/compose-dataset. Đọc AGENTS.md, AGRIVLA_CODEX_CONTEXT_2026-11-05.md, AGRIVLA_IMPLEMENTATION_PLAN_2026-11-05.md, AGRIVLA_IMPLEMENTATION_STATUS.md và source report/manifests thật của bước 1. Chưa sinh scene.

Source manifest hiện tại là agri-soybean-vla/reports/source/v0/sources.jsonl: 1.892 record raw, chỉ lọc eligible_v0=true còn 1.788 trước split. Không đưa 104 record annotation=5/wilt_score=null vào split v0. original_path tương đối với module/base_dir; dùng split_group/camera_id/plot_id và kiểm tra near-duplicates từ ảnh/dhash64, không coi dHash bằng nhau tự động là bản trùng.

Implement split theo nhóm nguồn, khóa bằng config/hash. Phương án plan: val=4-228, test=11-46 sau quarantine, train=tám nhóm còn lại. Theo CSV dự kiến train/val/test=1.258/189/341; số cuối theo QA ảnh thật. Kiểm tra camera/plot, duplicate hash, ảnh gần trùng; không cùng source hoặc nhóm gần trùng ở các split khác nhau. Nếu split dự kiến không hợp lệ, cập nhật có lý do trước train, không nới chống leakage.

Xuất split manifests/report, bảng nhãn từng group/split, source/hash intersections, seed/config. Test group isolation, duplicate handling, reproducibility. Giải thích val chỉ có bốn ảnh mức 4, vì sao không chia ngẫu nhiên scene và vì sao một test plot chưa đại diện mọi ruộng.

Cập nhật status/context; bàn giao file, lệnh dry-run/run/test, số liệu thật và artifact. Giữ dữ liệu/demo/vendor; không commit/push, không train, dừng ở bước 2.
```

## Prompt 3 — Policy và câu lệnh tiếng Việt

```text
Làm BƯỚC 3 tại /home/hanhpham-duc/agriVLA trên feat/compose-dataset. Đọc AGENTS.md, AGRIVLA_CODEX_CONTEXT_2026-11-05.md, AGRIVLA_IMPLEMENTATION_PLAN_2026-11-05.md, AGRIVLA_IMPLEMENTATION_STATUS.md và configs hiện có.

Implement deterministic wilt_proxy_policy_v0: {0:0,1:0,2:1,3:2,4:3}; AUTO, WORST_ONLY, ZONE_ONLY, CAP_LEVEL, EXCLUDE_ZONE, STOP_ALL. WORST_ONLY giữ tất cả zone hòa max nếu max>=2, ngược lại toàn 0; template nói rõ tie. STOP_ALL luôn [0,0,0]. Validate intent/zone/cap; từ chối cấu hình mâu thuẫn/ngoài miền, không để LLM tự gán target.

Tạo template tiếng Việt đa dạng và nhất quán ngữ nghĩa. Giải thích ngưỡng héo 2 mới có action; không dùng đất khô nhất. Tách templates train/unseen-val/unseen-test; wilt/structured intent không vào model input. Có explain-policy CLI.

Test 125 tổ hợp wilt với intent/params, expected examples độc lập, STOP/tie/cap/exclusion/permutation/invalid input. In ví dụ [0,2,4] với từng câu/target và giải thích nhãn proxy.

Cập nhật status; bàn giao files, commands, examples, tests và blocker. Giữ demo/vendor; không train, không commit/push, không tự làm bước 4.
```

## Prompt 4 — Compose, QA, preview

```text
Làm BƯỚC 4 tại /home/hanhpham-duc/agriVLA trên feat/compose-dataset. Đọc AGENTS.md, AGRIVLA_CODEX_CONTEXT_2026-11-05.md, AGRIVLA_IMPLEMENTATION_PLAN_2026-11-05.md, AGRIVLA_IMPLEMENTATION_STATUS.md; kiểm tra source splits/policy/templates bước 1–3 đã pass.

Implement compose/validate/preview với seed/config, fail-if-exists, --dry-run. Mỗi scene có ba source image IDs khác nhau từ cùng split; có thể cùng plot khác thời điểm, luôn ghi synthetic_three_zone. Manifest đủ provenance, teacher labels, intent/params, language partition, target, policy version theo plan. Target khởi đầu 3.000 train/600 val/600 test nếu thỏa ràng buộc; val/test có paired seen/unseen wording.

Có contrastive cases, zone permutation đúng, tie/all-OFF/all-high/STOP/exclusion; thống kê quota/reuse. Không đưa nhãn/intent vào ảnh/features model. QA schema, paths, ảnh, action range, recomputed labels, leakage, duplicate trong scene, template isolation, semantic hash reproducibility.

Xuất JSONL, QA report, DATASET_CARD.md và preview HTML local có ba ảnh/lệnh/teacher/target/source. Xem ít nhất 60 mẫu phủ intent/edge case; nếu không xem được ảnh, ghi rõ chưa kiểm chứng phần này.

Giải thích fields bằng mẫu đầy đủ, số ảnh độc lập/scene/record/reuse/zero-action. Cập nhật status; bàn giao files/commands/tests/artifacts thật. Giữ demo/vendor; không train, không commit/push, dừng ở bước 4.
```

## Prompt 5 — LeRobot v3 và đọc ngược

```text
Làm BƯỚC 5 tại /home/hanhpham-duc/agriVLA trên feat/compose-dataset. Đọc AGENTS.md, AGRIVLA_CODEX_CONTEXT_2026-11-05.md, AGRIVLA_IMPLEMENTATION_PLAN_2026-11-05.md, AGRIVLA_IMPLEMENTATION_STATUS.md và dataset QA report. Chỉ convert dataset đã pass QA.

Dataset thật ở agri-soybean-vla/data/manifests/v0: 3.000 train/600 val/600 test records; kiểm dataset_lock.json SHA256 c59005a5d4eea36731051f5b82a9341d9be5ae50cca4aafcad1055743ef16563 rồi chạy validate hiện tại và đối chiếu manifest/config/schema/model-contract/implementation/upstream hashes. Paths ảnh tương đối module/base_dir, không phải thư mục JSONL. Đọc DATASET_CARD và visual_review.json ở data/manifests/v0/review; giữ nguyên splits/translation pairs/family pools, không sinh lại dataset hoặc tự chia episode cho eval.

Hoàn thiện/cài env train riêng với LeRobot commit 7e241bd630a3719a56157a497ce5d08f244784f1, extras smolvla,peft,training; không sửa vendor/env demo cũ. pip check, imports rồi lưu phiên bản đã xác minh. Conversion/round-trip chạy CPU; GPU chưa là điều kiện hoàn thành bước này.

Converter: ba RGB cameras, state float32[3] zeros với semantics mọi van ban đầu đóng, action float32[3], task là instruction tiếng Việt hoặc Anh của từng record, use_videos=false. Teacher/structured intent/case/provenance/generation_metadata chỉ ở metadata, không là features. Một scene+instruction là singleton episode, fps=1 là serialization. Train/val/test ở ba roots riêng, có episode↔sample mapping. Convert 64 mẫu smoke phủ cả vi/en, intents/action, đọc ngược pass rồi convert full.

Round-trip mọi record: text/camera order/pixels sau transform/state/action/IDs; stats finite. Kiểm tra batch action [B,1,3], action_is_pad, sampler không drop hết singleton episodes. Model dùng train stats, không normalize inference bằng val/test stats. Không giả 50 frame lặp.

Giải thích cấu trúc/shapes bằng batch thật; cập nhật status, bàn giao paths/counts/commands/tests/report/blocker. Không tải base weights hoặc train, không commit/push, dừng ở bước 5.
```

## Prompt 6 — Model, fine-tune config, inference/evaluator cơ bản

```text
Làm BƯỚC 6 tại /home/hanhpham-duc/agriVLA trên feat/compose-dataset. Đọc AGENTS.md, AGRIVLA_CODEX_CONTEXT_2026-11-05.md, AGRIVLA_IMPLEMENTATION_PLAN_2026-11-05.md, AGRIVLA_IMPLEMENTATION_STATUS.md và schema/stats LeRobot thật; bước 5 phải pass.

Được tải lerobot/smolvla_base và tokenizer/backbone cần thiết vào cache local bị ignore, ghi revision cụ thể. Implement wrapper train mỏng dùng upstream và fresh LoRA cho ba camera/state[3]/action[3]; không adapter đèn. Giữ padded dimensions tương thích weights, freeze vision/VLM, LoRA r=8 alpha=8, batch 1, chunk_size=n_action_steps=1, model resize thử 256, AMP, LR/schedule theo plan. Không thêm flag chưa có.

Train chỉ đọc train root, dataset.eval_split=0; evaluator dùng held-out roots. Loader dùng saved schema/processor/train stats/base revision; STATE IDENTITY. Audit tiếng Việt/token length không truncation, kiểm tra cả tokenizer processor được lưu. In trainable names/count và resolved config, tránh vô tình full-finetune backbone.

Chuẩn bị infer/evaluate cơ bản: inverse normalization, finite/range checks, quantization rõ rule, reset queue, JSON predictions và metrics test trên fixture. Cấu hình chưa được gọi là fit 4 GB trước bước 7.

Giải thích thay đổi so demo đèn, freeze/LoRA/stats/schema. Cập nhật status, bàn giao files/commands/checks/tests/artifacts. Không optimizer training/run dài, không commit/push, dừng ở bước 6.
```

## Prompt 7 — Kiểm chứng GPU 4 GB

```text
Làm BƯỚC 7 tại repo hiện đang mở trên máy GPU mới, branch feat/compose-dataset. Đọc thêm agri-soybean-vla/GPU_HANDOFF.md và MODEL_TRAINING.md. Dùng dataset WORDING_V1, 3600/720/720 records, tokenizer96, saved processor và train stats hiện hành, fresh LoRA r8/alpha8 đã kiểm CPU. Bước6 chưa có optimizer/backward/GPUfit proof. Không train từ v0. Đọc AGENTS.md, AGRIVLA_CODEX_CONTEXT_2026-11-05.md, AGRIVLA_IMPLEMENTATION_PLAN_2026-11-05.md, AGRIVLA_IMPLEMENTATION_STATUS.md, QA report, resolved config. Kiểm tra GPU/CUDA thực; GPU 4 GB phải nhìn thấy trong môi trường chạy.

Launcher hiện có train-upstream để gọi loop upstream và hook lưu contract/tokenizer, mới hỗ trợ fresh run; phải hiện thực/kiểm resume đủ optimizer/scheduler/RNG/step bằng API upstream thật, không coi checkpoint fixture chưa train là resume test. Được chạy train ngắn: 100 optimizer steps smoke, tối đa 300 steps overfit 32–64 mẫu train đa dạng. Kiểm tra forward/backward/update thật, finite loss/gradient/weights, adapter thay đổi; đo torch allocated/reserved, tổng VRAM nvidia-smi, step time, ảnh/token thật. Kiểm tra memory với prompt dài nhất trong workload.

Subset smoke dùng cùng schema/processor/normalization stats của full train sẽ dùng ở run dài; mọi thay đổi cần ghi config/version và rerun kiểm tra tương ứng.

Kiểm chứng save→fresh-process reload→inference→resume đủ optimizer/scheduler/RNG/step; so cùng seed/noise với tolerance ghi rõ. Overfit là chẩn đoán học khác biệt cơ bản, không phải kết quả test của đồ án.

Nếu OOM, theo thứ tự plan và rerun test sau mỗi thay đổi; không bỏ camera/split, không bịa accumulation/4-bit flags, không kill process lạ. Nếu thiếu GPU hoặc không fit, báo kiểm chứng chưa đạt và blocker, không gọi ready 4 GB.

Cập nhật smoke_report/status/config. Giải thích peak/headroom, tốc độ/ETA, tiny-overfit predictions, lệnh tái lập, giới hạn. Không run dài, không commit/push, dừng ở bước 7.
```

## Prompt 8 — Runbook cho tối nay

```text
Làm BƯỚC 8 tại repo hiện tại trên máy GPU, branch feat/compose-dataset. Dùng wording_v1 3600/720/720 và schema/processor/train stats đã kiểm bước6–7. Đọc AGENTS.md, AGRIVLA_CODEX_CONTEXT_2026-11-05.md, AGRIVLA_IMPLEMENTATION_PLAN_2026-11-05.md, AGRIVLA_IMPLEMENTATION_STATUS.md, dataset QA và smoke_report thực trên GPU 4 GB. Không tuyên bố ready nếu bước 7 chưa pass trên máy train.

Hoàn thiện run_training.sh và TRAINING_4GB.md: preflight, config/dataset hash, dry-run in lệnh thật/resolved config, output mới, logs, save mỗi 500 steps, phân biệt new-run/resume. Dùng upstream trainer và profile đã pass smoke. Kiểm tra CLI/config/resume path; không xóa/ghi đè run cũ.

Từ seconds/step thực, tính ETA và số epoch tương đương. Đề xuất 6.000 steps/batch 1 với 3.600 record ≈1,67 lượt dữ liệu; có thể chọn budget khác theo thời gian thật nhưng phải ghi rõ. Schedule khớp total steps đã chốt, giải thích resume.

Bàn giao lệnh copy chạy được cho install/data/validate/smoke/train/resume/eval/infer, paths logs/checkpoints, OOM troubleshooting; đánh dấu placeholder rõ. Xác minh code/docs được Git nhận còn raw/weights/outputs ignore.

Cập nhật status/context, báo ready hoặc blocker có bằng chứng. Chỉ dry-run/kiểm tra launcher; KHÔNG bắt đầu run dài, tôi tự chạy tối nay. Không commit/push, dừng ở bước 8.
```

## Prompt 9 — Đánh giá sau train

```text
Làm BƯỚC 9 tại /home/hanhpham-duc/agriVLA trên feat/compose-dataset sau khi có checkpoint thật. Đọc AGENTS.md, AGRIVLA_CODEX_CONTEXT_2026-11-05.md, AGRIVLA_IMPLEMENTATION_PLAN_2026-11-05.md, AGRIVLA_IMPLEMENTATION_STATUS.md, run config/log/manifests. Tìm checkpoint qua artifact bàn giao; không đoán đã train xong.

Hoàn thiện/chạy evaluate/infer. Chọn checkpoint bằng validation, đóng băng lựa chọn trước test. Dùng saved schema/base revision/processor/train stats, batch 1; tránh hai process model cùng chiếm GPU 4 GB. Không train thêm hoặc tuning trên test.

Báo exact vector/per-zone accuracy, ordinal MAE, confusion matrices, per-intent/per-level, STOP/tie/WORST_ONLY, permutation, seen/unseen wording. Lưu raw continuous predictions, finite/out-of-range rates và quantized output để clipping không che lỗi. Reset queue và kiểm soát seed/noise cho independent samples.

So baseline OFF/majority từ train, policy oracle kiểm tra generator, image-blanked/shuffled và language ablation; giải thích giới hạn của ablation. Xuất predictions.jsonl, metrics.json, báo cáo dễ hiểu và preview lỗi; ghi số ảnh/plot độc lập và thiếu dữ liệu mức 4 ở val.

Cập nhật status/context bằng kết quả thật. Không bịa confidence, không dùng train accuracy làm test hoặc gọi proxy là tưới tối ưu. Bàn giao lệnh inference ba ảnh+câu Việt→action[3]. Không gửi van, không commit/push; nếu thiếu checkpoint báo artifact còn thiếu.
```
