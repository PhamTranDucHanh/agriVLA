# AGRIVLA — PROJECT CONTEXT & CODEX IMPLEMENTATION HANDOFF

**Ngày tổng hợp:** 2026-11-05  
**Ngôn ngữ làm việc:** Tiếng Việt; tên biến/code và thuật ngữ kỹ thuật có thể dùng tiếng Anh.  
**Repo:** https://github.com/PhamTranDucHanh/agriVLA  
**Nhánh triển khai hiện tại:** `feat/compose-dataset`, được người dùng tạo từ `dev` ngày 2026-11-05 bằng `git checkout -b feat/compose-dataset`. Mọi thay đổi tiếp theo thực hiện trên nhánh này; không sửa trực tiếp `main` hoặc `dev`, giữ nguyên các demo hiện có.  
**Mục đích tài liệu:** Chuyển giao bối cảnh, thiết kế, giới hạn dữ liệu và cách đánh giá. **Bước 0–6 đã PASS_CPU, gồm browser HTML 4.5 và wording_v1/model preparation**. Bước7–9 phải tiếp tục trên máy GPU mới. Kết quả thực tế nằm trong [AGRIVLA_IMPLEMENTATION_STATUS.md](AGRIVLA_IMPLEMENTATION_STATUS.md).

## Hiện hành — đọc trước khi tiếp tục trên máy GPU (2026-11-06)

**Bước6 đã PASS_CPU**, chưa optimizer training hoặc chứng nhận4GB. Người dùng xác nhận máy này không có NVIDIA và đây là bước cuối làm trên máy này. Handoff máy mới: **agri-soybean-vla/GPU_HANDOFF.md**; model/inference chi tiết: MODEL_TRAINING.md. Tiếp theo làm bước7, không làm lại0–6 hoặc chạy LIBERO/demo đèn.

Người dùng yêu cầu câu văn nói song ngữ và xác nhận **WATER_ALL(level=1/2/3) ép [level]*3 bất kể wilt**. Không tự hiểu câu thiếu mức là AUTO hoặc tự chọn mức. STOP_ALL “Tắt hết.” vẫn[0,0,0]. Sáu intent cũ giữ ngưỡng2/tie/cap/exclusion. Vườn là alias zone tổng hợp.

**Dataset hiện hành wording_v1**: JSONL data/manifests/wording_v1 (3600/720/720); LeRobot data/lerobot/wording_v1/full/{train,val,test}; smoke64 phủ7intents/3force levels. Giữ source triplets/splits/pair/family pools và target4200records cũ; đổi1088/402/402 instructions, thêm600/120/120force records. 5040records/2160cases/1620triplets nhưng chỉ1685ảnh nguồn độc lập; 104label5excluded. Lock mới **92111dfa8086eef77b90e9ac47003c43f1847ea07a899e1df3c6232d2f977ba3**, parentv0 **c59005a5d4eea36731051f5b82a9341d9be5ae50cca4aafcad1055743ef16563** giữ nguyên. Không gộp hai versions. Policy mới wilt_proxy_with_operator_override_v1, schema scene_v1; teacher/structured intent/revision_metadata vẫn metadata-only.

**Stats bắt buộc:** wording_v1/full/train_stats.json SHA **b06849340aa520815152fd63921b3602ab9278f7c3fd242e600c6a33a086c78f**. STATE IDENTITY; ACTION MEAN_STD full train cho cả held-out/inference. Không dùng stats v0/val/test/quantiles singleton. Actual trainer factory chỉ mở train3600, eval_split=0, eval_dataset=None; camera[B,1,3,256,256], state/action[B,1,3], masksfalse, padding nội bộ32. Encoder CPUhook thật256. Train launcher có train-only export hash gate và explicit CUDA --execute guard.

**Model đã tải/pin:** smolvla_base **5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6**; backbone/tokenizer SmolVLM2 **7b375e1b73b11138ff12fe22c8f2822d8fe03467**; LeRobot **7e241bd630a3719a56157a497ce5d08f244784f1**, clean. .venv-train, frozen vision/VLM, freshLoRA r8/alpha8 only371328/450417504params (0.08244%), no light adapter. Batch1, horizon1, resize256, CUDAAMP, LR3e-4/warmup100/decay6000→3e-5; candidate only. Tokenizer **96**, maxVI81, zero truncation5040records; saved processor exact276unique prompts. 64 would truncate483VIrecords. Prepared bundle models/prepared/wording_v1 contains schema/stats/config/tokenizer/provenance/initial_adapter/inputproof/locks; initial adapter is UNTRAINED.

**Checks:** 530data testsPASS+1intentional native skip; native2PASS, handoff recheck2PASS. Full independent round-trip5040records/15120images; strict base/fresh adapter reload, CPU no-grad forward/inference, saved checkpoint fixture, raw3images+VIinstruction and trainer dry-runPASS. Raw untrained example predicted[1,2,2] for target[2,2,2]; **technical PASS does not mean learned policy or Vision competence**. Inverse normalization uses saved train stats; finite/range checks, clip[0,3] then floor(x+0.5), reset queue per independent sample. Basic infer/evaluate and JSON metrics fixtures exist; full checkpoint selection/baseline/ablation remains9. No optimizer/backward/resume/GPUfit proof yet.

**Browser is current:** http://127.0.0.1:8765/?view=samples&page_size=48, 5040records/105pages48, 840WATER_ALL, raw1892/quarantine104. Default browse.sh now browse-revision; --replace checks same repo/user before restart. Full HTTP/Chromium desktop/mobile checks and screenshots reports/browser/step6/, no JS/HTTP errors. If server session stopped, run browse.sh; Ctrl+Shift+R. CLI convert/verify default configs/lerobot_wording_v1.yaml, legacy requires explicit lerobot_v0 config; contract defaults v1, --revision v0 retains old file.

**Transfer:** outputs/handoff/agri-soybean-step6.tar +.sha256 +.manifest.json; includes source/context/data/reports/pinned modelcache, excludes venv/vendor/prepared hostpaths/outputs/volatilelogs. Reinstall using setup.sh, verify copies and prepare new bundle to rebase paths. Next model follows GPU_HANDOFF prompt7: actualGPU backward/update/VRAM/save/reload/resume and tiny overfit. smoke-train/resume/run_training wrappers are not yet implemented; do not invent flags/claim done. Preserve demo/vendor/raw/v0, no commit/push; stop6 on current host.

Master report: agri-soybean-vla/reports/verification-step6.json; status section6 lists paths/counts/commands and limitations.

## Lịch sử parent v0 — bước0–5 (không dùng làm default train hiện hành)

**Cập nhật bước 5 — 2026-11-06:** dataset input lock vẫn c59005a5d4eea36731051f5b82a9341d9be5ae50cca4aafcad1055743ef16563; đã chạy lại validator/check đầy đủ hashes trước conversion, giữ split/pair/template families/raw. Env data/browser .venv giữ nguyên; env ML **.venv-train** đã cài đúng LeRobot commit 7e241bd630a3719a56157a497ce5d08f244784f1, extras smolvla,peft,training; Torch 2.7.1+cu126, torchvision 0.22.1+cu126, TorchCodec 0.5, transformers 5.5.4, PEFT 0.21.2, datasets 4.8.5. Pip check/preflight/imports PASS, full versions ở requirements-train.lock.txt và reports/environment/train-step5-verified.json. CUDA unavailable chỉ WARN ở bước này. Không sửa vendor demo/upstream.

**LeRobot artifact thật:** agri-soybean-vla/data/lerobot/v0/smoke (64 train records = 32 cặp vi/en) và full/{train,val,test} với **3.000/600/600 episodes = frames**. Mỗi record một frame/episode, fps=1 chỉ serialization. Ba RGB giữ tỉ lệ LANCZOS 256×192 + đệm đen 32 pixel trên/dưới → 256×256; PNG lossless nhúng Parquet, không MP4 hoặc folder images độc lập. State float32[3] zeros; action float32[3]; task đúng instruction UTF-8 vi/en. Teacher/intent/case/provenance ở sample_metadata.jsonl, không ở features. Export info/locks pin source/config/code/Parquet/mapping. Đã convert smoke + đọc lại PASS rồi mới full; verifier độc lập đã so mọi text/IDs/state/action và **12.600 ảnh** pixel-exact, stats finite/action mean/std đúng source.

**Batch cần nhớ cho bước 6:** raw loader chỉ action delta [0] cho camera [B,3,256,256], state [B,3], action [B,1,3]. Factory train upstream resolve observation delta [0] nữa nên camera **[B,1,3,256,256]**, state **[B,1,3]**, action [B,1,3]; đã kiểm B=4 thật, mọi camera/state/action_is_pad [4,1] false, sampler giữ đủ 3.000/600/600. Đây là history=1, không giả nhiều frame. Model forward/tokenizer chưa được kiểm, không bỏ qua history axis khi nối processor/model.

**Stats cho model:** full/train_stats.json (SHA256 8cb4f96fb861d6d48f911d2d6956e5a0e632406d3b676025ed2cda97d94756e7), normalization_contract.json: STATE IDENTITY, ACTION MEAN_STD chỉ từ full train cho cả inference/val/test. Native normalizer/inverse đã kiểm ba split, batch error=0. Không dùng meta stats val/test hoặc smoke; helper từ chối. Quantile upstream aggregate theo episode nên không phải global action quantiles singleton; không dùng QUANTILES với các số đó (reports/action-stats-step5-reference.json giải thích). Root val/test có thể mang nhãn container LeRobot info.splits.train; source split thật xem root + export_info.source_split + metadata, tuyệt đối không tự chia episodes cho eval.

**Bằng chứng/đi tiếp:** data regression 363 PASS + 1 module native skip có chủ đích; native .venv-train đã chạy riêng 2 PASS. Reports: validate-step5-input.json, roundtrip-step5-smoke-factory.json, roundtrip-step5-full.json, verification-step5.json; hướng dẫn CONVERSION.md. CLI convert --profile smoke/full và verify-lerobot --root dùng .venv-train (convert --dry-run có thể dùng .venv). Config conversion mới configs/lerobot_v0.yaml, không đổi pinned dataset_v0/schema/model_contract JSON. Flags runtime trong model_contract là lịch sử bước 0; report mới là bằng chứng CPU, không phá hash cũ. **Bước 6** mới tải base weights/tokenizer/fresh LoRA, dùng full/train + train stats, dataset.eval_split=0 và audit cả vi/en/no truncation/resize/history axes. Chưa tải base weights, chưa train/commit/push; dừng bước 5.

**Trang HTML preview đã có — BƯỚC 4.5 PASS:** bước 4 giữ preview tĩnh 72 record tại `agri-soybean-vla/data/manifests/v0/review/index.html`; bước 4.5 đã làm thêm **trang HTML localhost để duyệt toàn bộ dataset**, không giới hạn 72 record. Có đủ 4.200 record và 1.892 ảnh nguồn, lọc/tìm tiếng Việt bỏ dấu/phân trang/đi thẳng sample số, xem bản dịch/seen–unseen/JSON/provenance và mở ảnh/thư mục gốc bằng link HTTP local. Entry: `bash agri-soybean-vla/scripts/browse.sh`, URL mặc định `http://127.0.0.1:8765/`; chỉ đọc, QA dataset trước mở port. Tab thư mục hiện absolute paths thật, copy path và link /files/images/, /files/manifests/, /files/raw/… Browser HTTP không tự mở file manager; hướng dẫn xdg-open nằm trong docs. Source UI tại `agri-soybean-vla/src/agri_vla/web/browser.html`, browser.css/browser.js; server tại browser.py. **343 tests PASS**; đã kiểm tất cả 4.200 IDs qua HTTP khớp thứ tự manifest, đủ 1.892 nguồn/104 quarantine; Playwright/Chromium đã thử desktop/mobile, trang 350 và mẫu #4200, kết hợp filters, cặp dịch, ảnh gốc, copy path, tải JSON và folder 1.892 file, không có lỗi JS/HTTP trong kịch bản. Screenshots/reports: `agri-soybean-vla/reports/browser/step45/`; hướng dẫn từng thao tác: `agri-soybean-vla/BROWSE_DATASET.md`. Dataset lock vẫn `c59005a5d4eea36731051f5b82a9341d9be5ae50cca4aafcad1055743ef16563`, không sửa raw/splits/labels/ảnh. Công cụ UI test cài riêng /tmp, runtime không cần Playwright/npm/Torch/GPU. Server phải đang chạy mới dùng được link localhost; nếu phiên tool đã dừng, chạy lại script. Bước 4.5 chỉ làm browser; conversion xem cập nhật bước 5 ở trên, chưa train/commit/push.

**Cập nhật sau bước 4 (lịch sử):** compose/validate/preview đã chạy thật, dataset ở agri-soybean-vla/data/manifests/v0/: train/val/test=3.000/600/600 records, 1.500/150/150 cases và 1.440/90/90 ordered scenes. Ảnh nguồn thực dùng 1.258/173/254 trong 1.258/189/341 ảnh đã split; reuse cases max=22/32/22, toàn OFF=22,8%/35,33%/34,67%. Hai upstream locks giữ nguyên. Dataset lock SHA256 c59005a5d4eea36731051f5b82a9341d9be5ae50cca4aafcad1055743ef16563; dry-run/run cùng byte/semantic hashes. Có train.jsonl/val.jsonl/test.jsonl, qa_report.json, dataset_lock.json, DATASET_CARD.md, preview HTML/contact sheets. QA source/schema/teacher/recomputed policy/pairs/quartets/quota/template coverage/permutation/leakage đã pass. Train dùng đủ 66 family–params ở mỗi ngôn ngữ; eval seen đối chiếu train thật, unseen đúng pool. 104 label 5 không vào dataset. Paths ảnh tương đối module/base_dir; ảnh gốc nguyên bytes. Metadata mới: case_id, provenance và generation_metadata; không đưa vào model features. **Bước 5** phải validate dataset lock hiện tại trước conversion/round-trip, chỉ chọn instruction + ba ảnh + state zeros và target_action cho loss; không train hoặc tự chia split. Visual review và test evidence chi tiết trong status.

**Cập nhật sau bước 3 (lịch sử):** wilt_proxy_policy_v0 đã implement mapping {0:0,1:0,2:1,3:2,4:3}, sáu intents và validation strict. WORST_ONLY giữ mọi zone hòa max nếu max≥2, STOP_ALL luôn zeros cho input hợp lệ. Registry configs/language_v0.yaml có 60 family Việt–Anh, 220 câu sau điền params: 132 train, 44 unseen-val, 44 unseen-test; hai bản dịch giữ cùng family/pool. 282 tests PASS, gồm 1.375 policy cases theo expected độc lập và 8.250 kiểm tra permutation. CLI explain-policy/audit-policy chạy data-only. Artifacts ở agri-soybean-vla/reports/policy/v0/: truth table, template catalog, examples, report, lock; policy lock SHA256 c6c0d5d2be39fb61581a10aa39757e03b840518ddf6c210a372c5fb2853b4d11. Không thay source split bước 2, không sinh scene/train tại thời điểm bước 3. Các yêu cầu kiểm hash/family/translation_pair ở bước 4 đã được triển khai như cập nhật trên. Wilt/structured intent chỉ là teacher metadata; instruction không chứa đáp án/intent enum. Đây là proxy ordinal levels, chưa hiệu chuẩn lượng nước hoặc kiểm chứng khả năng hiểu ngôn ngữ của model.

**Kế hoạch triển khai chi tiết:** [AGRIVLA_IMPLEMENTATION_PLAN_2026-11-05.md](AGRIVLA_IMPLEMENTATION_PLAN_2026-11-05.md).
**Bộ prompt giao từng bước:** [AGRIVLA_IMPLEMENTATION_PROMPTS_2026-11-05.md](AGRIVLA_IMPLEMENTATION_PROMPTS_2026-11-05.md).
**Mục tiêu trước run dài:** dataset ba zone đầy đủ/QA/LeRobot và cấu hình fresh LoRA được smoke test trên đúng máy GPU 4 GB. Chưa có kết quả train/VRAM mới tại thời điểm lập kế hoạch.
**Cập nhật yêu cầu ngôn ngữ:** dataset nông nghiệp phải có cả tiếng Việt và tiếng Anh. Đây là yêu cầu hiện tại cho bước 3 và toàn bộ compose/conversion/train/eval; ngôn ngữ giải thích và tài liệu làm việc vẫn là tiếng Việt.

---

## 0. BẮT ĐẦU TỪ ĐÂU / TASK KỲ VỌNG CHO CODEX

1. Đọc repo **thật** tại `feat/compose-dataset` (nhánh nguồn: `dev`) và xác nhận phiên bản code/LeRobot, nhất là các file ở phần 2. Kiểm tra `git branch --show-current` trước khi sửa code. Đừng chỉ tin tài liệu handoff nếu source tại thời điểm thực thi đã đổi.
2. Giữ nguyên demo LIBERO và demo điều khiển đèn. Tạo module mới dành cho **Soybean 3-Zone VLA PoC**; ưu tiên không sửa file cũ, trừ khi thực sự cần và giải thích rõ.
3. Ưu tiên hiện thực tuần tự: kiểm tra Soybean raw data → split theo nhóm nguồn → scene generator ba ảnh → instruction generator → rule-based action labeler → validators → JSONL manifests → LeRobot v3 converter → smoke test `chunk_size=1` → train/eval/inference. **Chưa cần triển khai thật ESP32 3 van trong bước dataset đầu tiên.**
4. Mọi script có seed/config, có CLI, hỗ trợ `--dry-run` khi nguy cơ ghi đè, có kiểm thử tự động. Không tự tải 655 MB dữ liệu, train dài hoặc flash firmware trừ khi môi trường/người dùng cho phép. Không xóa directory dữ liệu đã tồn tại.
5. Luôn phân biệt: `raw factual label (wilt 0–4)` / `synthetic reference-policy action (0–3)` / `model prediction` / `physically validated irrigation action`.
6. Trong phản hồi đầu tiên, Codex nên kiểm tra repo rồi liệt kê phát hiện, các bất định kỹ thuật và kế hoạch thay đổi file. Khi triển khai, làm từng milestone với artifact và lệnh test có thể tái lập.

## 1. BỐI CẢNH ĐỒ ÁN TỐT NGHIỆP

**Tên đề tài chính thức (đã cung cấp trong trao đổi):** “Nghiên cứu và xây dựng hệ thống tưới tiêu nông nghiệp thông minh điều khiển bằng mô hình Vision-Language-Action (VLA) cho trạm tưới đa khu vực.”

### Yêu cầu của đồ án

- Khảo sát hệ thống tưới, 3 van phân luồng, độ ẩm đất, nhiệt độ, lưu lượng và kiến trúc VLA.
- Thu thập dữ liệu ảnh camera + cảm biến từ các khu vực canh tác.
- Tích hợp/fine-tune VLA để hiểu lệnh ngôn ngữ kết hợp hình ảnh và trạng thái, từ đó đề xuất hành động tưới từng vùng.
- Kết nối VLA (inference server) đến controller ESP32-S3 (có thể qua gateway Raspberry Pi), điều khiển 3 van độc lập; tương lai mở rộng lưu lượng, thời gian, cảnh báo.
- Có safety guard/validator trước khi thực thi; đo độ chính xác perception/action, latency, reliability, lượng nước, trạng thái sau tưới trong kiểm thử thật.

### Kiến trúc dự kiến (không phải toàn bộ đã hiện thực)

```text
             Camera / 3 zone images
                       |
Natural-language command + sensor measurements + actuator state
                       |
                       v
                SmolVLA on server
                       |
           predicted continuous action[3]
                       |
       postprocessing / 4-level quantization
                       |
               Safety / Guard Layer
                       |
                 Gateway (optional)
                       |
                 ESP32-S3 controller
                       |
                Valve 1 / 2 / 3
                       |
         flow/moisture feedback (future v1)
```

Phân biệt **đề xuất hành động** từ VLA và **thực thi an toàn** từ controller. SmolVLA không được trực tiếp bỏ qua giới hạn an toàn. Một van ON/OFF thông thường không tự điều chỉnh lưu lượng liên tục; bốn mức v0 có thể chuyển thành thời gian cấp nước được hiệu chuẩn khi có phần cứng, không phải bốn tốc độ dòng khác nhau mặc định.

## 2. TRẠNG THÁI REPO ĐÃ XÁC MINH 2026-11-05

**Repo:** `PhamTranDucHanh/agriVLA`, default `main`.  
**GitHub compare `main...dev`:** `dev` ahead 2 commits, behind 0 tại thời điểm xác minh. `main` SHA `83883af3fcca9402f392157a869ac307881cf607`; `dev` SHA đã từng xác minh `041ad1e33901d32852538a52fccf1e4ee1e15a8a` (Codex kiểm tra lại HEAD khi clone).

### Cập nhật local sau khi Codex đọc repo ngày 2026-11-05

**Cập nhật sau bước 2:** đã implement/chạy `split --dry-run` và run thật, giữ nguyên assignment dự kiến: train 1.258 ảnh/tám nhóm, val 189 ảnh/group `4-228`, test 341 ảnh/group `11-46`; 104 record nhãn 5 excluded. Output ở `agri-soybean-vla/data/manifests/source_splits/v0/`, gồm `train.sources.jsonl`, `val.sources.jsonl`, `test.sources.jsonl`, `excluded.sources.jsonl`, `split_lock.json`, `split_report.json`, group stats và review panels. Lock SHA256 `3be04c67ed375c76e4a97d425416e3a890bd1c060ce1167b9bbd5f16d6fcaa8d`, seed 42. Không giao source ID/file hash/pixel hash/group/camera/plot/duplicate component giữa splits. dHash≤8 có 3.645 candidate pairs; so ảnh RGB/gray xác nhận ba near pairs, 42 chưa rõ đều cùng source group/split, 3.600 cặp distinct theo policy. Không có candidate qua split trong bán kính đã khai báo; đã xem 19 cặp chẩn đoán gồm cả 12 cặp qua split gần nhất ngoài bán kính. Heuristic không bao quát mọi crop/rotation. 69 tests PASS. Bước 4 phải xác minh lock/manifest hashes và giữ nguyên split, không đọc excluded hoặc tự chia lại ảnh. `duplicate_component_id`/`source_split_version` là metadata, không đưa vào model. Val chỉ bốn ảnh gốc wilt 4, test chỉ một plot; không coi ảnh cùng plot là các quan sát thống kê độc lập. Chưa sinh scene/action, train/commit/push.

**Cập nhật sau bước 1:** đã chạy `fetch-source`/`inspect-source`, xác minh MD5 CSV và ZIP. ZIP có đúng 1.892 ảnh JPEG RGB 640×480; không thiếu/thừa/hỏng/non-RGB, không duplicate chính xác theo encoded/pixel SHA256. Có 1.788 ảnh đủ điều kiện v0 sau QA và 104 ảnh annotation=5 tại group `11-46` giữ quarantine. Timestamp tên ảnh: 2019-08-09 đến 2019-09-22, timezone chưa biết. 46 tests PASS; đã xem 6 contact sheets (72 mẫu). Source manifest: `agri-soybean-vla/reports/source/v0/sources.jsonl`, **chỉ chọn `eligible_v0=true` khi làm bước 2**; `original_path` tương đối với module/base_dir, `annotation_raw` giữ nhãn gốc, `wilt_score=null` cho nhãn ngoài miền. Có `dhash64` để hỗ trợ kiểm tra near-duplicates nhưng chưa grouping/split. Raw và reports vẫn ignore, demo/vendor giữ nguyên; chưa train/commit/push.

**Cập nhật sau bước 0:** module `agri-soybean-vla/` đã có CLI `preflight`/`contract`, spec, JSON Schema và cặp ví dụ Việt–Anh, setup riêng data/train và tests. Đã dựng `.venv` bằng CPython 3.12.16 local, cài data lock có hashes, data preflight/pip check PASS và 22 tests PASS. Mode train chỉ dry-run; chưa cài LeRobot/Torch, chưa tải weights/train. `.gitignore` đã nhận các tài liệu/source và vẫn bỏ qua env/data/cache/outputs. Các ghi nhận bên dưới là **lịch sử audit ban đầu**, xem status để biết trạng thái hiện tại.

- Đã xác nhận nhánh làm việc hiện tại là `feat/compose-dataset`, tạo từ `dev`; code nền lúc audit ở commit `041ad1e`. Các mô tả và URL `dev` bên dưới là nguồn tham chiếu của code cũ, không phải chỉ dẫn chuyển về `dev` để triển khai.
- Đã đọc file handoff, README, scripts cài đặt/chạy LIBERO, toàn bộ scripts demo đèn, Web UI, HTTP client, firmware ESP32 và cấu hình dataset/checkpoint. Module Soybean ba zone chưa tồn tại; M0 mới hoàn thành phần đọc/audit, chưa có `DATASET_SPEC.md` hay pipeline mới.
- Parquet demo đèn có các trường ảnh `bytes`/`path` và dữ liệu PNG nhúng. Vì vậy, không có thư mục `images/` riêng chưa đủ để kết luận dataset thiếu ảnh. Vẫn cần kiểm tra đọc ngược bằng `LeRobotDataset` khi có môi trường.
- Installer riêng `my-vla-light-demo/scripts/install.sh` gọi `pip install -r requirements.txt`, nhưng repo hiện không có file đó. Installer gốc pin LeRobot v0.6.1 và các dependencies LIBERO; installer riêng demo đèn chưa kiểm tra commit sau clone. Cần xác minh dependency PEFT/SmolVLA trước khi tái lập.
- Phiên làm việc hiện tại chưa có `.venv` hay vendor checkout `lerobot/`; không tìm thấy `python3.12`, `nvidia-smi` hoặc `/dev/dxg` tại các vị trí đã kiểm tra. Chưa chạy inference, training, rollout LIBERO hoặc thử phần cứng; kết quả demo trong README là kết quả được ghi lại trước đây.
- Tại audit ban đầu, handoff bị `.gitignore` bỏ qua bởi `*`. Bước 0 đã allowlist context/plan/prompts/status; hiện các file hiển thị untracked và chưa commit/push.
- Hướng triển khai tiếp theo trên `feat/compose-dataset`: M0 spec → M1 kiểm tra CSV/ZIP → M2 split nguồn rồi ghép ba zone → M3 instruction/reference-policy labels → M4 validation/JSONL/LeRobot. Smoke test action horizon một bước và train/eval thực hiện sau đó.
- Khi lập kế hoạch, đã tải và thống kê CSV thật: 1.892 fullID khác nhau, 10 nhóm camera/plot; nhãn 0/1/2/3/4 lần lượt có 524/473/369/216/206 dòng; **104 dòng mang nhãn 5**, đều ở nhóm `11-46`. CSV MD5 khớp `538bbf95130b3bc637acfcbed503de64`. Có 1.788 dòng trong miền 0–4, chưa kiểm tra đủ ảnh ZIP. V0 phải giữ nhãn 5 trong quarantine, không tự sửa thành 4. Xem plan để biết split đề xuất và tiêu chí kiểm chứng.

### `main`
- README/demo LIBERO Object task 9: robot Franka nhặt hộp nước cam và đặt vào giỏ.
- Tải checkpoint **đã fine-tune** `HuggingFaceVLA/smolvla_libero`; script `scripts/run_orange_juice.sh` đánh giá mô phỏng; đây không phải training của bài toán tưới.
- `scripts/install.sh` cài môi trường/LeRobot; repo hướng tới LeRobot **v0.6.1** (pinned commit `7e241bd630a3719a56157a497ce5d08f244784f1`).

### `dev`: thêm `my-vla-light-demo/` — PoC Language→Action
File thực tế đáng xem:

- `my-vla-light-demo/scripts/create_light_dataset.py`: tạo LeRobot dataset giả lập bật/tắt đèn.
- `my-vla-light-demo/datasets/light_switch_demo/meta/info.json`, `stats.json`, `data/...parquet`, `tasks.parquet`: schema/metadata dataset cũ.
- `my-vla-light-demo/models/light_lora_1000/pretrained_model/train_config.json`: tham số huấn luyện ghi trong checkpoint.
- `my-vla-light-demo/models/light_lora_1000/pretrained_model/adapter_config.json`, `adapter_model.safetensors`: adapter LoRA.
- `my-vla-light-demo/scripts/demo_light_vla.py`: load base + LoRA, pre/postprocessor, inference, ánh xạ action sang ON/OFF.
- `my-vla-light-demo/scripts/light_web_server.py`, `web/*`: Web UI nhập lệnh.
- `my-vla-light-demo/scripts/esp32_client.py`, `firmware/esp32_light/src/main.cpp`: HTTP GET `/state`, POST `/light`, điều khiển LED (không phải 3 van).
- `my-vla-light-demo/README.md`: hướng dẫn PoC.

### PoC đèn thực sự làm gì

- Dataset giả lập **20 episodes × 60 frames = 1.200 frames**, `fps=10`, 10 kiểu câu lệnh ON/OFF tiếng Việt/Anh; hai ảnh dummy 256×256 **giống nhau giữa ON/OFF**; state[8] nhiễu ngẫu nhiên; action[7] mà `action[0]` gần `+1` (ON) hay `-1` (OFF), sáu chiều còn lại không có ý nghĩa.
- Cấu hình: `lerobot/smolvla_base`; `use_peft=true`, LoRA `r=8`, `alpha=8`; ghi `steps=1000`, `batch_size=1`; `freeze_vision_encoder=true`, `train_expert_only=true`, `train_state_proj=true`; `chunk_size=50`, `n_action_steps=50`, `num_steps=10` cho flow matching; input 2 ảnh + state[8] + task; output action[7].
- `demo_light_vla.py` hiện load **mẫu ảnh cố định** từ dataset và state zeros, thay mỗi câu lệnh; gọi `preprocessor` → `select_action` → `postprocessor`, lấy action[0], Python quyết định `>=0` ON, `<0` OFF. Có thể gửi HTTP sang ESP32 nếu bật flag; video là animation giả lập, không phải feedback camera hay phép đo LED thật.
- **Vì ảnh dummy không phân biệt trạng thái nên PoC này CHƯA chứng minh mô hình sử dụng Vision, chưa biết héo lá, chưa biết tưới.** Checkpoint đèn không phải checkpoint cho soybean và không được dùng như mô hình đã biết tưới.
- Tài liệu repo mô tả có image files, nhưng tree `dev` công khai khi kiểm tra **không hiển thị thư mục images** tương ứng. Audit local đã thấy dữ liệu PNG nhúng trong Parquet; không mặc định coi thiếu thư mục ảnh là lỗi. Codex phải kiểm tra dataset hoàn chỉnh có đọc được trong clean clone hay không; đừng khẳng định đã chạy nếu chưa test.
- **CẢNH BÁO:** `create_light_dataset.py` hiện đặt ROOT cứng tại `~/projects/smolVLA/datasets/light_switch_demo` và gọi `shutil.rmtree(ROOT)` nếu tồn tại. **Không sao chép hành vi xóa này sang pipeline mới**; ưu tiên output riêng, fail-if-exists hoặc `--overwrite` xác nhận rõ.
- `train_config.json` có path tuyệt đối theo máy người trước; không tái sử dụng path cũ. File config ghi 1000 steps không tự chứng minh toàn bộ quá trình train/log đã chạy; kiểm chứng riêng nếu cần.
- Firmware ESP32 mới điều khiển GPIO LED, LAN HTTP không auth; chưa điều khiển ba van và chưa đo feedback vật lý.

## 3. DATASET VISION ĐÃ CHỌN CHO PoC

**Zenodo record:** https://zenodo.org/records/8256382  
**CSV annotations:** https://zenodo.org/records/8256382/files/dataAnns_goodFiles.csv?download=1  
**ZIP ảnh:** https://zenodo.org/records/8256382/files/full_data.zip?download=1  
**Preview CSV:** https://zenodo.org/records/8256382/preview/dataAnns_goodFiles.csv  
**Bài báo liên quan:** https://doi.org/10.3390/agriculture15060614  
**GitHub nghiên cứu:** https://github.com/ARoS-NCSU/Soybean-Leaf-Wilt-Classification

Dataset **Study of Leaf Wilt in Soybean Plants**:

- Bản Zenodo công bố khoảng **1.892 ảnh RGB** đậu tương, `full_data.zip` ~655.7 MB, CSV `dataAnns_goodFiles.csv` có tên ảnh/plot hoặc camera/nhãn. Bài báo liên quan phân tích tập con **1.788 ảnh**; KHÔNG đồng nhất hai con số nếu chưa thống kê lại CSV.
- Miền nhãn héo **được Zenodo mô tả** là ordinal 0,1,2,3,4; `0` héo ít nhất, `4` héo nhiều nhất; do chuyên gia chấm dựa trên quan sát tán cây. **CSV thật còn có 104 dòng nhãn 5 ngoài miền v0**, như cập nhật audit ở phần 2. Không tự suy ý nghĩa nhãn 5. Đây **không phải** số đo % độ ẩm đất, nhu cầu tưới, thừa nước hay lượng nước cần cấp.
- Theo mô tả trong bài báo liên quan, ảnh thu ngoài ruộng với góc camera cố định nghiêng ~45 độ và khoảng 1.52 m phía trên tán; ảnh khoảng 640×480; chụp theo chu kỳ. Lưu ý đối chiếu khác biệt **timestamp năm 2019 trong CSV** và **mô tả thí nghiệm 2020 trong paper** trước khi khẳng định toàn bộ ảnh có cùng nguồn/ngày.
- CSV dùng các cột `treatment_camera`, `fullID`, `Annotation`. Cần thống kê thực tế bằng code: số dòng, unique file, phân bố 0–4, số plot/camera, timestamp, ảnh có thực trong ZIP và các ảnh gần trùng.
- License của bản ghi Zenodo được thảo luận trước đây là chưa xác minh đầy đủ; việc tải công khai không tự động cho phép đóng gói/phân phối lại dữ liệu. Codex cần kiểm tra bản ghi nguồn và chỉ giữ raw data trong thư mục được gitignore.
- Dataset KHÔNG có: 3 camera chụp đồng thời một trạm, 3 vùng tưới thực tế, soil moisture được đồng bộ, weather/humidity chính xác, nhãn overwatering, van, flow, action hoặc câu lệnh tiếng Việt.
- **Lợi thế cho PoC:** một loại cây, góc gần camera cố định, ảnh thật và nhãn héo do chuyên gia tạo; phù hợp kiểm tra Vision hơn dataset không nhãn.

### Các dataset đã nghiên cứu khác (hiện chỉ là phương án phụ)
- AgriField-40K: https://huggingface.co/datasets/dtu-pcas/AgriField-40K — 39.963 ảnh RGB tổng hợp từ nhiều nguồn, 512², không còn nhãn; tốt cho nghiên cứu thị giác, không phải action supervision.
- Maize UAV water stress: https://zenodo.org/records/19108517 (mini) / https://zenodo.org/records/22062459 (bản đầy đủ mới trong cuộc trao đổi) — stress phân loại/mask dựa phần lớn vào pseudo-label quang phổ; camera UAV, không đồng nghĩa với lượng tưới.
- Multi-Sector Irrigation: https://data.mendeley.com/datasets/c837v6p8ph/2 — dữ liệu sensor/telemetry/nhật ký tưới các vùng, không phải ảnh RGB đồng bộ với soybean.
- Lettuce RGB/Thermal: https://data.mendeley.com/datasets/294zk6k5wf/2 — thí nghiệm hai cây xà lách có ảnh và độ ẩm, quy mô nhỏ.
- LAB-Lettuce: https://data.mendeley.com/datasets/b82wmb5yj6/1 — ảnh fixed-camera theo thời gian; không phải dataset action VLA hoàn chỉnh.

**Quyết định hiện tại:** Ưu tiên Soybean Zenodo 8256382 làm nguồn Vision cho **synthetic 3-zone PoC**; không lạc sang dataset khác trừ khi cần giải quyết blocker hoặc người dùng yêu cầu.

## 4. Ý TƯỞNG DATASET MỚI — AGREED PoC v0

Đầu vào dự kiến: **3 ảnh RGB**, mỗi ảnh đại diện một zone giả lập, + **lệnh tưới tiếng Việt hoặc tiếng Anh** + `observation.state` nếu cần thiết cho LeRobot (chưa có sensor thật). Đầu ra: **vector action 3 chiều**, mỗi chiều nhận một trong bốn mức **0,1,2,3**:

| Giá trị | Ngữ nghĩa PoC | Phần cứng thật về sau |
|---|---|---|
| 0 | OFF / không tưới | Van đóng |
| 1 | LOW / tưới nhẹ | Lượng/thời gian ít, phải hiệu chuẩn |
| 2 | MEDIUM | Lượng/thời gian trung bình, phải hiệu chuẩn |
| 3 | HIGH | Lượng/thời gian lớn hơn, phải hiệu chuẩn |

**Đã chốt: 4 trạng thái tổng cộng, bao gồm OFF**. Không nhầm thành 4 mức dương + OFF = 5 trạng thái. 

**Hợp đồng song ngữ Việt–Anh:** mỗi record có một câu thuộc `language=vi|en`; hai bản dịch tương đương dùng cùng scene/intent/params/target và được liên kết bằng `translation_pair_id`. Cân bằng train 50/50 trong mỗi intent. Dùng `template_family_id` để tách seen/unseen cho cả hai ngôn ngữ cùng lúc, và `wording_partition` để ghi seen/unseen; không đưa các metadata này vào model input. Pilot 3.000 train = 1.500 cặp dịch; mỗi val/test 600 record = 150 trường hợp × 2 ngôn ngữ × seen/unseen wording. Đánh giá riêng vi/en, wording generalization và độ nhất quán giữa cặp dịch; hai câu cùng dự đoán vẫn có thể cùng sai.

Một mẫu manifest trung gian đã thiết kế trong hội thoại:

```json
{
  "sample_id": "scene_000132_auto",
  "scene_type": "synthetic_three_zone",
  "crop": "soybean",
  "images": {
    "zone1": "images/soy_001.jpg",
    "zone2": "images/soy_102.jpg",
    "zone3": "images/soy_340.jpg"
  },
  "teacher_annotations": {
    "wilt_scores": [0, 2, 4]
  },
  "instruction": "Tưới các khu vực có biểu hiện héo.",
  "intent": "AUTO",
  "target_action": [0, 1, 3],
  "label_source": "wilt_proxy_policy_v0",
  "label_type": "synthetic_policy_action",
  "split": "train"
}
```

Tên ảnh `soy_001.jpg`... trong ví dụ **là placeholder**, KHÔNG phải tên đã xác minh tồn tại trong Zenodo. Sau khi parse CSV, phải dùng `fullID` thật hoặc đường dẫn từ ZIP.

### INPUT vs LABEL vs METADATA

- **ĐƯA VÀO MODEL:** ba ảnh RGB, `task=instruction`, và optional state **hợp lệ, không tiết lộ nhãn**.
- **TARGET CHO LOSS:** `target_action` → `action` float32[3] trong LeRobot.
- **CHỈ DÙNG ĐỂ SINH NHÃN / DEBUG / EVAL:** `teacher_annotations.wilt_scores`, `intent`, `sample_id`, `label_source`, `label_type`, `split`, source image provenance.
- **CẤM LABEL LEAKAGE:** không đưa `wilt_scores` [0,2,4] trực tiếp vào `observation.state` nếu mục đích là chứng minh Vision; nếu làm thế model có thể suy ra action mà không cần nhìn ảnh. Không cho `intent` dạng nhãn cấu trúc vào input cùng câu lệnh nếu mục tiêu là thử Language understanding.
- **Sơ đồ:** `3 RGB images + instruction (+ valid state) -> VLM/visual tokens/state features -> Action Expert -> continuous action[3] -> inverse normalization -> quantization {0,1,2,3} -> safety guard -> actuator`.
- Không nhét JSON nguyên văn vào prompt chatbot của SmolVLA. Phải parse và serialize đúng field type/scheme của LeRobot.
- Trong v0, 3 ảnh có thể được ghép từ 3 plot/thời điểm khác nhau: **synthetic_three_zone**, không được gọi là quan sát đồng thời hay khẳng định hệ thống thực đã đo cảm biến.

## 5. ACTION LABEL GENERATION — PHÂN BIỆT KHOA HỌC VÀ GIẢ ĐỊNH

**Nguồn khoa học:** nhãn `wilt_score` do chuyên gia chấm; FAO-56 và tài liệu quản lý nước cây đậu tương giải thích tưới thực tế phải dựa trên nước vùng rễ, loại đất, giai đoạn cây, thời tiết, lượng nước/bơm, v.v.:
- https://www.fao.org/4/X0490E/x0490e0e.htm
- https://content.ces.ncsu.edu/north-carolina-soybean-production-guide/10-water-management-in-soybeans

**Không có tài liệu nào trong trao đổi chứng minh bảng số héo → lượng tưới 0–3 như một quy tắc nông học đã được xác nhận.** V0 chỉ là **proxy policy / heuristic có tính đơn điệu** để học hành động từ nhãn trực quan. Không khẳng định hiệu quả sinh trưởng hay mức nước tối ưu.

Quy tắc ban đầu đã thảo luận:

```python
WILT_TO_LEVEL = {
    0: 0,  # không biểu hiện héo -> OFF
    1: 0,  # héo rất nhẹ -> OFF trong proxy v0
    2: 1,  # mức đề xuất LOW
    3: 2,  # MEDIUM
    4: 3,  # HIGH
}
```

Các mức 0–3 là **ngưỡng action giả lập/đề xuất**, không phải lít/giây. Trường hợp lá héo do nhiệt nóng/bệnh/úng nước có thể không nên tưới. V1 phải thay bằng soil-moisture/soil-water-deficit policy đã hiệu chuẩn, giới hạn tưới, có feedback.

### Instruction / intent generation

Tạo từ **intent có ngữ nghĩa rõ ràng**, sinh nhiều mẫu câu tiếng Việt và tiếng Anh tương đương bằng template/paraphrase được kiểm tra, rồi tính action từ **intent + wilt_scores + các tham số**; KHÔNG để LLM gán nhãn action tự do.

Intent dự kiến:
- `AUTO`: “Tưới những vùng có biểu hiện héo” → áp dụng bảng cho cả 3.
- `WORST_ONLY`: “Chỉ tưới vùng héo nặng nhất” → chọn vùng max wilt nếu đạt điều kiện; quy tắc xử lý hòa điểm phải được chốt và ghi trong policy.
- `ZONE_ONLY`: “Chỉ tưới vùng 2 nếu cây có biểu hiện héo” → chỉ giữ mức của zone được yêu cầu, khác 0 khi mức proxy cho phép.
- `CAP_LOW` / `CAP_LEVEL`: “Tưới nhẹ những vùng đang héo” → min(level, cap).
- `STOP_ALL`: “Dừng tưới toàn bộ” → [0,0,0], ưu tiên dừng.
- `EXCLUDE_ZONE`: “Tưới vùng cần nước, trừ vùng 3” → 0 vùng bị loại, khác theo policy.

Lệnh phải phản ánh đúng thông tin nguồn: nên dùng “vùng có biểu hiện héo” thay vì “đất khô nhất”, vì Soybean **không có nhãn độ ẩm đất**. Câu “tưới vùng 2” không tự động an toàn ở v1 nếu vùng đó đủ/ứ nước; cần policy xử lý lệnh trực tiếp/trái điều kiện.

Tạo **contrastive pairs**:
- Cùng 3 ảnh nhưng lệnh khác → action khác.
- Cùng một lệnh nhưng đổi 3 ảnh/trạng thái → action khác.
- Hoán đổi ảnh giữa zone1/2/3 → action phải hoán vị tương ứng (ngoại trừ các lệnh khóa vùng).
- Có STOP_ALL/NO_ACTION/những trường hợp all healthy; tránh dataset thiên về ON.

### Ví dụ đúng cho v0

```text
3 ảnh có wilt scores: [0, 2, 4]
- AUTO -> [0, 1, 3]
- WORST_ONLY -> [0, 0, 3]
- CAP_LOW -> [0, 1, 1]
- ZONE_ONLY(zone2) -> [0, 1, 0]
- STOP_ALL -> [0, 0, 0]
```

## 6. PIPELINE TẠO DATASET — THIẾT KẾ TRƯỚC KHI TRAIN

```text
Zenodo full_data.zip + dataAnns_goodFiles.csv
         |
         v
prepare_soybean.py (parse, inspect, validate, manifest)
         |
         v
Split raw image GROUPS (plot/camera/sequence) into train/val/test
         |
         v
generate_scenes.py (choose 3 independent source images per scene)
         |
         v
generate_instructions.py (intent + controlled Vietnamese and English paraphrases)
         |
         v
generate_labels.py (deterministic reference policy)
         |
         v
validate_dataset.py (hard assertions, leakage, distribution)
         |
         v
train.jsonl / val.jsonl / test.jsonl
         |
         v
convert_to_lerobot.py -> LeRobotDataset
         |
         v
train SmolVLA base + fresh LoRA
         |
         v
evaluate.py / infer_agri.py -> quantized action[3]
```

### Tránh đánh giá giả

- **PHẢI split từ ảnh gốc theo plot/camera/phiên trước** rồi mới tổ hợp scene và paraphrase; không chia random sample đã sinh, vì cùng ảnh có thể xuất hiện cả train và test.
- Kiểm tra unique original images; một source image chỉ nằm trong một split; xét ảnh gần trùng, cùng chuỗi thời gian; có thể dùng group-based split theo plot/camera, chọn cân bằng theo wilt level nếu khả thi.
- Có **held-out language templates/paraphrases** riêng để đánh giá tổng quát với cách nói chưa gặp, đồng thời giữ ý nghĩa lệnh như nhau.
- Random seed cố định, log manifest/source provenance, cách gán nhãn, cảnh nào chứa ảnh nào và lớp nào, thống kê các intent/action/level theo split.
- Cần tránh chọn cùng ảnh ba lần vào một scene (trừ khi cố tình làm control case); cân bằng vị trí zone, level, intent, số lần reuse ảnh.
- 1.892 ảnh nguồn nhưng có thể tạo 3.000–10.000 mẫu kết hợp; **đó không phải 10.000 ảnh độc lập**. Số lượng này mới là target thử nghiệm, không phải đảm bảo hội tụ.

## 7. SMOLVLA — BẢN CHẤT VÀ ÁNH XẠ TECHNICAL

### Model hoạt động thế nào

1. `observation.images.camera1/2/3` (RGB tensors [3,H,W]) → preprocessing (resize/pad/normalize) → visual encoder của VLM → visual embeddings.
2. `task` (câu tiếng Việt hoặc tiếng Anh) → tokenizer → language tokens/VLM context.
3. `observation.state` (nếu có) → state projection → context. V0 KHÔNG được làm giả độ ẩm đất hay nhét nhãn héo vào state.
4. **Action Expert** (flow matching) học dự đoán trường vận tốc trên các action bị nhiễu có điều kiện ảnh + ngôn ngữ + state; target của loss là hành động chuẩn từ reference policy.
5. Inference dùng nhiều bước denoising để sinh **continuous action vector**; LeRobot postprocessor khôi phục scale gốc; script lượng tử hóa 0,1,2,3 và safety validate.
6. SmolVLA **không tự trả về text “mở van”**. Đó là phần mapping hậu xử lý/actuator client mà nhóm tự viết.

### Repo cũ vs phiên bản PoC mới

| | Demo cũ | PoC mới dự kiến |
|---|---|---|
| Images | 2 dummy RGB | 3 Soybean RGB thật (3 synthetic zones) |
| Language | ON/OFF light | irrigation instructions/intent groups |
| State | float32[8] random/zero | minimal/optional valid state, subject to library compatibility |
| Action | float32[7], dùng dim0 | float32[3], each 0..3 |
| Dataset | synthetic light episodes | synthetic three-zone decision samples |
| Base | `lerobot/smolvla_base` | `lerobot/smolvla_base` |
| Adapter | `light_lora_1000` | **fresh** `agri_lora_v0` |
| Action horizon | `chunk_size=50` | **test** `chunk_size=1`, `n_action_steps=1` |
| Execution | on/off LED via HTTP | predict 3 intensity levels; hardware only later |

**Important implementation checks:**
- LeRobot actually supports `LeRobotDataset.create(...)` / `add_frame({... "task": ...})` / `save_episode()` / `finalize()`; verify current installed v0.6.1 signatures.
- Train script can infer input/output features from dataset metadata when creating policy; inference script must restore matching feature schema before applying adapter, and load correct dataset normalization stats.
- `SmolVLAPolicy` uses `prepare_state` accessing `observation.state`; a no-sensor v0 **may still need a placeholder state feature** for compatibility. Candidate: `[valve1_current, valve2_current, valve3_current]`, all zeros if **explicitly modeling all valves initially closed**. If constant, ensure normalization is safe (`IDENTITY` or verified zero-variance handling) and no label leakage. Don’t invent measured moisture.
- `chunk_size=1`, `n_action_steps=1` is a **technical proposal**, not yet validated on actual environment. Run smoke test; if unsupported, design documented workaround without faking sequence data. Old 50-step chunk must not be copied blindly.
- One independent static scene can be one single-step synthetic episode **if converter and trainer confirm correct behavior**. `fps=1` only a serialization convention, **not an assertion that source images were acquired at 1 Hz**. Metadata must describe synthetic episodic nature. For future v1, collect genuine time-ordered multi-step episodes.
- Base `SmolVLA` VLM is pretrained; current LoRA preset `freeze_vision_encoder=true`, `train_expert_only=true`: new Action Expert can exploit existing visual features, **but does not prove encoder can recognize soybean wilt reliably**. No need to immediately rewrite pretrained VLM. Test first, consider vision adaptation only if ablation/perception tests show need.
- PEFT/LoRA adapter from light demo is not reused; train fresh for action[3] / three image features / data stats.
- Actions may be stored as 0.0/1.0/2.0/3.0. Train-time action normalization and inference unnormalization should be consistent; quantize only **after** unnormalization. Numeric quantization does not convert SmolVLA into 4-class classifier; compare an ordinary ordinal/classification baseline as appropriate.
- Need test model/tokenizer separately with Vietnamese and English; don't assume multilingual generalization without evaluation. Template split distinguishes seen vs unseen wording.

### Sketch LeRobot schema (for discussion; verify against source)

```python
features = {
    **{
        f"observation.images.camera{i}": {
            "dtype": "image",
            "shape": (3, 256, 256),
            "names": ["channels", "height", "width"],
        }
        for i in (1, 2, 3)
    },
    "observation.state": {
        "dtype": "float32", "shape": (3,),
        "names": ["valve1_current", "valve2_current", "valve3_current"],
    },
    "action": {
        "dtype": "float32", "shape": (3,),
        "names": ["zone1_level", "zone2_level", "zone3_level"],
    },
}
```

```python
# conceptual — code implementation must fill images, handle splitting and
# ensure correct episode semantics and no overwrite of existing dataset.
dataset.add_frame({
    "observation.images.camera1": image1_uint8_HWC,
    "observation.images.camera2": image2_uint8_HWC,
    "observation.images.camera3": image3_uint8_HWC,
    "observation.state": np.array([0., 0., 0.], dtype=np.float32),
    "task": sample["instruction"],
    "action": np.array(sample["target_action"], dtype=np.float32),
})
dataset.save_episode()
```

**Không đưa** `teacher_annotations.wilt_scores` / `intent` / `split` vào input training batch như feature mô hình trừ khi cố tình thiết kế ablation riêng. Giữ chúng trong manifest/metadata.

## 8. CẤU TRÚC FILE GỢI Ý (CHƯA TỒN TẠI)

```text
agriVLA/
├── README.md                         # Existing main/LIBERO demo
├── scripts/                          # Existing base scripts
├── my-vla-light-demo/                # Existing light PoC: preserve
└── agri-soybean-vla/                 # New proposal; choose name consistently
    ├── README.md
    ├── configs/
    │   ├── policy_v0.yaml           # Intent rules, WILT_TO_LEVEL, tie policy, seed
    │   ├── dataset_v0.yaml          # Paths, splits, camera size, generation counts
    │   └── train_v0.yaml            # Training params for installed LeRobot
    ├── data/
    │   ├── raw/                     # Local only; Zenodo ZIP + CSV, gitignored
    │   ├── manifests/               # JSONL train/val/test + provenance
    │   └── lerobot/                 # Generated; potentially large, gitignored
    ├── scripts/
    │   ├── prepare_soybean.py
    │   ├── split_sources.py
    │   ├── generate_scenes.py
    │   ├── generate_instructions.py
    │   ├── generate_labels.py
    │   ├── validate_dataset.py
    │   ├── convert_to_lerobot.py
    │   ├── train_agri.py            # only if a wrapper is needed
    │   ├── infer_agri.py
    │   └── evaluate.py
    ├── tests/
    │   ├── test_policy_v0.py
    │   ├── test_data_integrity.py
    │   ├── test_splits.py
    │   └── test_lerobot_conversion.py
    └── models/                    # checkpoints (gitignored)
```

Đây là **gợi ý tổ chức**, Codex có thể gom scripts theo module Python sạch hơn. Không tạo file rỗng chỉ để giống hình. Ưu tiên CLI rõ ràng, config nhất quán, README hướng dẫn chạy.

## 9. MILESTONES + ACCEPTANCE CRITERIA

### M0 — Audit & science spec
- [ ] Chạy `git status`, `git log`, `git branch`, xác nhận đang ở `feat/compose-dataset`, nguồn code v0.6.1 và env.
- [ ] Đọc file light demo hiện có; xác định API/normalization/schema issues.
- [ ] Tạo tài liệu `DATASET_SPEC.md` diễn giải data semantics, label origin, 4 levels, limitations, project scope, selected Zenodo.
- [ ] Khẳng định rõ không dùng action proxy để điều khiển tưới thật mà không có guard/soil feedback.

### M1 — Ingest Soybean raw data (không cần GPU)
- [ ] Đọc CSV, validate header `treatment_camera`, `fullID`, `Annotation`; kiểm tra 0–4 và số dòng.
- [ ] So khớp ZIP file names, phát hiện duplicate/missing/unreadable images.
- [ ] Thống kê per-plot, per-label, timestamp, actual dimensions, file sizes, report CSV/JSON.
- [ ] Tạo source manifest với `source_image_id`, `plot_id`, `wilt_score`, original_path, split_group, source_url, etc.
- [ ] Không tự tải hoặc bung ZIP trong repo Git tracked; hướng dẫn người dùng để dữ liệu tại local path.

### M2 — Group split + 3-zone generator
- [ ] Split theo source plot/camera/sequence **trước** scene synthesis; xử lý imbalance khoa học, preserve seed.
- [ ] Sinh scenes gồm 3 images khác nhau từ cùng split; hoán vị zone; có các trường hợp nhiều mức bằng nhau/all healthy/all wilt.
- [ ] Tạo thống kê phân bố; kiểm tra không cross-split image leakage.

### M3 — Language + policy labeler
- [ ] Chốt policy_v0 có bảng wilt→level + các intent; xử lý tie, cap, exclusion, stop, contradictory instructions rõ ràng.
- [ ] Sinh paraphrase song ngữ Việt–Anh đa dạng, nhưng ngữ nghĩa nhất quán và có held-out templates.
- [ ] Reference policy deterministic, unit tested với ví dụ `[0,2,4]`.
- [ ] Ghi `label_source=wilt_proxy_policy_v0`, `label_type=synthetic_policy_action` trong từng record; không lẫn với measured irrigation.

### M4 — Dataset validation + LeRobot conversion
- [ ] JSONL full schema, đường dẫn ảnh resolve, checksum/source provenance.
- [ ] Data QA: shape/range/image readability/unique splits/action[3] integer levels/no contradictions/no label leakage.
- [ ] LeRobot dataset có 3 RGB camera, optional safe state, `task` text, action float32[3], đúng feature metadata.
- [ ] `LeRobotDataset` có thể đọc ngược; mọi sample giữ đúng lệnh/ảnh/nhãn, `dataset.meta.stats` hợp lệ.
- [ ] Thử một tập rất nhỏ trước; không nhân frame tĩnh giả thành episode 50 steps.

### M5 — SmolVLA train smoke test
- [ ] Load `lerobot/smolvla_base` và custom config phù hợp; thử `chunk_size=1`, `n_action_steps=1`, LoRA.
- [ ] Một batch train/eval thật: forward loss finite, backward, optimizer step, checkpoint save/reload.
- [ ] Check CPU/CUDA/memory, PEFT target modules, state normalization, dataset statistics, three image keys.
- [ ] Chỉ sau smoke test thành công mới chọn step/epochs/hyperparams cho run dài.

### M6 — Inference + evaluation
- [ ] `infer_agri.py` nhận 3 ảnh test + lệnh, xây input chính xác, postprocess, quantize 0–3, output `[z1,z2,z3]` và confidence? **Không tự chế confidence** nếu model không cung cấp/calibrate. Có thể log continuous prediction và khoảng cách tới mức gần nhất.
- [ ] Baselines: rule-policy oracle (kiểm thử generator), image-blanked VLA, language-ablated VLA, simple classifier/ordinal baseline nếu khả thi.
- [ ] Metrics: exact 3-zone vector accuracy, per-zone level accuracy, MAE ordinal, confusion matrices, STOP_ALL correctness, worst-zone selection, robustness to zone permutation, unseen images/plots/unseen paraphrases; tách vi/en × seen/unseen wording, translation-pair agreement và tỷ lệ cả hai bản dịch cùng đúng.
- [ ] Giới hạn: chỉ chứng minh **imitation of synthetic proxy action**, chưa chứng minh irrigation efficiency hoặc lượng nước phù hợp ngoài thực tế.

### M7 — Future integration, OUT OF SCOPE for immediate first milestone
- [ ] Thu dữ liệu 3 zone thực có timestamps đồng bộ, moisture calibrated, temperature/humidity, tank/flow/valve states, irrigation outcomes.
- [ ] Thiết kế nhãn mức nước thật theo agronomic reference (FAO-56 / soybean-specific water requirements / flow calibration).
- [ ] ESP32-S3 nhận lệnh 3 van, authenticated transport, command acknowledgment, idempotency, watchdog, safe-off, maximum-on timers, hard interlocks, fail-safe for loss of network/power.
- [ ] Safety guard độc lập model, human override/STOP_ALL, pump capacity limitation; closed-loop sensor feedback.
- [ ] Evaluation on real hardware with water volume, soil moisture outcome, latency, reliability, false irrigation / under/over watering, baselines.

## 10. CRITICAL DO-NOT-MISINTERPRET RULES

1. **Soybean wilt 0–4 ≠ volumetric soil moisture ≠ physical watering duration/liters.** Proxy mapping v0 is a research hypothesis/labeling heuristic; document it as such.
2. **3 randomly composed images ≠ camera observations taken simultaneously.** Mark scenes synthetic; never fabricate synchronized environmental sensors.
3. **Generated action labels are imitation targets of a rule**, not expert-measured optimal irrigation. Evaluate honestly against reference and on independent samples.
4. **Task instructions are generated** and must be semantically consistent; avoid leaking structured `intent` directly into model input for language test.
5. **Original `wilt_scores` are teacher metadata**, not model features in Vision-dependent experiment.
6. **Dataset train/val/test must split by source plot/camera/sequence before synthetic combinations.** No same original image across splits.
7. `light_lora_1000` is a different task/model head schema; **train new adapter**, do not use it as agri checkpoint.
8. `chunk_size=1` is **proposed and needs verification**, because original SmolVLA learned action sequences. Do not blindly keep 50 or fabricate 50-step episodes.
9. A four-level policy over ON/OFF valves is **not an adjustable flow-rate device**. To physically implement 1/2/3, define calibrated duration/volume and state machine with fail-safe.
10. Action Expert outputs continuous values, not inherently discrete classes. **Normalize during training; unnormalize then quantize; validate safety.**
11. Do not overclaim Vietnamese language generalization, visual grounding, sensor usage, hardware efficacy, model convergence or train steps without tests.
12. Preserve working demos and avoid destructive scripts, especially hardcoded `shutil.rmtree` from old dataset script.
13. Respect licenses; public Zenodo link alone is not blanket redistribution permission.
14. Avoid arbitrary forced irrigation on severe wilt without checking other stress causes in v1. `STOP_ALL` and safety override must be authoritative.

## 11. CONCEPTUAL PSEUDOCODE

```python
# data generation; this is an outline, not runnable production code
raw_images = validate_source(soybean_csv, soybean_zip)
splits = group_split(raw_images, by=["plot_or_camera_group"])
for split, images in splits.items():
    for scene in generate_three_zone_scenes(images, seed=42):
        wilt_scores = scene.wilt_scores          # teacher only
        for intent in selected_intents(scene):
            instruction = render_paraphrase(intent, split=split)
            action = reference_policy(wilt_scores, intent)
            write_jsonl({
                "images": scene.image_paths,
                "teacher_annotations": {"wilt_scores": wilt_scores},
                "intent": intent.name,
                "instruction": instruction,
                "target_action": action,
                "label_source": "wilt_proxy_policy_v0",
                "scene_type": "synthetic_three_zone",
                "split": split,
                "source_image_ids": scene.source_image_ids
            })

# converter
for sample in load_jsonl(...):
    dataset.add_frame({
        "observation.images.camera1": load_rgb(sample.zone1),
        "observation.images.camera2": load_rgb(sample.zone2),
        "observation.images.camera3": load_rgb(sample.zone3),
        "observation.state": current_valve_state_or_approved_placeholder,
        "task": sample.instruction,
        "action": np.asarray(sample.target_action, dtype=np.float32),
    })
    dataset.save_episode()  # if verified singleton-episode design

dataset.finalize()

# inference
obs = build_inputs(three_new_images, new_user_instruction, current_state)
continuous = postprocessor(policy.select_action(preprocessor(obs)))
discrete_levels = np.clip(np.rint(continuous), 0, 3).astype(int)
validated = safety_guard(discrete_levels, live_state)
# Hardware execution only in real-system phase after validation
```

**Phân biệt ground-truth thế nào?** `wilt_score` là nhãn đánh giá của dataset gốc; `action = reference_policy(wilt_score, intent)` là **nhãn dẫn xuất do nhóm thiết kế**, không trực tiếp có trong Zenodo.

## 12. THÔNG TIN NÊN HIỂN THỊ TRONG BÁO CÁO/KẾT QUẢ CODEX

Codex khi hoàn thành từng milestone cần tóm tắt:
- Chính xác file tạo/sửa, hành vi của mỗi file.
- Lệnh chạy dry-run/smoke test, test results, số mẫu/thống kê thật.
- Các giới hạn chưa xử lý hoặc assumption (đặc biệt paper/CSV/time/license, action mapping).
- Không báo “trained”, “works on ESP32”, “model understands Vision” nếu chưa chạy kiểm chứng.
- Cách tái lập dataset từ seed/config và link raw data, không cần commit raw image ZIP hoặc model weights lớn.
- Next recommended milestone rõ ràng.

---

## 13. LINKS THAM KHẢO / SOURCE OF TRUTH

**Team repo:**
- Nhánh triển khai hiện tại: `feat/compose-dataset` (đã xác nhận local; chưa xác minh đã push remote). Các URL `dev` bên dưới tham chiếu code nền đã audit.
- https://github.com/PhamTranDucHanh/agriVLA/tree/dev
- https://github.com/PhamTranDucHanh/agriVLA/blob/dev/my-vla-light-demo/scripts/create_light_dataset.py
- https://github.com/PhamTranDucHanh/agriVLA/blob/dev/my-vla-light-demo/scripts/demo_light_vla.py
- https://github.com/PhamTranDucHanh/agriVLA/blob/dev/my-vla-light-demo/models/light_lora_1000/pretrained_model/train_config.json
- https://github.com/PhamTranDucHanh/agriVLA/blob/dev/my-vla-light-demo/scripts/esp32_client.py
- https://github.com/PhamTranDucHanh/agriVLA/blob/dev/my-vla-light-demo/firmware/esp32_light/src/main.cpp

**Official model/framework:**
- https://huggingface.co/lerobot/smolvla_base
- https://huggingface.co/blog/smolvla
- https://github.com/huggingface/lerobot/tree/v0.6.1
- https://github.com/huggingface/lerobot/blob/7e241bd630a3719a56157a497ce5d08f244784f1/src/lerobot/policies/smolvla/modeling_smolvla.py
- https://github.com/huggingface/lerobot/blob/7e241bd630a3719a56157a497ce5d08f244784f1/src/lerobot/datasets/lerobot_dataset.py

**Original soybean dataset & paper:**
- https://zenodo.org/records/8256382
- https://zenodo.org/records/8256382/preview/dataAnns_goodFiles.csv
- https://doi.org/10.3390/agriculture15060614
- https://github.com/ARoS-NCSU/Soybean-Leaf-Wilt-Classification

**Agronomy reference (for future physically grounded policies):**
- https://www.fao.org/4/X0490E/x0490e0e.htm
- https://content.ces.ncsu.edu/north-carolina-soybean-production-guide/10-water-management-in-soybeans

---

## 14. ĐOẠN YÊU CẦU CODEx NÊN NHẬN SAU KHI ĐỌC FILE

> Hãy đọc toàn bộ `AGRIVLA_CODEX_CONTEXT_2026-11-05.md` và audit nhánh `feat/compose-dataset` của repo (được tạo từ `dev`). Mọi triển khai tiếp theo thực hiện trên `feat/compose-dataset`; không sửa trực tiếp `main` hoặc `dev`. Mục tiêu hiện tại là tạo **AgriVLA-Soybean-v0**: một pipeline sinh dataset ba zone tổng hợp từ ảnh Soybean Zenodo 8256382, mỗi sample gồm ba ảnh, câu lệnh song ngữ Việt–Anh đa dạng có intent rõ, action `[0,1,2,3]` trên mỗi zone theo `wilt_proxy_policy_v0`, teacher metadata và provenance; chuyển sang LeRobot Dataset để tiến tới fine-tune SmolVLA base + LoRA, bắt đầu với action horizon một bước nếu tương thích. **Không sửa demo đang có, không xóa dữ liệu, không bịa sensor hoặc nhãn lượng nước, không tự train dài/điều khiển van.** Trước hết hãy báo cáo audit, các vấn đề kỹ thuật cần test và kế hoạch M0–M4 với file/lệnh/acceptance criteria. Khi implement, bảo đảm unit tests, group split trước sinh scene, kiểm tra label leakage, reproducibility và phân biệt rõ synthetic labels với agronomic ground truth.
