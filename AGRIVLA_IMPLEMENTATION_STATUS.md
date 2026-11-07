# AgriVLA — trạng thái triển khai

Cập nhật: 2026-11-06. Nhánh: feat/compose-dataset. Code nền lúc bắt đầu: 041ad1e33901d32852538a52fccf1e4ee1e15a8a.

**Bước 0–6, gồm 4.5: PASS_CPU.** Hiện hành wording_v1: 3.600/720/720 records và LeRobot singleton episodes, browser8765 đã đồng bộ. Base/fresh LoRA/processors/tokenizer và CPU forward/inference đã kiểm. Chưa optimizer training/GPU4GB; bước7–9 trên máy NVIDIA mới. V0 giữ nguyên làm parent; xem mục bước6 trước các mục lịch sử bên dưới.

## Tiến độ theo bước

| Bước | Phạm vi | Trạng thái thực |
|---|---|---|
| 0 | Package/CLI, spec, env data, train recipe, preflight, Git allowlist | PASS |
| 1 | Tải/đối chiếu CSV–ZIP, source manifest, quarantine | PASS |
| 2 | Group split và chống leakage | PASS |
| 3 | Policy labeler + templates Việt–Anh | PASS |
| 4 | Compose/QA/preview/JSONL | PASS |
| 4.5 | Trang HTML localhost duyệt toàn bộ dataset và nguồn local | PASS |
| 5 | Cài train dependencies và LeRobot conversion/round-trip CPU | PASS |
| 6 | Model/fresh LoRA/config/inference–evaluator + wording_v1/handoff | PASS_CPU |
| 7 | GPU smoke/backward/save/reload/resume | NOT_STARTED |
| 8 | Launcher/runbook/run dài do người dùng bắt đầu | NOT_STARTED |
| 9 | Validation selection/test/baseline/ablation | NOT_STARTED |

Audit CSV ban đầu được giữ làm lịch sử; bước 1 hiện đã kiểm chứng đầy đủ cả ZIP/ảnh thật. Hai record example vẫn chỉ minh họa cấu trúc, không phải generated training dataset.

## Bước 6 — PASS_CPU; wording_v1 và bàn giao máy GPU (2026-11-06)

Người dùng xác nhận máy hiện tại không có NVIDIA và yêu cầu dừng ở bước 6. Trong bước này người dùng yêu cầu thêm câu văn nói VI/EN, rồi xác nhận “tưới cả 3 vườn” phải là tưới bắt buộc theo mức chỉ định. Vì vậy đã tạo revision riêng **wording_v1**, không ghi đè v0 hoặc sửa code/schema/config đang được lock v0 pin.

### Dataset hiện hành và đồng bộ browser

| Split | Records = singleton episodes | Records cũ giữ target | WATER_ALL mới | Ảnh nguồn độc lập đã dùng | Lượt ảnh pixel-exact |
|---|---:|---:|---:|---:|---:|
| train | 3600 | 3000 | 600 | 1258 | 10800 |
| val | 720 | 600 | 120 | 173 | 2160 |
| test | 720 | 600 | 120 | 254 | 2160 |

5.040 records, 2.160 cases, 1.620 ordered triplets, vẫn chỉ 1.685 ảnh nguồn độc lập. 1.088/402/402 instructions kế thừa đổi sang văn nói, có bản dịch tương ứng. Source triplets/splits/translation pairs/family pools cũ và target sáu intent giữ nguyên; revision metadata truy về parent. WATER_ALL có thêm 6/2/2 families train/unseen-val/unseen-test, mỗi level 1/2/3 chiếm 200/40/40 records ở train/val/test. 104 nguồn annotation=5 vẫn excluded. QA không có source/group intersections; không tạo thêm triplet hoặc nới anti-leakage.

WATER_ALL(level=1/2/3) → [level,level,level] **bất kể wilt**, là synthetic operator override; thiếu mức không tự đoán. “Tưới cả 3 vườn mức 2 nhé” → [2,2,2], cả với wilt [0,0,0]. “Tắt hết.” → STOP_ALL → [0,0,0]. AUTO/WORST/ZONE/CAP/EXCLUDE giữ quy tắc ngưỡng héo 2, WORST vẫn giữ mọi tie. Vườn là tên gọi zone tổng hợp, chưa phải ba ruộng thật đồng thời.

Paths hiện hành:
- JSONL/QA/card/lock: agri-soybean-vla/data/manifests/wording_v1/.
- LeRobot smoke: agri-soybean-vla/data/lerobot/wording_v1/smoke (64 = 32 cặp vi/en, đủ 7 intents và force levels).
- Full: agri-soybean-vla/data/lerobot/wording_v1/full/{train,val,test}.
- Dataset lock v1: 92111dfa8086eef77b90e9ac47003c43f1847ea07a899e1df3c6232d2f977ba3.
- Parent v0 lock c59005a5d4eea36731051f5b82a9341d9be5ae50cca4aafcad1055743ef16563 giữ nguyên; không gộp hai versions vào train.

Browser mặc định và scripts/browse.sh đã chuyển sang wording_v1 tại **http://127.0.0.1:8765/?view=samples&page_size=48**. --replace chỉ thay browser cùng repo/user sau QA. Đã kiểm cả 105 trang/all 5040 IDs/text/target/source khớp JSONL, 840 WATER_ALL, link tải manifests đúng bytes, 9 ảnh gốc đúng SHA, 104 quarantine; Chromium desktop/mobile, filter/tìm “tắt hết”, chi tiết/bản dịch và trang cuối PASS, không lỗi JS/HTTP trong kịch bản. Badge version và hướng dẫn WATER_ALL đã đồng bộ. Reports/screenshots: reports/browser/step6/. Server phải còn chạy; nếu đã dừng do phiên tool kết thúc, chạy lại browse.sh.

### Model, LoRA, processor và CPU checks

LeRobot 7e241bd630a3719a56157a497ce5d08f244784f1, vendor clean; .venv-train tách riêng, .venv data không có Torch/LeRobot. Pip check/imports/preflight PASS; CUDA unavailable là WARN của bước CPU, không chứng nhận 4 GB. Dependencies vẫn dùng lock bước 5.

- Base lerobot/smolvla_base revision **5e8d12a6e2975b0e5e5fce7c8caf47c371d257b6**.
- Backbone/tokenizer HuggingFaceTB/SmolVLM2-500M-Video-Instruct revision **7b375e1b73b11138ff12fe22c8f2822d8fe03467**.
- Cache .cache/models/ và assets_lock.json pin file SHA; không dùng main/latest trong runtime.
- Fresh LoRA r=8/alpha=8, modules_to_save=[]; expert q/v và state/action/time projections theo upstream preset. **371328 / 450417504 parameters trainable = 0,08244%**, chỉ LoRA, VLM/vision frozen. Fresh B matrices zero; không nạp adapter đèn.
- State/action features [3], giữ max_state_dim=max_action_dim=32 để strict-load base weights. Ba RGB, model resize 256, chunk_size=n_action_steps=1, batch1/workers0/compile=false, CUDA AMP. LR3e-4, warmup100, cosine tới3e-5 ở step6000. Đây là candidate, chưa fit-certified.
- **Tokenizer96**: cả 5040 records Unicode round-trip, zero truncation; max vi train/val/test=74/81/76, en=22/22/23. 64 sẽ cắt 483 records tiếng Việt của bản hiện hành. Saved processor được reload và thực thi tokenizer step, exact token sequences cho đủ5040 records/276 prompts unique.
- Train chỉ full/train, dataset.eval_split=0; actual upstream factory giữ đủ3600, eval_dataset=None. Factory images uint8 được đổi float32/255 đúng upstream trước processor. Batch B1 camera[1,1,3,256,256], state/action[1,1,3], action_is_pad[1,1]false. Encoder hook thật cả12 calls nhận[1,3,256,256], state/action pad[1,32]/[1,1,32]. CPU no-grad loss finite (1,6771 và4,5175), inference finite[1,3]. Không backward/optimizer.

**Stats hiện hành bắt buộc**: wording_v1/full/train_stats.json SHA256 **b06849340aa520815152fd63921b3602ab9278f7c3fd242e600c6a33a086c78f**. STATE IDENTITY; ACTION MEAN_STD chỉ từ full train cho inference/val/test. Action mean=[1.0322222222,0.9994444444,0.9777777778], std=[1.1523240071,1.1340682530,1.1282018907]. Canonical nguồn và bản sao pretty-printed được so theo giá trị, đồng thời pin SHA bytes nguồn và bundle lock. Không dùng stats v0/held-out hoặc quantiles aggregate singleton.

Prepared bundle: **agri-soybean-vla/models/prepared/wording_v1/** gồm base mirror/config, saved pre/post processors/stats, tokenizer, schema, provenance, fresh initial_adapter, trainable names/count/dtypes, resolved train config, all-record input_verification và bundle_lock. initial_adapter và checkpoint fixture là **chưa train**, không có optimizer state để resume. Checkpoint hook của launcher thêm initialization provenance/schema/train stats/tokenizer/lock sau upstream save; actual optimizer step/resume state vẫn ở training_state upstream.

### Inference/evaluator, files, lệnh và bằng chứng

Infer reset queue mỗi sample, chỉ allowlist ba cameras/state/task. Inverse normalization rồi reject NaN/Inf/sai shape; giữ continuous/clipped actions và cờ out_of_range; quantize **clip[0,3] rồi floor(x+0.5)**. Không override predictions bằng intent/teacher. infer nhận held-out records hoặc ba ảnh RGB + instruction; câu mẫu thiếu mức bị từ chối, input vượt96tokens bị từ chối thay vì truncation. Evaluator chỉ val/test, metrics theo language/intent/wording, exact-vector/zone accuracy, continuous MAE, false-watering/OFF/STOP và out-of-range; fixture expected độc lập.

Đã fresh-process load checkpoint fixture kèm saved processors/tokenizer và infer2val records; đã infer raw3ảnh + câu Việt. **Ví dụ raw untrained dự đoán quantized [1,2,2] cho lệnh mức2, trong khi target policy là[2,2,2].** Đây là PASS kỹ thuật I/O, không phải model đã học đúng lệnh. Chất lượng/nhận biết héo phải kiểm sau fine-tune/ablation.

Source/config/tests/docs mới hoặc cập nhật: training.py, evaluation.py, handoff.py, revision.py, revision_io.py; CLI/browser/web/browse.sh; configs/training_v0.yaml, revision_v1.yaml, language_wording_v1.yaml, policy_extension_v1.yaml, lerobot_wording_v1.yaml; schemas/scene_v1.schema.json; tests training/revision/handoff/browser; MODEL_TRAINING.md, GPU_HANDOFF.md và các context/plan/prompts/spec/card/readme/conversion/browser docs. Writer/core v0 và demo/vendor giữ nguyên. CLI convert/verify mặc định current config; legacy dùng --config configs/lerobot_v0.yaml. contract mặc định v1, --revision v0 giữ output lịch sử.

Lệnh đã chạy thật, từ repo root:
```bash
agri-soybean-vla/.venv/bin/python -m agri_vla.cli compose-revision
agri-soybean-vla/.venv-train/bin/python -m agri_vla.cli convert --config agri-soybean-vla/configs/lerobot_wording_v1.yaml --profile smoke
agri-soybean-vla/.venv-train/bin/python -m agri_vla.cli convert --config agri-soybean-vla/configs/lerobot_wording_v1.yaml --profile full
agri-soybean-vla/.venv-train/bin/python -m agri_vla.cli prepare-training
agri-soybean-vla/.venv-train/bin/python -m agri_vla.cli check-model-inputs --output agri-soybean-vla/reports/model-inputs-step6.json
agri-soybean-vla/.venv-train/bin/python -m agri_vla.cli infer --split val --limit 2 --checkpoint agri-soybean-vla/outputs/checkpoint-fixture-step6/pretrained_model --output agri-soybean-vla/outputs/infer/step6-checkpoint-fixture
agri-soybean-vla/.venv-train/bin/python -m agri_vla.cli train-upstream --steps 100 --output agri-soybean-vla/outputs/train/step6-dry-run-unused --dry-run
bash agri-soybean-vla/scripts/browse.sh --replace
```

prepare re-runs full independent round-trip before creating bundle. Train wrapper rechecks prepared proof/export file hashes of **train root only**, GPU guard and output-new guard, then calls pinned upstream loop only with explicit --execute. --dry-run did not create training output. Raw infer commands/results are in reports/infer-step6-raw.json; exact install/run/transfer commands in MODEL_TRAINING.md/GPU_HANDOFF.md. Existing outputs are refused; do not rerun create commands over them.

Final tests: **530 passed +1 intentional native-module skip** in .venv; native round-trip fixtures **2 passed** in .venv-train; handoff focused recheck **2 passed** after excluding volatile .log from transfer. Full input round-trip5040records/15120pixels, strict base load/processor/tokenizer/CPU forward/inference, saved checkpoint fixture, trainer dry-run and live browser checks all PASS. Logs from early failed/interrupted attempts are diagnostic history; final reports below are authoritative.

Artifacts: reports/verification-step6.json (master), roundtrip-step6-v1-full.json, model-inputs-step6.json, infer-step6-raw.json, trainer-step6-dry-run.json, preflight-step6-train.json, tests-step6-final-data.xml, tests-step6-native.xml, tests-step6-handoff.xml; reports/browser/step6/browser_report.json/screenshots; prepared bundle reports/config/locks. Transfer target: **outputs/handoff/agri-soybean-step6.tar** +.sha256 +.manifest.json. Source/data/model cache included; env/vendor/prepared host paths/outputs/volatile logs excluded. Rebuild env+bundle on new host per GPU_HANDOFF.md; do not copy venv. No commit/push.

**Đi tiếp:** không có blocker CPU bước6. Máy mới phải NVIDIA/CUDA PASS và làm bước7: actual backward/update, VRAM/ETA, fresh-process save/reload/resume optimizer/scheduler/RNG/step, tiny overfit. Profile chưa được gọi fit4GB. Chưa có smoke-train/resume/run_training CLI hoàn chỉnh; prompt7/runbook ghi rõ phải hiện thực phần đó. Không optimizer training/run dài, không bước7 tự động trên máy hiện tại. Dừng ở bước6.

## Bước 5 — PASS: env ML riêng và LeRobot v3 round-trip CPU (2026-11-06)

Đã đọc AGENTS/context/plan/status, dataset QA report/DATASET_CARD/visual_review, kiểm Git thật trên feat/compose-dataset. Dataset input lock SHA256 khớp c59005a5d4eea36731051f5b82a9341d9be5ae50cca4aafcad1055743ef16563. Chạy lại validator hiện tại và đối chiếu manifest/config/schema/model-contract/implementation/upstream hashes trước export. Không sinh lại scene, chia lại splits/pairs/families, sửa raw/demo/vendor cũ hoặc tải base weights/train/commit/push.

### Môi trường đã xác minh

- .venv vẫn data/browser, không cài ML vào đó; **.venv-train** là env ML mới, Python 3.12.16. setup.sh --mode train hiện chọn env này; data mode không đổi.
- Checkout riêng .vendor/lerobot HEAD đúng **7e241bd630a3719a56157a497ce5d08f244784f1**, worktree sạch. Extras **smolvla,peft,training** đã resolve/cài; không chỉnh source upstream.
- Torch 2.7.1+cu126, torchvision 0.22.1+cu126, TorchCodec 0.5, transformers 5.5.4, PEFT 0.21.2, datasets 4.8.5, pyarrow 26.0.1, numpy 2.2.6, av 15.1.0, accelerate 1.15.0.
- pip check: **No broken requirements found**. Preflight/imports torchvision/datasets/PEFT/LeRobotDataset/SmolVLA modeling PASS. CUDA unavailable = WARN; CPU conversion là phạm vi bước này, chưa xác nhận fit GPU 4 GB.
- requirements-train.constraints.txt đã chuyển từ recipe sang verified core constraints; thêm pin TorchCodec 0.5.0 tương thích Torch 2.7 theo [matrix chính thức](https://github.com/meta-pytorch/torchcodec#compatibility-with-torch-versions). requirements-train.lock.txt lưu full versions, bỏ editable paths; setup cài CUDA wheels rồi lock và editable pinned vendor. Đã kiểm lock --dry-run --no-index khớp installed environment.
- Snapshot/freeze, preflight và full package versions có trong reports/environment; không gọi snapshot editable Git HEAD là bản code triển khai đã commit.

### Counts/path thực

| Root dưới agri-soybean-vla/data/lerobot/v0/ | Nguồn | Episodes | Frames | Ảnh so pixel | Tasks khác nhau | Data Parquet |
|---|---|---:|---:|---:|---:|---:|
| smoke | Train 32 cặp Việt–Anh | 64 | 64 | 192 | 42 | 1 |
| full/train | Train đầy đủ | 3.000 | 3.000 | 9.000 | 132 | 11 |
| full/val | Validation đã khóa | 600 | 600 | 1.800 | 138 | 3 |
| full/test | Test đã khóa | 600 | 600 | 1.800 | 138 | 3 |

Dung lượng quan sát: smoke ~23 MiB, full train ~1,1 GiB, val ~207 MiB, test ~215 MiB. 64 smoke được chọn deterministic seed=42 từ train, đủ vi/en, sáu intents, mọi zone/cap, mọi action level theo từng chiều và panel contrastive; không lấy tùy tiện 64 dòng đầu. Convert smoke, finalize và round-trip PASS; CLI verify riêng PASS rồi mới full. Full tự kiểm lại smoke pins/record selection trước khi ghi.

Một record scene+instruction → một episode một frame; bản dịch/paraphrase là các episode liên quan, không tăng nguồn độc lập. fps=1, timestamp=0 trong mỗi episode chỉ serialization; không lặp frame giả thành chunk 50. Giữ nguyên 3.000/600/600 records/splits.

Ảnh RGB giữ tỉ lệ LANCZOS 256×192, đệm đen 32 pixel trên/dưới →256×256. Writer nhận uint8 HWC, metadata image feature CHW [3,256,256], reader trả float32 CHW [0,1]. PNG lossless **nhúng Parquet**; sau finalize không còn PNG riêng hoặc MP4. Raw bytes không đổi.

Writer chỉ nhận ba image features + state float32[3] zeros + action float32[3] + task instruction Việt/Anh. LeRobot thêm index/frame/episode/timestamp/task_index nội bộ. Teacher/intent/params/case/provenance/generation_metadata không ở features; **sample_metadata.jsonl** giữ episode↔sample/nguồn đầy đủ và ba transformed pixel hashes. Export info/locks pin sidecar/selection/source/config/converter/Parquet/metadata.

### Kiểm chứng mọi record và batch thật

- Verifier tích hợp sau finalize và verifier CLI độc lập đều PASS; full **4.200 records / 12.600 ảnh** so exact pixels với transform từ ảnh gốc, không chỉ so số frame. Đối chiếu instruction UTF-8, sample/source IDs, episode/frame/index/timestamp, camera order, state/action/dtypes, masks, features và hashes.
- Stats mọi feature finite. Action mean/std so trực tiếp toàn bộ target_action nguồn; state std=0 được xử lý bằng IDENTITY.
- Raw loader chỉ action delta [0]: camera [4,3,256,256], state [4,3], action [4,1,3], action_is_pad [4,1] false.
- **Factory train upstream thật** resolve observation delta [0] thêm time axis: camera **[4,1,3,256,256]**, state **[4,1,3]**, action **[4,1,3]**, mọi camera/state/action_is_pad [4,1] false. Task là list 4 strings Việt/Anh. Không suy shape model từ raw loader; cả hai path được lưu report.
- SmolVLAConfig(chunk_size=1,n_action_steps=1) cho action_delta_indices=[0], observation_delta_indices=[0]. EpisodeAwareSampler drop_n_last_frames=0; train/val/test epoch giữ đủ 3.000/600/600 index duy nhất. Smoke giữ 64. Chưa policy forward/backward hoặc tokenizer audit.
- LeRobot mặc định meta/info.splits.train cho toàn container trong cả root val/test; đó không phải source train. Split thật xem **root + export_info.source_split + sidecar**. Không tự chia episodes hoặc hợp nhất roots; bước 6 train chỉ full/train, dataset.eval_split=0.

### Train-only normalization và giới hạn stats đã thấy

full/train_stats.json copy đúng full/train/meta/stats.json về dữ liệu; normalization_contract.json yêu cầu **STATE IDENTITY / ACTION MEAN_STD từ full train**, kể cả inference/val/test. Helper train_statistics từ chối smoke/val/test. Native NormalizerProcessorStep/UnnormalizerProcessorStep đã chạy trên batch cả ba split với cùng train stats; sai số action round-trip **0.0 trong batch đã kiểm**, state giữ nguyên zeros.

Train action mean=[0.8386666667,0.7993333333,0.7733333333], std=[1.1114427061,1.0815419034,1.0692468793]. Val/test có stats riêng theo chuẩn LeRobot cho QA, không dùng để normalize model. SmolVLA default mapping cần override ở bước 6; bước 5 không thay processor/model ngầm.

Phát hiện thực: quantiles upstream aggregate theo episode, nên q01/q10/q50/q90/q99 action ở singleton exports là episode-aggregated values, không phải true global quantiles. Mean/std/min/max đúng và contract dùng MEAN_STD. Không sửa vendor/meta stats; action-stats-step5-reference.json ghi true quantiles và đối chiếu; **không dùng QUANTILES normalization với các giá trị aggregate đó**.

~~~text
Input dataset lock SHA256 (không đổi):
c59005a5d4eea36731051f5b82a9341d9be5ae50cca4aafcad1055743ef16563
Conversion config SHA256:
8407de4b5dba29552d6fa6be4a43293f3b0a0945b59c1363c244c115bcf214c5
Full dependency lock SHA256:
27c7fd34c3a21e7730d313dc16779d6d8df990341e7b292c76b0f1715ee49c36
Export lock SHA256:
smoke d1bedb586256e2b491384092b380917bf2578f6b8a175afb4974aea02efe978a
train 92de2fc3c803fa45c3f2c98f1a8a8860c7341a97da079dc95cb123b76ac187c0
val   89b50e58c198ad26a699caeff70d2cc3135495e3ef9b017ea49d9a6254703263
test  9734f54bf8d80573b6c946a0dbfd7b1d964259f3acb451a3580e5fc67bfbabad
Full train meta/stats.json SHA256:
ca4a811731bd125ebfccc070e3a55e526e8e77c4e6ad1975f4012f2f0bab15f0
Full train_stats.json SHA256 (canonical JSON copy):
8cb4f96fb861d6d48f911d2d6956e5a0e632406d3b676025ed2cda97d94756e7
Normalization contract SHA256:
996fe1ea81357d1125c60104db046fcca3c0d6472f3305be15c0a03a2065cbdc
~~~

### Files/lệnh/test/artifacts

Tạo configs/lerobot_v0.yaml, lerobot_io.py, lerobot_checks.py, requirements-train.lock.txt, tests/test_conversion.py, tests/test_lerobot_roundtrip.py và CONVERSION.md. Sửa setup.sh, preflight.py, CLI convert/verify-lerobot, constraints/.gitignore và README/spec/card/context/plan/prompts/status. **Không sửa configs/dataset_v0.yaml, schema/model_contract JSON hoặc implementation đã pin trong dataset lock**; flags runtime cũ trong contract là lịch sử bước 0, report bước 5 là bằng chứng mới.

Lệnh thật từ repo root:

~~~bash
AGRI_DATA_PY=agri-soybean-vla/.venv/bin/python
AGRI_TRAIN_PY=agri-soybean-vla/.venv-train/bin/python
$AGRI_DATA_PY -m agri_vla.cli validate --manifests agri-soybean-vla/data/manifests/v0 --output agri-soybean-vla/reports/validate-step5-input.json
bash agri-soybean-vla/scripts/setup.sh --mode train
$AGRI_TRAIN_PY -m pip check
$AGRI_TRAIN_PY -m agri_vla.cli preflight --mode train --json --output agri-soybean-vla/reports/preflight-step5-train.json
$AGRI_DATA_PY -m agri_vla.cli convert --profile smoke --dry-run
$AGRI_TRAIN_PY -m agri_vla.cli convert --profile smoke
$AGRI_TRAIN_PY -m agri_vla.cli verify-lerobot --root agri-soybean-vla/data/lerobot/v0/smoke --output agri-soybean-vla/reports/roundtrip-step5-smoke-factory.json
$AGRI_TRAIN_PY -m agri_vla.cli convert --profile full --dry-run
$AGRI_TRAIN_PY -m agri_vla.cli convert --profile full
$AGRI_TRAIN_PY -m agri_vla.cli verify-lerobot --root agri-soybean-vla/data/lerobot/v0/full --output agri-soybean-vla/reports/roundtrip-step5-full.json
$AGRI_DATA_PY -m pytest agri-soybean-vla/tests -q --junitxml=agri-soybean-vla/reports/tests-step5-data.xml
$AGRI_TRAIN_PY -m pytest agri-soybean-vla/tests/test_lerobot_roundtrip.py -q --junitxml=agri-soybean-vla/reports/tests-step5-native.xml
~~~

Default roots/reports trên đã tồn tại. Muốn convert lại phải --output-dir mới, full chọn --smoke-root đã verify; không xóa/ghi đè roots. Verify mặc định read-only/in report, --output yêu cầu file mới; output vào raw bị chặn. Converter --dry-run data-only không import Torch/LeRobot/NumPy; conversion/verify thật bắt buộc .venv-train và dùng CPU/offline Hub.

**Tests thật:** data regression **363 PASS, 1 module native skipped có chủ đích**, native env **2 PASS**. 20 tests mới data kiểm smoke coverage/pairing/reproducibility, invalid config/inputs, train-only stats guard, feature allowlist và no-ML/no-write dry-run. Native tests kiểm RGB/letterbox/pixels/camera order/text/action singleton, raw/factory batch/sampler và normalizer/inverse zero-state. Native module đã chạy riêng, không bỏ qua chức năng ML. Input validator/independent full export verifier đều PASS.

- [Hướng dẫn/shapes](agri-soybean-vla/CONVERSION.md), [train dependency lock](agri-soybean-vla/requirements-train.lock.txt).
- [Env versions](agri-soybean-vla/reports/environment/train-step5-verified.json), [train preflight](agri-soybean-vla/reports/preflight-step5-train.json), [setup log](agri-soybean-vla/reports/setup-step5-train.log).
- [Input QA lại](agri-soybean-vla/reports/validate-step5-input.json), [smoke verifier + factory](agri-soybean-vla/reports/roundtrip-step5-smoke-factory.json), [full verifier + factory](agri-soybean-vla/reports/roundtrip-step5-full.json).
- [Full summary](agri-soybean-vla/data/lerobot/v0/full/roundtrip_report.json), [train stats](agri-soybean-vla/data/lerobot/v0/full/train_stats.json), [normalization contract](agri-soybean-vla/data/lerobot/v0/full/normalization_contract.json), [quantile diagnostic](agri-soybean-vla/reports/action-stats-step5-reference.json).
- [Smoke mapping](agri-soybean-vla/data/lerobot/v0/smoke/sample_metadata.jsonl), [train mapping](agri-soybean-vla/data/lerobot/v0/full/train/sample_metadata.jsonl); val/test có cùng filenames dưới roots riêng.
- [Data JUnit](agri-soybean-vla/reports/tests-step5-data.xml), [native JUnit](agri-soybean-vla/reports/tests-step5-native.xml), [verification/hash bàn giao](agri-soybean-vla/reports/verification-step5.json).

**Blocker/điều kiện sang bước 6:** không có blocker CPU dataset/conversion. Dùng đúng .venv-train/commit/exports và train stats; giữ observation history axis, sampler drop=0 và split roots riêng. Bước 6 mới tải model/tokenizer/fresh LoRA, audit vi/en truncation/processor/resize; bước 7 mới GPU forward/backward/save/reload/resume. Chưa chứng nhận fit 4 GB/model hiểu Vision/Language hoặc tưới thật. Dừng đúng bước 5.

## Bước 4.5 — PASS: trang HTML localhost để tự browse dataset

Yêu cầu bổ sung: trang đơn giản, đọc toàn bộ samples, dễ tìm/lọc/xem ảnh và mở đúng thư mục gốc local. Đã implement, chạy server thật và kiểm bằng HTTP + Chromium; không thay pipeline/dataset đã khóa ở bước 4.

**Mở trang:** http://127.0.0.1:8765/. Đã chạy trong phiên này bằng scripts/browse.sh. Link chỉ hoạt động trong lúc process server đang chạy; phiên sau cần chạy lại nếu process đã dừng.

~~~bash
cd /home/hanhpham-duc/agriVLA
bash agri-soybean-vla/scripts/browse.sh
# Đợi terminal in Dataset QA PASS và URL, giữ terminal mở.
# Nếu port bận: bash agri-soybean-vla/scripts/browse.sh --port 8766
# Chỉ kiểm tra, không mở port/ghi file:
bash agri-soybean-vla/scripts/browse.sh --dry-run
~~~

**Các phần trên trang:**

- Mẫu tổng hợp: đủ 4.200 record; page size 12/24/48, trang đầu/cuối/nhập trang/đi thẳng mẫu #1–4200. Ba ảnh riêng, instruction, target mỗi zone, split/language/wording và manifest line.
- Tìm kiếm bỏ dấu Việt + lọc kết hợp split, vi/en, seen/unseen, intent, nhãn héo, group, action, loại case; reset và empty state rõ ràng.
- Chi tiết: bảng bản dịch/quartet, cùng ảnh/lệnh khác, tải đúng record JSON, copy link mẫu, provenance/teacher được thu gọn. Mỗi ảnh có link original bytes, thư mục gốc, absolute path/checksum.
- Ảnh nguồn: đủ 1.892 ảnh, 1.788 eligible, 104 quarantine nhãn 5, 103 eligible chưa chọn; số lần dùng theo record và liên kết tới các mẫu chứa ảnh.
- Thư mục & tệp gốc: năm nhóm manifests/images/raw/source-splits/source-report. Liệt kê file thật bằng HTTP local, hiện/copy path; không cho duyệt toàn repo hoặc chỉnh file. Hướng dẫn mở OS Files bằng xdg-open trong BROWSE_DATASET.md.
- Cách đọc mẫu: scene/case/record, bốn mức action, state giả định và phân biệt metadata/model input.

**Files tạo/sửa:** browser.py; web/browser.html, browser.css, browser.js; scripts/browse.sh; tests/test_browser.py; CLI browse; pyproject package-data; .gitignore chỉ thêm allowlist assets source; BROWSE_DATASET.md, README, context/plan/status. Không thêm runtime dependency, không sửa file implementation/schema/config đã pin trong dataset lock. Tool test Playwright/Chromium và các .deb thư viện bổ sung chỉ ở /tmp/agrivla-ui-check, không cài vào hệ thống hoặc env data/train.

Server bind 127.0.0.1, chạy full validator trước khi mở port. Chỉ phục vụ assets và những file dataset cụ thể trong allowlist, không mở .git/.venv. Ảnh gốc kiểm encoded SHA256 khi đọc; thumbnails resize trong RAM/cache riêng, không sửa raw. Manifest/source/config/policy/split lock vẫn giữ nguyên.

**Kiểm chứng thật:**

- Full pytest **343 PASS**, 0 failures/errors/skips: 326 cũ + 17 browser tests; JUnit tests-step45.xml. Có kiểm pagination toàn bộ fixture, filters/negative queries, originals, download JSON, pair/quartet, local folder allowlist, read-only methods, dry-run không socket/ghi file và không import ML.
- HTTP trên server dataset thật: 88 trang × tối đa 48 record, đủ 4.200 IDs đúng thứ tự manifest, không thiếu/trùng; 40 trang đủ 1.892 nguồn/104 quarantine. Đối chiếu SHA256 9 ảnh original across train/val/test.
- Chromium/Playwright: trang đầu và trang cuối 350, mẫu #4200/quartet/full JSON download/copy path/original tab; kết hợp val+en+unseen+STOP trả đúng 20; tìm bỏ dấu; empty/reset; translation pair; cùng scene trả 22 record; quarantine 104; năm folder cards và folder ảnh có 1.892 file.
- Desktop 1440px và mobile 390px, modal chi tiết hoạt động, mobile không tràn ngang; không có JS page errors hoặc HTTP errors trong kịch bản. Đã mở xem screenshots desktop/mobile/detail. Không chỉ kiểm HTML bằng text.
- Dataset lock vẫn c59005a5d4eea36731051f5b82a9341d9be5ae50cca4aafcad1055743ef16563; không train/convert/commit/push, demo/vendor giữ nguyên.

**Artifacts và hướng dẫn:**

- [Hướng dẫn từng bước/thao tác/kỳ vọng/troubleshooting](agri-soybean-vla/BROWSE_DATASET.md).
- [Desktop](agri-soybean-vla/reports/browser/step45/desktop.png), [lọc dữ liệu](agri-soybean-vla/reports/browser/step45/filtered.png), [chi tiết](agri-soybean-vla/reports/browser/step45/detail.png), [mobile](agri-soybean-vla/reports/browser/step45/mobile.png), [thư mục gốc](agri-soybean-vla/reports/browser/step45/folders.png), [quarantine](agri-soybean-vla/reports/browser/step45/quarantine.png).
- [UI test report](agri-soybean-vla/reports/browser/step45/ui-test-report.json), [HTTP toàn dataset](agri-soybean-vla/reports/browser/step45/http-test-report.json), [JUnit](agri-soybean-vla/reports/tests-step45.xml), [verification bước 4.5](agri-soybean-vla/reports/verification-step45.json).
- Code/UI/docs được allowlist Git; data/reports/screenshots/cache vẫn ignored. Preview tĩnh 72 record cũ không bị ghi đè.

**Giới hạn:** URL localhost và absolute paths trỏ tới máy chạy server; máy khác qua SSH/VS Code phải forward port. Browser HTTP mở listing local, không tự gọi file manager. Server đọc snapshot đã QA lúc khởi động; khi dataset đổi phải khởi động lại. Đây là công cụ xem dữ liệu, chưa có LeRobot round-trip/model/GPU 4GB validation. Không có blocker để dùng browser hoặc tiếp tục bước 5 khi người dùng giao.

## Bước 4 — PASS: scene JSONL, QA và preview thật

Đã đọc AGENTS/context/plan/status/configs, kiểm Git thật trên feat/compose-dataset, data preflight PASS và chạy lại 282 tests bước 0–3 trước sửa. Không sửa source split, policy/templates đã khóa, raw, demo hoặc vendor; không train/commit/push. Các mục bước 0–3 bên dưới là lịch sử tại thời điểm bàn giao đó.

### Số liệu thật

| Split | Ảnh có sẵn / đã dùng | Ordered scenes | Cases | Cặp dịch | Records | vi / en | Toàn OFF |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 1.258 / 1.258 | 1.440 | 1.500 | 1.500 | 3.000 | 1.500 / 1.500 | 684 (22,8%) |
| val | 189 / 173 | 90 | 150 | 300 | 600 | 300 / 300 | 212 (35,33%) |
| test | 341 / 254 | 90 | 150 | 300 | 600 | 300 / 300 | 208 (34,67%) |

Tổng 4.200 record từ 1.685 ảnh nguồn khác nhau, không phải 4.200 ảnh/quyết định độc lập. 103 ảnh eligible chưa được chọn vào panel (16 val, 87 test); không bị loại khỏi source splits và không phải ảnh hỏng. 104 annotation=5 vẫn ngoài v0, không tự relabel.

Ordered scene là bộ ba ảnh theo thứ tự zone, case là scene + intent/params; cùng scene có thể có nhiều lệnh đối chiếu. Train mỗi case hai bản dịch, val/test mỗi case bốn record vi/en × seen/unseen. Paraphrase/tổ hợp không tăng số nguồn độc lập; ảnh cùng plot/thời gian còn tương quan. Val vẫn chỉ bốn ảnh gốc wilt 4 và test chỉ một plot.

| Quota cases | AUTO | WORST_ONLY | ZONE_ONLY | CAP_LEVEL | EXCLUDE_ZONE | STOP_ALL |
|---|---:|---:|---:|---:|---:|---:|
| train | 450 | 300 | 225 | 225 | 150 | 150 |
| mỗi val/test | 30 | 30 | 30 | 20 | 20 | 20 |

Mỗi zone/cap trong một intent chênh không quá một case; mỗi case có đủ vi/en. Train chứa đủ 66 tổ hợp family–params của registry ở từng ngôn ngữ, tránh một family chỉ đi với một zone/cap. Cả 36 train families thực sự được dùng; unseen-val và unseen-test đủ 12 family mỗi pool, không giao train hoặc nhau. Seen val/test kiểm theo family **và params** từng xuất hiện trong train.

Reuse theo cases (đếm một lần mỗi source/case, không nhân ngôn ngữ/wording): train min/median/max=1/3/22, val=0/1/32, test=0/1/22. Tổng source placements cases=4.500/450/450. Max theo records=44/128/88 do train nhân 2, eval nhân 4. Báo cả reuse theo ordered scene, case, record và từng zone/từng source trong qa_report.json; các số 0 ở val/test là nguồn chưa chọn. Nguồn wilt 4 của val ít nên tái sử dụng cao, không được diễn giải thành nhiều quan sát mới.

### Implementation và QA

- dataset.py: config strict/seed, safe paths/output, upstream lock/checksum/source accounting, ảnh thật và within-scene duplicate checks.
- scenes.py: RNG cục bộ seed=42, sampler ưu tiên ảnh ít dùng và vị trí zone, quotas, cặp dịch/quartets, IDs từ nội dung, provenance và controlled panels. Không LLM gán target.
- validate.py: schema + source membership/paths/order/RGB/dimensions/encoded/pixel/dHash, teacher và policy tái tính, cặp dịch/seen-unseen, actual family/params coverage, quotas, leakage, permutation, byte/semantic hashes và dataset card.
- preview.py: HTML local có full JSON, ảnh gốc có link, thumbnails/contact sheets tách riêng khỏi model images; chọn mẫu phân tầng, mặc định ≥60, --count có thể xem toàn bộ dataset.
- cli.py: compose/validate/preview với --help, --config, --dry-run, output mới; từ chối path đã tồn tại và output trong raw. Không --overwrite/rmtree.
- configs/dataset_v0.yaml: thêm generation config/locks/quota/output/within-scene policy. Không sửa configs/split_v0.yaml, policy_v0.yaml hoặc language_v0.yaml.
- schemas/scene_v0.schema.json và model_contract.json: case/provenance/generation metadata có schema và là metadata_only. Field mới optional trong structural schema để giữ placeholder ví dụ bước 0; full QA scene thật bắt buộc có đủ.
- tests/test_generation.py: fixture ảnh tự tạo, không cần download; kiểm quotas/pairing/coverage, contrastive examples/permutation expected độc lập, tamper/negative cases, duplicate handling, impossible sources, upstream lock/image changes, dry-run/no-overwrite/reproducibility, preview/HTML và no-ML-import.
- DATASET_CARD.md, README/spec/context/plan/prompts/status: thông tin dataset thật và bàn giao bước 5.

Mỗi scene có ba source IDs, encoded hashes, pixel hashes và duplicate components khác nhau. Trong bán kính dHash của bước 2, cặp near hoặc uncertain bị từ chối trong cùng scene; dHash bằng nhau tự nó chưa bị coi là trùng. Giữa splits không giao source ID/hash/pixel/group/camera/plot/component; ảnh/source hashes được kiểm lại từ bytes hiện tại, không chạy lại/nới chính sách chia nhóm. Heuristic chưa bao quát mọi crop/rotation.

Có 11 permutation cases mỗi split (33 tổng), remap zone đúng cả params/câu/target. Panel gồm [0,2,4], [0,0,0], [4,4,4], [4,4,1], [1,1,0]; bao phủ cùng ảnh/khác lệnh, cùng câu/khác ảnh, tie/all-OFF/all-high/STOP/exclusion. Mọi zone có đủ action 0/1/2/3 trong dataset thật. Policy vẫn là proxy ordinal, không hiệu chuẩn lượng nước và không đo đất khô nhất.

### Reproducibility và artifacts

~~~text
Seed: 42
Compose config SHA256: 89b6339b9da5e5fa976574ea6246a317d2f38cf1748f1f0a0d4f73ba3fb0525b
Source split lock (không đổi): 3be04c67ed375c76e4a97d425416e3a890bd1c060ce1167b9bbd5f16d6fcaa8d
Policy lock (không đổi): c6c0d5d2be39fb61581a10aa39757e03b840518ddf6c210a372c5fb2853b4d11
Dataset lock SHA256: c59005a5d4eea36731051f5b82a9341d9be5ae50cca4aafcad1055743ef16563

Manifest byte SHA256:
train 0d130b0d81ab61418fa2f5abc5cc27b399d96a33d0103fd306bdb783ebc38d9f
val   99dd184ef07b9bd8bb440b535a6be6cdd9d2d4ba38456d1c2f89f90b451986f9
test  1f108dc5d141c8e87ce94bd24e83110448b95e57922486638814d1c4fb38a82b

Manifest semantic SHA256 (sort theo sample_id, canonical JSON):
train 1e4c79745f955634d16353bdd61100d4ab174c4904fca2c5bc79e187593b1fa0
val   4d61a3d6599d8017c94cb47feb862ca6f13c317f6a0f3b7d9fa53a8d7b3c182d
test  0775f6c6b3f35a1d362432583a92ee29a8196df8e85a02934ef69b83bbfbbfd9
~~~

Dry-run/run trên nguồn thật có cùng lock, byte hashes, semantic hashes và thống kê. Unit test chạy hai output directories, so byte-for-byte toàn bộ outputs. Validator CLI độc lập trả report giống hệt QA trong compose. Dataset lock pin config snapshot/file hash, generator implementation, schema, model contract, upstream locks và manifests; không phụ thuộc output override. Không sửa code đã pin rồi mặc định nhận dataset cũ mà không audit/version lại.

Artifacts local (data/reports vẫn Git ignored):

- [Train JSONL](agri-soybean-vla/data/manifests/v0/train.jsonl), [val JSONL](agri-soybean-vla/data/manifests/v0/val.jsonl), [test JSONL](agri-soybean-vla/data/manifests/v0/test.jsonl).
- [QA report](agri-soybean-vla/data/manifests/v0/qa_report.json), [dataset lock](agri-soybean-vla/data/manifests/v0/dataset_lock.json).
- [DATASET_CARD](agri-soybean-vla/DATASET_CARD.md), bản sinh cùng dataset ở data/manifests/v0/DATASET_CARD.md.
- [Preview HTML cuối](agri-soybean-vla/data/manifests/v0/review/index.html), [72 record đã xem](agri-soybean-vla/data/manifests/v0/review/preview.samples.jsonl), [visual review receipt](agri-soybean-vla/data/manifests/v0/review/visual_review.json).
- [Panel 1](agri-soybean-vla/data/manifests/v0/review/review_01.jpg) đến review_12.jpg; [AUTO high](agri-soybean-vla/data/manifests/v0/review/review_07.jpg) và [STOP cùng ảnh high](agri-soybean-vla/data/manifests/v0/review/review_10.jpg).
- [Mẫu đầy đủ JSON thật](agri-soybean-vla/reports/example-step4-record.json); README/spec giải thích mọi field. Wilt [0,2,4], ZONE_ONLY(2), instruction Việt → [0,1,0]; ba ảnh gốc từ 5-180/10-87/10-87, không phải chụp đồng thời.
- [Standalone validation](agri-soybean-vla/reports/validate-step4.json), [verification](agri-soybean-vla/reports/verification-step4.json), [JUnit](agri-soybean-vla/reports/tests-step4.xml).
- reports/compose-step4-dry-run.json, compose-step4-run.json, preview-step4-review-dry-run.json và preview-step4-review-run.json: receipts lệnh thật.

Đã mở ảnh contact sheets bằng view_image: **72 record / 24 cases / 22 ordered scenes / 55 source images**, phủ cả split, hai ngôn ngữ, seen/unseen, sáu intents và bảy loại generation case. Mười panel cuối trùng byte với panel đã mở ban đầu; hai panel thay đổi được mở lại để bổ sung cặp AUTO [3,3,3] và STOP [0,0,0] trên cùng ảnh [4,4,4]. Teacher/targets nằm ngoài ảnh trên review sheet, không chèn vào training pixels. Đã kiểm HTML có 72 cards/216 ảnh và mọi link local resolve; không dùng browser renderer. Đây là review thumbnail/câu/nhãn/provenance, không chấm lại nhãn chuyên gia hoặc xác nhận hiệu quả tưới.

Lệnh đã chạy từ repo root:

~~~bash
AGRI_PY=agri-soybean-vla/.venv/bin/python
$AGRI_PY -m agri_vla.cli preflight --mode data
$AGRI_PY -m pytest agri-soybean-vla/tests -q
$AGRI_PY -m agri_vla.cli compose --config agri-soybean-vla/configs/dataset_v0.yaml --dry-run > agri-soybean-vla/reports/compose-step4-dry-run.json
$AGRI_PY -m agri_vla.cli compose --config agri-soybean-vla/configs/dataset_v0.yaml > agri-soybean-vla/reports/compose-step4-run.json
$AGRI_PY -m agri_vla.cli validate --manifests agri-soybean-vla/data/manifests/v0 --output agri-soybean-vla/reports/validate-step4.json
$AGRI_PY -m agri_vla.cli preview --manifests agri-soybean-vla/data/manifests/v0 --output-dir agri-soybean-vla/data/manifests/v0/review --count 72 --dry-run > agri-soybean-vla/reports/preview-step4-review-dry-run.json
$AGRI_PY -m agri_vla.cli preview --manifests agri-soybean-vla/data/manifests/v0 --output-dir agri-soybean-vla/data/manifests/v0/review --count 72 > agri-soybean-vla/reports/preview-step4-review-run.json
$AGRI_PY -m pytest agri-soybean-vla/tests -q --junitxml=agri-soybean-vla/reports/tests-step4.xml
~~~

Compose/preview paths trên hiện đã tồn tại; chạy lại phải chọn --output-dir mới (kể cả dry-run). Validator mặc định chỉ in report; --output yêu cầu file mới và --dry-run không ghi. Preview ban đầu được giữ ở preview/, final review tại review/, không xóa/ghi đè artifacts.

**Test results: 326 PASS**, 0 failures/errors/skips (282 test cũ + 44 test bước 4), full suite cuối 28,11 giây. Xem JUnit và verification-step4.json. Mọi lệnh data mới có import-poison test chặn Torch/LeRobot/Transformers/PEFT. Không tải/cài train dependencies hoặc weights.

**Blocker/điều kiện sang bước 5:** không có blocker dataset cho conversion. Chạy validate hiện tại và kiểm dataset lock/manifest hashes trước convert; dùng instruction + ba ảnh nguyên gốc + state zeros, chỉ target_action làm loss target. Metadata case/provenance/teacher/intent không thành model features. Bước 5 mới cài train dependencies/LeRobot đã pin, convert CPU và round-trip. Tokenizer/truncation, GPU 4 GB, model hiểu ảnh/tiếng Việt và hiệu quả tưới vật lý chưa chứng nhận. Dừng đúng bước 4.

## Bước 3 — PASS: deterministic policy và registry Việt–Anh

Đã đọc AGENTS/context/plan/status/configs, xác minh nhánh thật feat/compose-dataset và data preflight PASS. Mọi lệnh Python dùng agri-soybean-vla/.venv. Không cài/tải model hoặc sửa demo/vendor. Các phần bước 0–2 bên dưới là lịch sử bàn giao tại thời điểm đó.

**Policy:** wilt_proxy_policy_v0, label_type=synthetic_policy_action; mapping {0:0,1:0,2:1,3:2,4:3}. Bốn mức action là OFF/LOW/MEDIUM/HIGH, chưa phải lượng nước hiệu chuẩn. Ngưỡng héo 2 mới có action; wilt 0/1 đều OFF. Chỉ quan sát héo lá, không có cảm biến chứng minh vùng nào có đất khô nhất.

| Intent | Tham số hợp lệ | Quy tắc | Target với wilt [0,2,4] |
|---|---|---|---|
| AUTO | Không | Mapping từng zone | [0,1,3] |
| WORST_ONLY | Không | Max ≥2: giữ tất cả zone hòa max; còn lại OFF | [0,0,3] |
| ZONE_ONLY | zone=1/2/3 | Chỉ giữ zone được chỉ định | zone=2: [0,1,0] |
| CAP_LEVEL | cap=1/2 | Chặn từng mức tại cap | cap=1: [0,1,1] |
| EXCLUDE_ZONE | zone=1/2/3 | Tắt zone chỉ định, giữ mapping các zone khác | zone=3: [0,1,0] |
| STOP_ALL | Không | Luôn [0,0,0] với input hợp lệ | [0,0,0] |

WORST_ONLY([4,4,1])=[3,3,0], WORST_ONLY([1,1,0])=[0,0,0]. STOP vẫn kiểm tra schema/domain của input trước khi trả zeros. Từ chối wilt ngoài 0–4, boolean/float, zone/cap thiếu/thừa/ngoài miền, intent không hỗ trợ hoặc ghép nhiều intent, YAML trùng key và config trái semantics v0. Không có parser tự suy đoán intent từ câu tự do hoặc LLM gán target; generator nhận intent/params đã validate rồi policy tính nhãn.

**Ngôn ngữ:** 60 family song ngữ = 120 template gốc; mỗi intent có 6 train + 2 unseen-val + 2 unseen-test. Sau điền zone/cap có 220 câu: train 66 vi + 66 en, unseen-val 22 vi + 22 en, unseen-test 22 vi + 22 en. Family và câu chuẩn hóa không giao giữa ba pool. Cả hai bản dịch được giữ cùng family/partition. Đã đọc toàn bộ template để kiểm tra threshold, tie, cap, exclusion và STOP; lexical guards/test không chứng minh ngữ nghĩa mọi câu tự do hoặc khả năng hiểu tiếng Việt của model.

Renderer chỉ điền zone/cap; không nhận wilt, không chèn teacher label/action/intent enum vào instruction. Wilt và structured intent/params nằm ở teacher/provenance metadata vì chứa thông tin dùng tính đáp án; model text chỉ là instruction, kết hợp ba ảnh và state [0,0,0] (giả định mọi van ban đầu đóng). Model contract vẫn giữ nguyên. Translation_pair_id gắn với scene sẽ tạo ở bước 4.

Seed mặc định 42 dùng RNG cục bộ. CLI chọn một family để giải thích có tính tái lập; bước 4 phải phân phối family cân bằng hoặc dùng seed dẫn xuất theo sample, không gọi seed cố định cho mọi sample rồi bỏ sót các family. Val/test seen chỉ được dùng family **thực sự xuất hiện trong train**; coverage của dataset chưa thể kiểm tra khi chưa sinh scene. Chưa đo tokenizer/truncation hoặc chạy model.

**Files tạo/sửa:** configs/policy_v0.yaml, configs/language_v0.yaml; src/agri_vla/policy.py, language.py, policy_audit.py và cli.py; tests/test_policy.py, test_language.py, test_policy_audit.py; README/spec/context/plan/prompts/status. Không sửa source manifests/config split.

~~~text
Policy config SHA256: 0a65a310ded77c4021cdb7c8501beca076bfc6f2e52b7a981c7f7f76bbb22495
Language config SHA256: ec27e319033fad420ffdc7810db930c130e7163c2397dc456fd18993a7b22a15
Policy lock SHA256: c6c0d5d2be39fb61581a10aa39757e03b840518ddf6c210a372c5fb2853b4d11
Source split lock SHA256 (không đổi): 3be04c67ed375c76e4a97d425416e3a890bd1c060ce1167b9bbd5f16d6fcaa8d
~~~

Lệnh đã chạy từ repo root:

~~~bash
AGRI_PY=agri-soybean-vla/.venv/bin/python
$AGRI_PY -m agri_vla.cli preflight --mode data
$AGRI_PY -m agri_vla.cli explain-policy --wilt 0 2 4
$AGRI_PY -m agri_vla.cli explain-policy --wilt 4 4 1 --intent WORST_ONLY --split test --wording unseen
$AGRI_PY -m agri_vla.cli audit-policy --output-dir agri-soybean-vla/reports/policy/v0 --dry-run
$AGRI_PY -m agri_vla.cli audit-policy --output-dir agri-soybean-vla/reports/policy/v0
$AGRI_PY -m pytest agri-soybean-vla/tests -q --junitxml=agri-soybean-vla/reports/tests-step3.xml
~~~

**Kết quả thật: 282 PASS** (69 test cũ + 213 test mới), 0 failure/error/skip. Bao phủ 125 tổ hợp wilt × 11 intent/params = 1.375 vector so với test specification độc lập, các ví dụ expected viết sẵn, 8.250 phép kiểm hoán vị với zone remap, AUTO monotonicity, STOP/tie/cap/exclusion/invalid config và CLI, registry isolation, no-overwrite, dry-run và byte-for-byte reproducibility. Import-poison test xác nhận các lệnh data mới không import Torch/LeRobot. Audit xuất bảng kiểm chứng nhãn, không phải scene dataset.

**Artifact đã tạo và kiểm hash:**

- [Policy report](agri-soybean-vla/reports/policy/v0/policy_report.json), [lock](agri-soybean-vla/reports/policy/v0/policy_lock.json).
- [Truth table 1.375 dòng](agri-soybean-vla/reports/policy/v0/policy_truth_table.csv).
- [Catalog 220 câu Việt–Anh](agri-soybean-vla/reports/policy/v0/template_catalog.csv).
- [Sáu ví dụ song ngữ [0,2,4] → target](agri-soybean-vla/reports/policy/v0/policy_examples.md).
- [Verification hash/dry-run/source unchanged](agri-soybean-vla/reports/verification-step3.json).
- [Tests JUnit](agri-soybean-vla/reports/tests-step3.xml).

Dry-run và run thật cùng policy lock hash; cả ba artifact hashes trong lock đã đối chiếu độc lập. Source split config/lock và hashes cả bốn manifests bước 2 không đổi; source manifests vẫn không có instruction/target_action. Reports/data/raw/env vẫn Git ignored; source/config/tests/docs nằm trong allowlist. Không commit/push/train.

**Blocker/điều kiện sang bước 4:** không có blocker cho compose. Bước 4 phải xác minh source split lock/manifests và policy/language config hashes trong policy lock, dùng đúng ba eligible source splits, giữ family isolation, kiểm coverage seen thực tế và tạo translation_pair_id theo scene. Tokenizer/model/GPU 4 GB và hiệu quả tưới vật lý vẫn chưa được chứng nhận. Dừng ở bước 3, chưa implement compose/validate/preview.

## Bước 2 — PASS: source group split đã khóa

Implement split theo cấu hình tường minh, không sinh scene. Chỉ dùng eligible_v0=true: 1.788 ảnh; 104 record annotation=5/wilt_score=null giữ ở excluded.sources.jsonl, không gán vào train/val/test.

| Split | Groups | Ảnh | Wilt 0 | Wilt 1 | Wilt 2 | Wilt 3 | Wilt 4 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Train | 8 | 1.258 | 466 | 343 | 248 | 65 | 136 |
| Val: 4-228 | 1 | 189 | 21 | 48 | 49 | 67 | 4 |
| Test: 11-46 | 1 | 341 | 37 | 82 | 72 | 84 | 66 |

Camera 5 có hai plot, cả hai ở train. Toàn bộ nhóm/chuỗi thời gian cùng plot được giữ nguyên trong một split. Không phải đổi assignment dự kiến vì audit không phát hiện xung đột qua split.

**Hash khóa và provenance:**

~~~text
Seed: 42
Config SHA256:
ee69efaae62429f63ffb82dca87ef062e15abe875c365b55d6a0ced6579eb353

Source manifest SHA256:
421596b6e773ccd3c5d191c7a291521a88158cf3679458f63ee2f7ecdc65f110

Source report SHA256:
8b660622a92b9d5c9c04f9df27e04812831e8e39a45db6c027299928404f8d98

Split lock SHA256:
3be04c67ed375c76e4a97d425416e3a890bd1c060ce1167b9bbd5f16d6fcaa8d
~~~

Config pin cả source manifest và source report. Mỗi ảnh được đối chiếu lại encoded SHA256, pixel SHA256, dHash, mode/dimensions từ file hiện tại trước khi chia. Dry-run và run thật cho cùng lock hash. Unit test tái lập so byte-for-byte toàn bộ output ở hai directory; audit độc lập đã kiểm tra cả bốn manifest hashes trong lock và metadata nguồn được giữ nguyên.

**Duplicate audit:**

- Mọi cặp source eligible được xét khoảng cách dHash64; 3.645 cặp có Hamming ≤8 đi tiếp vào pixel comparison.
- Thumbnail RGB 64×48: near khi MAE/255 ≤0.035, gray Pearson ≥0.985, gray std/255 ≥0.02. Vùng review và giới hạn đầy đủ nằm trong config/README; đây là heuristic, không phải khẳng định phát hiện mọi biến thể crop/rotation.
- 3 cặp near_duplicate theo policy, 42 needs_review, 3.600 distinct_under_policy. Ba near pairs gồm một cặp train trong 7-130 và hai cặp test trong 11-46; không xóa ảnh/đổi nhãn.
- Cả 42 cặp chưa rõ cũng nằm cùng source group/split nên không gây leakage. Không có candidate đi qua hai split trong bán kính đã khai báo.
- Bốn candidates khác source group đều train–train, metrics và ảnh cho thấy khác cảnh. Đã mở đủ 5 review panels gồm 19 cặp: bốn cross-group candidates, 12 cặp gần nhất qua split (kể cả ngoài candidate radius), ba near examples.
- Các giao train–val/train–test/val–test theo source IDs, encoded/pixel hashes, group, camera, plot và duplicate components đều rỗng.

Exact/near edges được gom component liên thông; cặp needs_review qua split hoặc exact duplicate mâu thuẫn nhãn chặn xuất lock/manifests. Manual review phải có reason và file SHA256; không thể override exact duplicate thành distinct. Không có manual override hoặc nới threshold cho run hiện tại.

**File đã tạo/sửa:**

- configs/split_v0.yaml: nhóm/expected counts, isolation camera/plot, source pins, seed và near policy.
- src/agri_vla/split.py: validation, image confirmation, duplicate graph, intersections, deterministic manifests/lock và review panels.
- src/agri_vla/cli.py: thêm split, --dry-run và --output-dir.
- src/agri_vla/model_contract.json: đánh dấu duplicate_component_id/source_split_version là metadata, không là model input.
- tests/test_split.py: 23 tests về group isolation, quarantine, checksum/tamper, dHash collision, exact/pixel/near duplicate leakage, label conflict, manual review, reproducibility, no-overwrite, dry-run và no-ML-import.
- README/spec/context/plan/prompts/status: cập nhật tiến độ, artifacts và giới hạn đánh giá.

Lệnh đã chạy từ root:

~~~bash
AGRI_PY=agri-soybean-vla/.venv/bin/python
$AGRI_PY -m agri_vla.cli preflight --mode data
$AGRI_PY -m agri_vla.cli split --config agri-soybean-vla/configs/split_v0.yaml --dry-run
$AGRI_PY -m agri_vla.cli split --config agri-soybean-vla/configs/split_v0.yaml
$AGRI_PY -m pytest agri-soybean-vla/tests -q --junitxml=agri-soybean-vla/reports/tests-step2.xml
~~~

Kết quả full suite: **69 PASS**. Không import Torch/LeRobot trong split. Chưa train/commit/push; raw và demo/vendor không đổi.

**Artifact:** dưới agri-soybean-vla/data/manifests/source_splits/v0/ (Git ignored).

- [Split report](agri-soybean-vla/data/manifests/source_splits/v0/split_report.json)
- [Split lock](agri-soybean-vla/data/manifests/source_splits/v0/split_lock.json)
- [Train source manifest](agri-soybean-vla/data/manifests/source_splits/v0/train.sources.jsonl)
- [Val source manifest](agri-soybean-vla/data/manifests/source_splits/v0/val.sources.jsonl)
- [Test source manifest](agri-soybean-vla/data/manifests/source_splits/v0/test.sources.jsonl)
- [Excluded source manifest](agri-soybean-vla/data/manifests/source_splits/v0/excluded.sources.jsonl)
- [Group/label stats CSV](agri-soybean-vla/data/manifests/source_splits/v0/group_stats.csv)
- [Candidate metrics/decisions](agri-soybean-vla/data/manifests/source_splits/v0/near_duplicate_pairs.jsonl)
- [Review panel 1](agri-soybean-vla/data/manifests/source_splits/v0/review/near_pairs_01.jpg), còn bốn trang cùng directory.
- [Independent verification](agri-soybean-vla/reports/verification-step2.json)
- [Tests JUnit XML](agri-soybean-vla/reports/tests-step2.xml)

Val chỉ có bốn ảnh gốc wilt 4 do giữ nguyên plot 4-228; không dùng ảnh plot khác để làm đẹp phân bố. Nhiều tổ hợp/paraphrase từ bốn ảnh không tăng số ảnh gốc, và ảnh cùng plot/thời điểm còn tương quan. Test chỉ một plot chưa đại diện các ruộng/camera/mùa vụ khác. Chia ngẫu nhiên scene về sau sẽ làm cùng ảnh nguồn xuất hiện ở nhiều split qua tổ hợp/lệnh khác nhau, gây đánh giá quá lạc quan.

**Điều kiện sang bước 3:** không có blocker cho implement policy/templates Việt–Anh. Giữ nguyên config/lock/source assignments, chưa sinh scene. Bước 4 phải đọc ba *.sources.jsonl đã khóa, không tự chia lại manifest gốc hoặc lấy excluded.sources.jsonl. Muốn chạy lại step 2, chọn --output-dir mới; lock ID không phụ thuộc output override. Dataset ba zone và training vẫn chưa thực hiện.


## Bước 1 — PASS: raw/source audit

Phần này giữ kết quả bàn giao bước 1; tiến độ split hiện tại xem bước 2 ở trên.

Đã xác minh CLI/env bước 0, implement và chạy fetch-source/inspect-source. CSV cục bộ được tái sử dụng sau kiểm tra MD5, ZIP tải mới từ Zenodo; chạy fetch lần nữa xác nhận reuse cả hai file, không tải lại hoặc ghi đè.

| Kiểm tra thật | Kết quả |
|---|---:|
| Dòng CSV / fullID khác nhau | 1.892 / 1.892 |
| File ảnh trong ZIP / đối chiếu có nhãn | 1.892 / 1.892 |
| RGB / kích thước | 1.892 ảnh, tất cả 640×480 |
| Thiếu / thừa / không đọc được / non-RGB | 0 / 0 / 0 / 0 |
| Duplicate tên/đường dẫn ZIP và duplicate source IDs | 0 |
| Duplicate chính xác theo encoded SHA256 / decoded pixel SHA256 | 0 / 0 |
| Đủ điều kiện v0 sau kiểm tra file thật | 1.788 |
| Quarantine ngoài miền nhãn | 104, tất cả annotation=5 tại group 11-46 |
| Nhóm camera/plot | 10 |
| Timestamp trong tên ảnh | 2019-08-09T12:27:11 → 2019-09-22T19:33:27; timezone unknown |
| Contact sheets đã mở xem | 6 nhãn × 12 ảnh = 72 mẫu |
| Full test suite sau bước 1 | 46 PASS, gồm 24 test ingest/source |

Phân bố nhãn 0/1/2/3/4/5 = 524/473/369/216/206/104. Các ảnh annotation=5 cũng đọc được nhưng không đủ điều kiện v0 do miền nhãn, không phải file hỏng. Giữ annotation_raw=5, annotation=5, wilt_score=null, eligible_v0=false; không suy ý nghĩa nhãn 5 hoặc đổi thành 4.

Checksum CSV (83.748 bytes):

~~~text
MD5    538bbf95130b3bc637acfcbed503de64
SHA256 5c4eb0dd55c164ebf8c83fc0d211fb44dbf4647824a2bbeac05988aa9ef02e4f
~~~

Checksum ZIP (655.705.161 bytes; tổng ảnh giải nén 698.237.784 bytes):

~~~text
MD5    73e83e3abed5680d82d03d46984ad2a6
SHA256 35b6a61432eda0d9d0b49c9fa95c0ee49c750400cf502c8eb04e985792506a41
~~~

Các file/source mới và thay đổi:

- configs/dataset_v0.yaml: URLs/MD5/counts, seed, đường dẫn relative-to-config và giới hạn archive.
- src/agri_vla/source.py: config, checksums, fetch/import, dry-run và publish no-overwrite; download lỗi giữ partial.
- src/agri_vla/ingest.py: parse CSV/timestamp, archive guard/CRC/ảnh/hash, source/quarantine reports và contact sheets; materialize nguyên bytes, không re-encode.
- src/agri_vla/cli.py: thêm hai subcommand source bằng lazy imports; data path không import Torch/LeRobot.
- tests/test_ingest.py: fixture sinh trong tmp, gồm header rỗng/sai, label 5, metadata, missing/extra/corrupt/CRC/non-RGB, duplicate, traversal/symlink/size limits, checksum/partial download, no-overwrite, reuse, dry-run và chặn imports ML.
- README/spec/context/plan/prompts/status: bổ sung kết quả thật và hướng dẫn bước 2 đọc đúng source manifest.

Lệnh đã chạy:

~~~bash
AGRI_PY=agri-soybean-vla/.venv/bin/python
$AGRI_PY -m agri_vla.cli preflight --mode data
$AGRI_PY -m agri_vla.cli fetch-source --config agri-soybean-vla/configs/dataset_v0.yaml \
  --local-csv /tmp/agrivla-plan-audit/dataAnns_goodFiles.csv --dry-run
$AGRI_PY -m agri_vla.cli fetch-source --config agri-soybean-vla/configs/dataset_v0.yaml \
  --local-csv /tmp/agrivla-plan-audit/dataAnns_goodFiles.csv
$AGRI_PY -m agri_vla.cli inspect-source --config agri-soybean-vla/configs/dataset_v0.yaml --dry-run
$AGRI_PY -m agri_vla.cli inspect-source --config agri-soybean-vla/configs/dataset_v0.yaml
$AGRI_PY -m agri_vla.cli fetch-source --config agri-soybean-vla/configs/dataset_v0.yaml
$AGRI_PY -m pytest agri-soybean-vla/tests -q --junitxml=agri-soybean-vla/reports/tests-step1.xml
~~~

Đã đối chiếu riêng SHA256 của **từng file ảnh materialized với source manifest**, kiểm tra raw/report bị Git ignore và demo cũ không có diff. Không sửa vendor, không train, không commit/push.

Artifact bàn giao bước 1:

- [Source report JSON](agri-soybean-vla/reports/source/v0/source_report.json)
- [Source manifest — 1.892 record](agri-soybean-vla/reports/source/v0/sources.jsonl)
- [Quarantine — 104 dòng](agri-soybean-vla/reports/source/v0/quarantine.csv)
- [Group stats CSV](agri-soybean-vla/reports/source/v0/group_stats.csv)
- [Archive inventory](agri-soybean-vla/reports/source/v0/archive_inventory.jsonl)
- [Contact sheet nhãn 0](agri-soybean-vla/reports/source/v0/contact_sheets/annotation_0.jpg), các nhãn 1–5 ở cùng thư mục.
- [Fetch/reuse receipt](agri-soybean-vla/reports/source-fetch-step1.json)
- [Kiểm chứng file/Git và visual review](agri-soybean-vla/reports/verification-step1.json)
- [Tests JUnit XML](agri-soybean-vla/reports/tests-step1.xml)

Raw ở agri-soybean-vla/data/raw/zenodo_8256382/. Tất cả raw/generated artifact local bị ignore; code/config/docs/tests được allowlist. Không copy raw vào Git.

**Điều kiện sang bước 2:** không có blocker từ source audit. Đọc reports/source/v0/sources.jsonl, chỉ chọn eligible_v0=true (1.788 record); original_path tương đối với module/base_dir. Có split_group/camera_id/plot_id và dhash64. Chưa kiểm chứng/nhóm near-duplicates, chưa tạo train/val/test, chưa sinh scene/action; những phần đó thuộc bước sau.

Audit default đã tồn tại và được bảo toàn. Muốn audit lại, dùng --output-dir với một thư mục mới; không xóa report cũ. Annotation=5 vẫn chưa rõ semantics; không đưa vào v0 chỉ vì ảnh đọc được.



## Bước 0 đã tạo/sửa gì

- .gitignore: allowlist module và tài liệu bàn giao; .venv/.tools/.vendor/.cache/data/models/outputs/reports vẫn ignore. Source/docs hiện hiển thị untracked trong Git; chưa add/commit/push.
- agri-soybean-vla/pyproject.toml: package agri-vla-soybean 0.1.0, Python 3.12, entrypoint agri-vla, optional dependencies data/dev.
- agri-soybean-vla/src/agri_vla/: CLI preflight/contract, kiểm tra môi trường tách data/train, hợp đồng model JSON. Data CLI không import ML framework.
- agri-soybean-vla/scripts/setup.sh: env riêng, mode data/train, --python/--managed-python/--uv, --dry-run; không xóa env, không đổi vendor đã có.
- agri-soybean-vla/requirements-data.lock.txt: 15 dependencies data/dev với pins + SHA256, đã resolve/cài/kiểm chứng.
- agri-soybean-vla/requirements-train.constraints.txt: recipe chưa resolve/cài, ghi rõ UNVERIFIED.
- agri-soybean-vla/README.md, DATASET_SPEC.md: hướng dẫn lệnh, semantics, song ngữ, policy dự kiến và ranh giới kiểm chứng.
- agri-soybean-vla/schemas/scene_v0.schema.json và examples/scene_pair.example.jsonl: structural schema và một cặp ví dụ Việt–Anh.
- agri-soybean-vla/tests/: schema, song ngữ, invalid inputs, dependency failure, report overwrite protection, no-heavy-import và dry-run non-mutation.
- Context/plan được cập nhật theo kết quả bước 0.

## Môi trường và lệnh đã chạy

Host có Python 3.14.4 trên PATH, thiếu python3.12. Đã dùng uv 0.13.7 tải Python managed và dựng interpreter **CPython 3.12.16** trong .tools/python, virtualenv ở agri-soybean-vla/.venv; không đổi Python hệ thống. Công cụ uv tạm ở /tmp/agrivla-step0-tools. Cache uv/pip được đặt trong module.

Bootstrap thực tế (các biến chỉ áp dụng cho lệnh này):

~~~bash
UV_CACHE_DIR="$PWD/agri-soybean-vla/.cache/uv" \
UV_PYTHON_INSTALL_DIR="$PWD/agri-soybean-vla/.tools/python" \
UV_PYTHON_BIN_DIR="$PWD/agri-soybean-vla/.tools/bin" \
UV_PYTHON_INSTALL_MIRROR=https://releases.astral.sh/github/python-build-standalone/releases/download \
/tmp/agrivla-step0-tools/uv-x86_64-unknown-linux-gnu/uv venv \
  --python 3.12 --managed-python --seed "$PWD/agri-soybean-vla/.venv"
~~~

uv lấy từ máy chủ phát hành chính thức Astral sau khi URL GitHub trực tiếp gặp lỗi CA; không tắt xác minh TLS. Python bootstrap và dependencies nhỏ là các download môi trường duy nhất của bước 0.

Các lệnh kiểm chứng:

~~~bash
bash agri-soybean-vla/scripts/setup.sh --mode data
agri-soybean-vla/.venv/bin/python -m agri_vla.cli --help
agri-soybean-vla/.venv/bin/agri-vla --version
agri-soybean-vla/.venv/bin/python -m agri_vla.cli preflight --mode data
agri-soybean-vla/.venv/bin/python -m agri_vla.cli contract
agri-soybean-vla/.venv/bin/python -m pytest agri-soybean-vla/tests -q \
  --junitxml=agri-soybean-vla/reports/tests-step0.xml
bash agri-soybean-vla/scripts/setup.sh --mode train --dry-run
bash -n agri-soybean-vla/scripts/setup.sh
git diff --check
~~~

Kết quả: data setup/preflight PASS; pip check không có broken requirements; CLI/entrypoint dùng được; **22 tests PASS**. Đã thử Pillow PNG encode/decode trong RAM, PyYAML parse và jsonschema import. Data lock không chứa Torch/LeRobot/Transformers/PEFT.

Train dry-run in đúng commit LeRobot 7e241bd630a3719a56157a497ce5d08f244784f1, extras smolvla,peft,training, CUDA cu126 và constraints; không tạo vendor/download weights/training. Train preflight đã chạy chẩn đoán và trả exit 1 như kỳ vọng: train dependencies/vendor chưa được cài. Đây không phải lỗi của mode data và không phải chứng nhận train đã sẵn sàng.

## Artifact xem được

- [Hướng dẫn module](agri-soybean-vla/README.md)
- [Spec dữ liệu và semantics](agri-soybean-vla/DATASET_SPEC.md)
- [JSON Schema](agri-soybean-vla/schemas/scene_v0.schema.json)
- [Cặp record Việt–Anh minh họa](agri-soybean-vla/examples/scene_pair.example.jsonl)
- [Data preflight JSON](agri-soybean-vla/reports/preflight-step0-data.json)
- [Model contract JSON snapshot](agri-soybean-vla/reports/model-contract-step0.json)
- [Tests JUnit XML](agri-soybean-vla/reports/tests-step0.xml)
- [Log setup data](agri-soybean-vla/reports/setup-step0-data.log)
- [Git boundaries report](agri-soybean-vla/reports/step0-git-boundaries.json): 19 paths được allowlist, 11 paths local/generated bị ignore; demo cũ không thay đổi.
- [Train preflight chưa cài](agri-soybean-vla/reports/preflight-step0-train-not-installed.json)
- Environment snapshot: agri-soybean-vla/reports/environment/data-freeze-1BcSun5t.txt. Đây là snapshot pip freeze của máy hiện tại; editable entry có thể tham chiếu Git HEAD không chứa các file chưa commit. Không dùng snapshot này thay cho lock + source hiện tại.

Reports/venv/cache/interpreter local bị ignore nên không có trong clean clone. Cần chạy lại các lệnh để tạo bằng chứng ở máy khác.

## Giới hạn bàn giao bước 0 (lịch sử)

Các ghi nhận dưới đây mô tả thời điểm kết thúc bước 0; tiến độ hiện tại xem phần bước 1 ở trên.

Không có blocker cho bước 1 ở môi trường data đã kiểm chứng. Bước 1 cần đọc spec/schema này, dùng env riêng hiện có, rồi thực hiện Prompt 1 để tải hoặc nhận local CSV/ZIP. Bước 0 chưa tải ZIP, chưa tạo data/raw, chưa implement fetch-source/inspect-source.

Phải giữ quarantine nhãn 5, split theo nguồn trước compose, không lộ teacher vào model và dataset song ngữ Việt–Anh theo plan. Schema pass chưa kiểm chứng image existence, pair semantics, labels hoặc split leakage.

Train constraints chỉ mới chuẩn bị; resolver/imports LeRobot/API cần kiểm chứng ở bước 5. GPU/CUDA, horizon một bước, ba camera, normalization runtime, memory 4 GB và checkpoint resume phải kiểm chứng các bước 6–7. Chưa tải weights, chưa train, chưa flash/điều khiển phần cứng. Demo LIBERO, demo đèn và vendor cũ giữ nguyên.

