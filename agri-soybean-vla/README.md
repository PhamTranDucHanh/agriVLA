# AgriVLA-Soybean-v0

**Bản hiện hành: wording_v1** — 3.600 train / 720 val / 720 test records. Có câu Việt–Anh ngắn, văn nói và intent WATER_ALL(level=1/2/3): “Tưới cả 3 vườn mức 2 nhé” → [2,2,2] bất kể héo. “Tắt hết.” → [0,0,0]. Sáu intent cũ giữ target; v0 và lock cũ được giữ nguyên để đối chiếu.

**Browser đang dùng bản mới:** `bash agri-soybean-vla/scripts/browse.sh`, mở **http://127.0.0.1:8765/?view=samples&page_size=48**. Có đủ 5.040 records, 1.892 ảnh nguồn, filter WATER_ALL/STOP, link ảnh/thư mục gốc. Nếu server cũ còn chạy: `bash agri-soybean-vla/scripts/browse.sh --replace`, chờ QA PASS rồi Ctrl+Shift+R. Xem [hướng dẫn duyệt](BROWSE_DATASET.md).

**Bước6 đã PASS_CPU:** strict base load/fresh LoRA371.328 trainable parameters, tokenizer96 zero truncation5040records, saved processors, CPU forward/inference, checkpoint fixture và trainer dry-run đã kiểm. Regression530PASS +1native-module skip; native2PASS. Chưa optimizer/backward/GPUfit.

**Model và chuyển máy GPU:** [MODEL_TRAINING.md](MODEL_TRAINING.md), [GPU_HANDOFF.md](GPU_HANDOFF.md). Train config dùng **data/lerobot/wording_v1/full/train**, tokenizer 96, fresh LoRA r=8/alpha=8 và stats wording_v1 full train. GPU smoke/optimizer/save–resume vẫn thuộc bước 7; máy hiện tại chỉ kiểm CPU.

Module tạo dataset ba vùng tổng hợp từ ảnh Soybean và chuẩn bị fine-tune SmolVLA theo lệnh **tiếng Việt và tiếng Anh**.

**Lịch sử bước 0–5 (dataset v0 gốc), gồm browser 4.5:** dataset JSONL và LeRobot v3 đã QA/round-trip. Có 3.000 train / 600 val / 600 test record và tương ứng singleton episodes; số ảnh nguồn thực sự dùng là 1.258 / 173 / 254. 104 ảnh nhãn 5 vẫn bị loại. Môi trường ML riêng ở .venv-train, data/browser giữ .venv. Bước 6 đã tải base weights đúng revision và chuẩn bị fresh LoRA; chưa optimizer training.

**LeRobot v0 gốc đã convert (2026-11-06), giữ làm lịch sử:** xem [CONVERSION.md](CONVERSION.md) để biết env, lệnh, roots và batch thật. Smoke: data/lerobot/v0/smoke (64 episodes). Full: data/lerobot/v0/full/{train,val,test} (3.000/600/600 episodes, một frame/episode). `verify-lerobot` đã đọc lại mọi record, đúng 12.600 ảnh; dùng train_stats.json và normalization_contract.json ở full/. Bước 6 phải xử lý đúng observation history axis [B,1,C,H,W]/state [B,1,3], action [B,1,3], STATE IDENTITY và ACTION MEAN_STD từ full train.

Đọc [DATASET_SPEC.md](DATASET_SPEC.md) để hiểu input/target/metadata. Kế hoạch và prompt đầy đủ nằm ở root repo. Tiến độ thực tế được ghi trong [AGRIVLA_IMPLEMENTATION_STATUS.md](../AGRIVLA_IMPLEMENTATION_STATUS.md).

## Cài môi trường data

Chạy từ root repo trên Linux, nhánh feat/compose-dataset:

~~~bash
bash agri-soybean-vla/scripts/setup.sh --mode data
AGRI_PY=agri-soybean-vla/.venv/bin/python
$AGRI_PY -m agri_vla.cli --help
$AGRI_PY -m agri_vla.cli preflight --mode data
$AGRI_PY -m agri_vla.cli contract
$AGRI_PY -m pytest agri-soybean-vla/tests -q
~~~

Cần Python 3.12 với venv. Nếu Python 3.12 nằm ngoài PATH:

~~~bash
bash agri-soybean-vla/scripts/setup.sh --mode data --python /path/to/python3.12
~~~

Nếu chỉ có phiên bản Python khác, có thể dùng uv để dựng Python 3.12 trong module:

~~~bash
bash agri-soybean-vla/scripts/setup.sh --mode data --managed-python --uv /path/to/uv
~~~

Thay các đường dẫn /path/to/... bằng executable thực tế. Script không cài uv tự động. [Cách cài uv chính thức](https://docs.astral.sh/uv/getting-started/installation/); uv có thể [quản lý Python](https://docs.astral.sh/uv/guides/install-python/) khi máy thiếu phiên bản cần dùng. Cache/interpreter do mode managed tạo nằm trong .cache/ và .tools/ của module, không đổi Python hệ thống.

Mode data cài Pillow (đọc ảnh), PyYAML (config), jsonschema (kiểm tra cấu trúc), pytest (tests) và công cụ build nhỏ. CLI data dùng standard library, không import Torch, LeRobot, Transformers hoặc PEFT. Môi trường .venv của demo cũ không bị thay đổi.

## CLI môi trường và hợp đồng

- preflight: kiểm tra Python 3.12, virtualenv đúng module, Linux, nhánh Git, disk, khả năng ghi và dependency data. PASS chỉ xác nhận môi trường, chưa xác nhận ảnh/dataset/model hoặc khả năng train 4 GB.
- contract: in JSON liệt kê feature được đưa vào model, target và metadata phải tách riêng.

~~~bash
$AGRI_PY -m agri_vla.cli preflight --mode data --json
$AGRI_PY -m agri_vla.cli preflight --mode data --output agri-soybean-vla/reports/preflight-data.json
agri-soybean-vla/.venv/bin/agri-vla --version
~~~

--output chỉ tạo file mới; không ghi đè report đã có. Exit code: 0 = đạt các kiểm tra bắt buộc; 1 = preflight chưa đạt; 2 = sai tham số hoặc không ghi được report. Cảnh báo disk không tự xóa dữ liệu.

## Mode train: đã cài và kiểm chứng CPU ở bước 5

Xem các lệnh sẽ chạy, không tải/cài hay tạo thư mục:

~~~bash
bash agri-soybean-vla/scripts/setup.sh --mode train --dry-run
~~~

Mode train dùng .venv-train riêng, giữ nguyên .venv data/browser:

~~~bash
bash agri-soybean-vla/scripts/setup.sh --mode train
AGRI_TRAIN_PY=agri-soybean-vla/.venv-train/bin/python
$AGRI_TRAIN_PY -m agri_vla.cli preflight --mode train
# Chỉ bắt buộc GPU ở bước kiểm chứng CUDA:
$AGRI_TRAIN_PY -m agri_vla.cli preflight --mode train --require-cuda
~~~

Mode train chuẩn bị LeRobot v0.6.1, commit 7e241bd630a3719a56157a497ce5d08f244784f1, trong .vendor/lerobot riêng của module. Nếu checkout có sẵn khác commit, script dừng và giữ nguyên. Cài extras smolvla,peft,training và PyTorch CUDA 12.6 theo constraints. Mode này không tải model weights hoặc raw dataset, không bắt đầu training.

requirements-train.constraints.txt là core constraints đã kiểm, requirements-train.lock.txt ghi đầy đủ versions đã resolve/cài trên Python 3.12/Linux x86_64. pip check và imports đã PASS; CUDA hiện unavailable là WARN ở bước 5, GPU smoke vẫn thuộc bước 7. Không dùng CPU conversion PASS để tuyên bố train đã fit 4 GB.

## Tái lập và bảo toàn dữ liệu

requirements-data.lock.txt được resolve từ optional dependencies data/dev trong pyproject.toml, gồm phiên bản và hashes. setup dùng --require-hashes và cài project editable với --no-deps. Khi đổi dependencies, phải regenerate lock, cài lại và chạy tests.

Lệnh dành cho người bảo trì, không cần chạy khi cài thông thường:

~~~bash
UV_BIN=/path/to/uv
"$UV_BIN" pip compile agri-soybean-vla/pyproject.toml --extra data --extra dev \
  --python agri-soybean-vla/.venv/bin/python --generate-hashes \
  --no-header --output-file agri-soybean-vla/requirements-data.lock.txt
~~~

Sau setup thành công, snapshot pip freeze nằm trong reports/environment/. Snapshot có thể chứa đường dẫn editable theo máy; đây là bằng chứng môi trường đã cài, không thay thế portable lock. Train-mode lock đầy đủ sẽ chỉ được tạo khi đã cài/kiểm chứng.

Git chỉ nhận source/config/docs/tests và hai record ví dụ minh họa. .venv, .venv-train, .tools, .vendor, .cache, data, models, outputs và reports bị ignore. Script không rmtree, không reset checkout và không xóa directory khi lỗi.

## Cấu trúc hiện tại

~~~text
pyproject.toml                         Package và dependency declarations
requirements-data.lock.txt            Data/dev pins + hashes
requirements-train.constraints.txt    Core compatibility constraints đã kiểm
requirements-train.lock.txt           Full resolved train dependency versions
src/agri_vla/cli.py                    --help, preflight, contract
src/agri_vla/preflight.py              Kiểm tra môi trường theo mode
src/agri_vla/model_contract.json       Feature/target/metadata contract
scripts/setup.sh                      Setup riêng, hỗ trợ --dry-run
schemas/scene_v0.schema.json           Schema cấu trúc manifest tương lai
examples/scene_pair.example.jsonl      Một cặp Việt–Anh, paths giả minh họa
tests/                                Kiểm tra bootstrap và schema
~~~

## Bước 1: tải và kiểm tra nguồn

Config [dataset_v0.yaml](configs/dataset_v0.yaml) chứa URLs, MD5 công bố, counts kỳ vọng, seed, giới hạn ZIP và đường dẫn. base_dir tương đối với file YAML; các đường dẫn data/report tương đối với base_dir, không phụ thuộc shell cwd.

~~~bash
AGRI_PY=agri-soybean-vla/.venv/bin/python
$AGRI_PY -m agri_vla.cli fetch-source --config agri-soybean-vla/configs/dataset_v0.yaml --dry-run
$AGRI_PY -m agri_vla.cli fetch-source --config agri-soybean-vla/configs/dataset_v0.yaml
$AGRI_PY -m agri_vla.cli inspect-source --config agri-soybean-vla/configs/dataset_v0.yaml --dry-run
$AGRI_PY -m agri_vla.cli inspect-source --config agri-soybean-vla/configs/dataset_v0.yaml
~~~

Trong workspace hiện tại audit mặc định đã tồn tại, nên lệnh inspect-source cuối sẽ từ chối chạy lại vào cùng report directory. Muốn audit lại, chọn thư mục **mới**:

~~~bash
$AGRI_PY -m agri_vla.cli inspect-source --config agri-soybean-vla/configs/dataset_v0.yaml \
  --output-dir agri-soybean-vla/reports/source/v0-recheck
~~~

Nếu đã có raw ở nơi khác, fetch-source hỗ trợ --local-csv /path/to/dataAnns_goodFiles.csv và --local-zip /path/to/full_data.zip. Các file local được xác minh trước khi copy. Raw canonical đã có và khớp hash sẽ được reuse; sai hash thì dừng, giữ nguyên. Download ghi vào file .part-* riêng, chỉ publish tên cuối khi MD5 khớp. Download lỗi giữ partial để kiểm tra; không tự xóa hoặc ghi đè raw. --dry-run không tải mạng, không tạo directory/file.

inspect-source kiểm tra checksum, header có cột rỗng, ID/group/timestamp, archive paths/symlink/size limits, missing/extra/duplicate filenames, CRC, giải mã toàn bộ ảnh, RGB, dimensions, SHA256 file và pixel. Nhãn 5 giữ nguyên; không relabel. Các ảnh đọc được được materialize **nguyên bytes gốc** vào data/raw/zenodo_8256382/images, không resize/re-encode. File ảnh có sẵn chỉ reuse khi SHA256 khớp.

Exit code của lệnh source: 0 = pass/dry-run; 1 = audit hoàn tất nhưng có QA/count mismatch; 2 = input/config/checksum/unsafe archive/output conflict hoặc lỗi I/O. Audit có lỗi ở từng ảnh vẫn ghi report/quarantine để kiểm tra. Lỗi cấu trúc CSV/archive nguy hiểm dừng trước materialize; không tạo một báo cáo pass giả.

## Artifact nguồn đã tạo

| Artifact local | Nội dung |
|---|---|
| data/raw/zenodo_8256382/dataAnns_goodFiles.csv | CSV gốc, MD5 đã kiểm chứng |
| data/raw/zenodo_8256382/full_data.zip | ZIP gốc, 655.705.161 bytes, MD5 đã kiểm chứng |
| data/raw/zenodo_8256382/images/ | 1.892 ảnh nguyên bytes, gồm cả ảnh quarantine |
| reports/source/v0/sources.jsonl | Một record cho mỗi dòng CSV, tổng 1.892 record |
| reports/source/v0/quarantine.csv | 104 dòng nhãn 5 với lý do loại khỏi v0 |
| reports/source/v0/source_report.json | Checksum, QA, counts, dimensions, dates, duplicates và stats theo group |
| reports/source/v0/group_stats.csv | Rows/eligible/labels và khoảng timestamp của từng group |
| reports/source/v0/archive_inventory.jsonl | Tên member và kích thước từng file trong ZIP |
| reports/source/v0/contact_sheets/annotation_0.jpg ... annotation_5.jpg | 12 ảnh mẫu mỗi nhãn, seed=42; nhãn 5 ghi QUARANTINE |

sources.jsonl là **source manifest**, không phải scene manifest và chưa là input batch của model. Các field quan trọng:

- source_image_id: fullID gốc; csv_line/csv_values giúp truy nguyên CSV.
- annotation_raw/annotation: giữ giá trị gốc; wilt_score chỉ có 0–4, ngoài miền đặt null, không đổi nhãn.
- eligible_v0: cờ bắt buộc để chọn ảnh cho bước 2. **Chỉ đọc record có eligible_v0=true**, không lấy cả 1.892 dòng.
- qa_status/qa_issues/exclusion_reasons: hợp lệ, lỗi ảnh/metadata hoặc quarantine theo nhãn.
- split_group/treatment_camera, camera_id, plot_id: định danh nguồn để chia group sau này.
- original_path: đường dẫn ảnh tương đối với module/base_dir; archive_member: đường dẫn nguyên gốc trong ZIP.
- timestamp: parse DD_MM_YYYY_HH_MM_SS; timestamp_timezone=unknown, không tự gán UTC.
- sha256/pixel_sha256: duplicate chính xác; dhash64: metadata hỗ trợ bước kiểm tra near-duplicates sau này, chưa có kết luận/grouping near-duplicates ở bước 1.

Kết quả audit hiện tại: không missing/extra/unreadable/non-RGB, không duplicate chính xác. Khoảng timestamp 2019-08-09 đến 2019-09-22. 104 ảnh nhãn 5 đều thuộc group 11-46 và có ảnh đọc được; bị loại vì miền nhãn, không phải vì file hỏng. Chi tiết checksum nằm trong report/status.

## Bước 2: group split và khóa chống leakage

[split_v0.yaml](configs/split_v0.yaml) pin SHA256 source manifest/report bước 1 và khai báo nhóm nguồn tường minh. Không chia ngẫu nhiên ảnh hoặc scene. Camera 5 có hai plot, cả hai cùng train. Seed 42 dùng cho chọn ví dụ chẩn đoán; assignment là cố định theo config, không ngẫu nhiên theo seed.

| Split | Nhóm | Ảnh | Wilt 0 / 1 / 2 / 3 / 4 |
|---|---|---:|---|
| Train | 8 nhóm, gồm 10-87 | 1.258 | 466 / 343 / 248 / 65 / 136 |
| Val | 4-228 | 189 | 21 / 48 / 49 / 67 / 4 |
| Test | 11-46 sau quarantine | 341 | 37 / 82 / 72 / 84 / 66 |

Lệnh đã chạy lần đầu:

~~~bash
$AGRI_PY -m agri_vla.cli split --config agri-soybean-vla/configs/split_v0.yaml --dry-run
$AGRI_PY -m agri_vla.cli split --config agri-soybean-vla/configs/split_v0.yaml
~~~

Output hiện đã tồn tại, nên muốn tái kiểm tra phải chọn thư mục mới. --dry-run thực hiện đầy đủ kiểm tra source/hash/ảnh/gần trùng, nhưng không ghi file hoặc tải mạng:

~~~bash
$AGRI_PY -m agri_vla.cli split --config agri-soybean-vla/configs/split_v0.yaml \
  --output-dir agri-soybean-vla/data/manifests/source_splits/v0-check --dry-run
$AGRI_PY -m agri_vla.cli split --config agri-soybean-vla/configs/split_v0.yaml \
  --output-dir agri-soybean-vla/data/manifests/source_splits/v0-check
~~~

### Ảnh gần trùng được kiểm tra thế nào

Mỗi ảnh eligible được đối chiếu lại file SHA256, pixel SHA256 và dHash với bước 1. Mọi cặp được tìm candidate bằng khoảng cách Hamming dHash64 ≤8. Đây chỉ là bước tìm kiếm, không tự kết luận trùng.

Trên thumbnail chẩn đoán RGB 64×48, một candidate được xem là near_duplicate theo policy khi RGB MAE/255 ≤0.035, tương quan Pearson ảnh xám ≥0.985 và std ảnh xám/255 ≥0.02. Vùng chưa rõ gồm tương quan ≥0.94 và MAE ≤0.10, hoặc ảnh ít texture nhưng màu gần nhau. Threshold là heuristic có version trong config; không phải một bảo đảm phát hiện mọi kiểu crop/rotation/biến dạng.

Exact file/pixel duplicates và các cạnh near_duplicate tạo component liên thông. Cùng component phải ở cùng split, kể cả khi liên kết bắc cầu. Cặp needs_review qua split sẽ chặn xuất lock/manifests; không tự loại candidate để làm cho split pass. Exact duplicate có nhãn mâu thuẫn cũng chặn, không tự sửa nhãn. Candidate khác plot còn được đối chiếu ảnh chẩn đoán; các cặp cross-split gần nhất ngoài bán kính dHash cũng được xuất để xem mẫu.

Kết quả thật: 3.645 candidates → 3 near_duplicate, 42 needs_review và 3.600 distinct_under_policy. Cả 3 near pairs và 42 cặp chưa rõ đều trong cùng source group/split. Bốn candidates khác source group đều là train–train và khác cảnh khi xem; không có candidate qua split trong bán kính đã khai báo. Đã xem 19 cặp trên 5 review panels, gồm cả 12 ví dụ gần nhất qua split và 3 near pairs. Không cần đổi assignment; không xóa/bớt ảnh.

Nếu dữ liệu mới phát sinh xung đột, split sẽ FAIL và chỉ ghi chẩn đoán, chưa xuất lock hoặc manifest train/val/test. Có thể sửa assignment của **toàn bộ nhóm liên quan** hoặc dùng manual review có lý do và pinned SHA256. Review không thể phủ nhận exact duplicate. File review là JSON array với source_image_ids[2], decision=near_duplicate|distinct và reason; cặp ngoài candidate policy phải được xét lại bằng config/threshold công bố, không bypass ngầm.

### Artifact split

Tất cả nằm dưới data/manifests/source_splits/v0/ và bị Git ignore:

| File | Nội dung |
|---|---|
| train.sources.jsonl / val.sources.jsonl / test.sources.jsonl | Source records đủ điều kiện, thêm split và duplicate_component_id |
| excluded.sources.jsonl | 104 record bị loại, giữ nguyên metadata source |
| split_lock.json | Config/source hashes, seed, policy, assignments và hashes của bốn manifests |
| split_report.json | Counts/classes, giao ID/hash/group/camera/plot/component, near audit và giới hạn |
| group_stats.csv | Nhãn của từng group và split |
| near_duplicate_pairs.jsonl | Metrics/quyết định của toàn bộ 3.645 candidate pairs |
| review/near_pairs_01.jpg ... near_pairs_05.jpg | Các cặp ảnh chẩn đoán đã xem |

SHA256 của split_lock.json: 3be04c67ed375c76e4a97d425416e3a890bd1c060ce1167b9bbd5f16d6fcaa8d. Dry-run và run thật khớp lock này. Các giao giữa split theo source ID, encoded/pixel hash, camera, plot và duplicate_component_id đều rỗng. Model không nhận duplicate_component_id hoặc source_split_version; đó là metadata.

Val có **bốn ảnh gốc mức héo 4** vì cả plot 4-228 được giữ nguyên. Không đưa ảnh từ plot khác vào val chỉ để tăng lớp hiếm. Sinh nhiều paraphrase/scene từ bốn ảnh vẫn không tăng số ảnh gốc; các ảnh cùng plot/chuỗi thời gian cũng không mặc nhiên là các quan sát thống kê độc lập. Test chỉ có một plot 11-46 nên chưa chứng minh khái quát cho mọi ruộng/camera/mùa vụ.

Split trước khi compose là bắt buộc: chia ngẫu nhiên scene có thể để cùng ảnh nguồn ở cả train và test qua những tổ hợp/câu lệnh khác nhau, làm metric quá lạc quan.

## Policy và templates Việt–Anh — bước 3

Policy [policy.py](src/agri_vla/policy.py) tính nhãn deterministic, không gọi model/LLM. [policy_v0.yaml](configs/policy_v0.yaml) pin semantics wilt_proxy_policy_v0: 0→0, 1→0, 2→1, 3→2, 4→3. Wilt từ 2 mới đạt ngưỡng tưới; wilt 1 vẫn OFF. Mức 0/1/2/3 là OFF/LOW/MEDIUM/HIGH tổng hợp, chưa phải lít hoặc thời gian tưới đã hiệu chuẩn. Dữ liệu quan sát héo lá, không đo đất khô nhất.

| Intent | Params | Xử lý |
|---|---|---|
| AUTO | Không | Mapping cả ba zone |
| WORST_ONLY | Không | Giữ tất cả zone hòa max khi max ≥2; nếu không toàn OFF |
| ZONE_ONLY | zone=1/2/3 | Chỉ giữ zone được chọn |
| CAP_LEVEL | cap=1/2 | Giới hạn từng mapped level bằng cap |
| EXCLUDE_ZONE | zone=1/2/3 | Tắt zone được loại |
| STOP_ALL | Không | Luôn [0,0,0] với input hợp lệ |

Validation từ chối intent lạ/ghép nhiều intent, thiếu/thừa/sai params, bool/float thay int, wilt ngoài 0–4, YAML trùng key và config mâu thuẫn v0. Không suy đoán intent từ câu tự do. Mọi target phải từ policy đã version, không do LLM chọn.

Lệnh từ repo root:

~~~bash
AGRI_PY=agri-soybean-vla/.venv/bin/python
$AGRI_PY -m agri_vla.cli explain-policy --help
$AGRI_PY -m agri_vla.cli explain-policy --wilt 0 2 4
$AGRI_PY -m agri_vla.cli explain-policy --wilt 0 2 4 --intent CAP_LEVEL --cap 2 --json
$AGRI_PY -m agri_vla.cli explain-policy --wilt 4 4 1 --intent WORST_ONLY --split test --wording unseen
# Không ghi file:
$AGRI_PY -m agri_vla.cli audit-policy
# Chọn output CHƯA tồn tại; dry-run không tạo directory:
$AGRI_PY -m agri_vla.cli audit-policy --output-dir /tmp/agrivla-policy-review-new --dry-run
$AGRI_PY -m agri_vla.cli audit-policy --output-dir /tmp/agrivla-policy-review-new
$AGRI_PY -m pytest agri-soybean-vla/tests -q
~~~

explain-policy không truyền intent sẽ in sáu ví dụ Việt/Anh với zone=2, cap=1, exclude=3. Wilt [0,2,4] tương ứng AUTO [0,1,3], WORST_ONLY [0,0,3], ZONE_ONLY(2) [0,1,0], CAP_LEVEL(1) [0,1,1], EXCLUDE_ZONE(3) [0,1,0], STOP_ALL [0,0,0]. Wilt [4,4,1] + WORST_ONLY → [3,3,0], còn [1,1,0] → [0,0,0].

[language_v0.yaml](configs/language_v0.yaml) có 60 paired families/120 templates gốc, mỗi intent 6 train + 2 unseen-val + 2 unseen-test. Sau điền params có 220 câu, chia 132/44/44; mỗi pool cân bằng vi/en. Các family và câu chuẩn hóa không giao nhau. Val/test seen dùng train pool; unseen dùng riêng pool của split. Bản dịch đi cùng family, không dùng một ngôn ngữ train rồi coi bản dịch held-out là wording mới.

API load_policy(path).action(wilt, intent, params) tính target; load_registry(path).render_pair(family_id, params, split=...) tạo hai instruction. Renderer không nhận wilt, không sinh target hoặc translation_pair_id theo scene. Chỉ instruction trở thành model text; wilt và structured intent/params nằm ở metadata phục vụ teacher/QA để tránh cho model đọc sẵn thông tin tính đáp án. State [0,0,0] giả định mọi van ban đầu đóng, không phải số đo độ ẩm.

CLI choose_pair dùng RNG cục bộ seed=42 để tái lập ví dụ. Bước 4 phải cân bằng family hoặc dẫn xuất seed theo sample, kiểm mọi seen-family thực sự có trong train và tạo translation_pair_id theo scene/intent/params/family. Lexical guards kiểm slots/threshold/tie/teacher leakage cộng với curated review; không chứng minh tương đương mọi câu ngôn ngữ tự do, tokenizer/truncation hoặc model hiểu tiếng Việt.

Artifact thật đã tạo (ignored, mở local): [report](reports/policy/v0/policy_report.json), [lock](reports/policy/v0/policy_lock.json), [1.375 policy cases](reports/policy/v0/policy_truth_table.csv), [220 câu](reports/policy/v0/template_catalog.csv), [sáu cặp câu/target](reports/policy/v0/policy_examples.md), [verification](reports/verification-step3.json), [282 tests PASS](reports/tests-step3.xml). Audit table/catalog là diagnostic, không phải scene dataset. Source split lock và bốn manifest hashes giữ nguyên.

## Compose, validate, preview — bước 4

Dataset thật ở [data/manifests/v0](data/manifests/v0), tách khỏi source manifests bước 2. [DATASET_CARD.md](DATASET_CARD.md) ghi nguồn, counts, policy, split, reuse và giới hạn. Mỗi dòng JSONL là một record; ảnh tham chiếu nguyên bytes tại paths tương đối với **module/base_dir**, không phải thư mục JSONL. Không copy, re-encode hoặc vẽ nhãn lên ảnh dùng train.

| Split | Ảnh nguồn có sẵn/đã dùng | Ordered scenes | Cases | Records | Action toàn OFF |
|---|---:|---:|---:|---:|---:|
| train | 1.258/1.258 | 1.440 | 1.500 | 3.000 | 684 (22,8%) |
| val | 189/173 | 90 | 150 | 600 | 212 (35,3%) |
| test | 341/254 | 90 | 150 | 600 | 208 (34,7%) |

Ordered scene là bộ ba ảnh theo đúng thứ tự zone; case là scene + intent/params. Train mỗi case có hai bản dịch; val/test mỗi case có vi/en × seen/unseen = bốn record. Cùng scene có thể có nhiều lệnh đối chiếu nên số cases lớn hơn scenes. Các tổ hợp, bản dịch và ảnh cùng plot/thời điểm không tăng số quan sát thống kê độc lập. Val vẫn chỉ có bốn ảnh gốc wilt 4, test chỉ một plot.

Generator seed=42 giữ quota train AUTO/WORST/ZONE/CAP/EXCLUDE/STOP = 450/300/225/225/150/150 cases. Mỗi val/test = 30/30/30/20/20/20 cases. Zone/cap cân bằng chênh tối đa một case. Chọn ảnh theo số lần dùng và vị trí zone, không chia lại source split. Cùng scene luôn có ba IDs/hash/pixel/component khác nhau và không có cặp near/uncertain theo heuristic bước 2. Bảng reuse cho từng ảnh nằm trong qa_report.json; val/test không bắt buộc dùng hết ảnh khi panel cố định nhỏ hơn nguồn.

Có panel cùng ảnh/khác lệnh, cùng câu/khác ảnh, 11 ca hoán vị có remap zone, all-OFF/all-high/tie/STOP/exclusion. Family được phân phối theo từng intent/params: cả 66 tổ hợp train family–params xuất hiện ở mỗi ngôn ngữ; seen val/test được đối chiếu với train thực tế. Eval unseen families giữ đúng pool riêng.

Lệnh tạo lần đầu, từ repo root (mọi output phải chưa tồn tại):

~~~bash
AGRI_PY=agri-soybean-vla/.venv/bin/python
$AGRI_PY -m agri_vla.cli compose --config agri-soybean-vla/configs/dataset_v0.yaml --dry-run
$AGRI_PY -m agri_vla.cli compose --config agri-soybean-vla/configs/dataset_v0.yaml
$AGRI_PY -m agri_vla.cli validate --manifests agri-soybean-vla/data/manifests/v0
$AGRI_PY -m agri_vla.cli preview --manifests agri-soybean-vla/data/manifests/v0 --output-dir agri-soybean-vla/data/manifests/v0/review --count 72
$AGRI_PY -m pytest agri-soybean-vla/tests -q
~~~

Dataset/preview đã tồn tại trên máy này nên compose/preview lại sẽ từ chối, kể cả dry-run. Muốn tái lập, thêm --output-dir với một directory mới, ví dụ /tmp/agrivla-scenes-review-new. validate mặc định chỉ in report; --output lưu file mới, --dry-run không ghi file. Generated output trong raw bị chặn. Không có --overwrite hoặc xóa directory tự động.

[Mẫu đầy đủ thật](reports/example-step4-record.json): wilt [0,2,4], ZONE_ONLY(zone=2), câu “Chỉ tưới vùng 2 theo mức quy định nếu đạt ngưỡng héo.” → [0,1,0]. Các field:

| Fields | Ý nghĩa |
|---|---|
| schema_version / scene_type / split / seed | Phiên bản, synthetic_three_zone, phân vùng nguồn và seed |
| sample_id / scene_id / case_id | Dòng ngôn ngữ cụ thể / bộ ba ảnh có thứ tự / scene + intent/params |
| images / source_image_ids / source_groups | Ba đường dẫn gốc theo zone, IDs và nhóm nguồn tương ứng |
| instruction / language | Câu duy nhất của một ngôn ngữ; instruction trở thành task của model |
| intent / intent_params / teacher_annotations | Ý định có cấu trúc và điểm héo từ nguồn; chỉ teacher metadata |
| target_action / label_source / label_type / policy_version | Nhãn action proxy, nguồn và phiên bản policy tính nhãn |
| template_id / template_family_id / translation_pair_id / wording_partition | Template cụ thể, family song ngữ, cặp dịch theo case và seen/unseen |
| provenance | Hash config, hai upstream locks, source manifest, ba encoded/pixel hashes và duplicate components |
| generation_metadata | Loại ca kiểm thử; permutation_of và permutation one-based nếu có hoán vị |

QA độc lập kiểm schema, source membership/paths/ảnh/checksums, teacher, recomputed action, pair/quartet equivalence, quotas, template coverage, source/hash/group/camera/plot/component leakage, hoán vị và semantic hashes. Model contract chỉ cho ảnh + instruction + state zeros; metadata mới đều nằm ở metadata_only. Serialization/round-trip đã PASS ở bước 5; policy forward/backward vẫn phải kiểm ở bước 6–7.

Mở [preview HTML](data/manifests/v0/review/index.html) trực tiếp bằng trình duyệt; không cần server hoặc mạng. Có ba ảnh, lệnh, teacher/target và full JSON. Contact sheets là ảnh review riêng, không là input training. Xem [QA report](data/manifests/v0/qa_report.json), [dataset lock](data/manifests/v0/dataset_lock.json), [JUnit](reports/tests-step4.xml) và [status bàn giao](../AGRIVLA_IMPLEMENTATION_STATUS.md).

Bước 5 conversion/round-trip đã PASS; xem CONVERSION.md. Bước tiếp theo là bước 6 (model/processor/fresh LoRA), sau đó bước 7 mới chứng nhận train GPU 4 GB. Không tự bắt đầu train.

