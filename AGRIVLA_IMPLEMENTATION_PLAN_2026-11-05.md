# AgriVLA-Soybean-v0 — kế hoạch dataset và fine-tune trên GPU 4 GB

Ngày: 2026-11-05. Nhánh triển khai: `feat/compose-dataset`.

**Trạng thái tài liệu:** kế hoạch tổng thể và tiến độ bàn giao. Bước0–6 và browser4.5 đã chạy thật ở phạm vi CPU; CLI model/prepare/infer/evaluate/train-upstream đã có, bước7–9 chưa hoàn thành. Dataset JSONL và LeRobot v3 đã QA/round-trip CPU; VRAM và chất lượng model còn phải đo.

**Tiến độ cập nhật:** bước 0–5 và 4.5 đã hoàn thành môi trường/source audit/group split/policy/templates/compose/browser và LeRobot conversion/round-trip CPU. Source split lock `3be04c67ed375c76e4a97d425416e3a890bd1c060ce1167b9bbd5f16d6fcaa8d` và policy lock `c6c0d5d2be39fb61581a10aa39757e03b840518ddf6c210a372c5fb2853b4d11` không đổi. Dataset ở `agri-soybean-vla/data/manifests/v0/` có 3.000/600/600 records, dùng 1.258/173/254 ảnh nguồn; 104 nhãn 5 excluded. Dataset lock SHA256 `c59005a5d4eea36731051f5b82a9341d9be5ae50cca4aafcad1055743ef16563`. Bước6 đã PASS_CPU; các bước7–9 chưa thực hiện. Xem [status thực tế](AGRIVLA_IMPLEMENTATION_STATUS.md) để biết tests/visual review và hạn chế. Mode train .venv-train đã cài/import/pip-check; chưa chứng nhận GPU 4 GB.

Để giao từng bước cho một model mới, dùng [bộ prompt độc lập](AGRIVLA_IMPLEMENTATION_PROMPTS_2026-11-05.md). Sau mỗi bước, ghi kết quả thật vào `AGRIVLA_IMPLEMENTATION_STATUS.md`; tạo file này ở bước 0. Chỉ làm bước được giao, không tự chạy bước kế tiếp hay run train dài.

**Cập nhật 2026-11-06 — bước 5 PASS:** env ML riêng .venv-train với LeRobot đã pin/extras và dependencies lock đã cài/import/pip-check; .venv vẫn data-only. LeRobot ở data/lerobot/v0/smoke (64) và full/{train,val,test} (3.000/600/600 singleton episodes); verifier độc lập đã kiểm tất cả records/12.600 ảnh, source lock không đổi. Native factory batch camera [B,1,3,256,256], state/action [B,1,3], masks false và sampler không drop. full/train_stats.json + normalization_contract.json bắt buộc STATE IDENTITY/ACTION MEAN_STD train-only. Dataset copy/normalization CPU đã PASS, model/GPU vẫn chưa; xem CONVERSION.md/status để biết artifacts, 363 data tests + 2 native tests và giới hạn.

**Overlay hiện hành sau bước6 (2026-11-06):** dùng wording_v1 **3600/720/720**, lock92111dfa8086eef77b90e9ac47003c43f1847ea07a899e1df3c6232d2f977ba3; parentv0/c590 giữ nguyên. Người dùng chốt WATER_ALL(level1/2/3) ép tưới cả3 bất kể héo; casualVI/EN và STOPngắn đã QA/convert/round-trip. Configmodel configs/training_v0.yaml dùng data/lerobot/wording_v1/full/train và train stats SHA b06849340aa520815152fd63921b3602ab9278f7c3fd242e600c6a33a086c78f. Tokenizer96 zero truncation5040records, freshLoRA371328params, padding32, three-camera256 CPUforward/inference đãPASS. Browser8765/currentCLI đã đồng bộ. Không có optimizer/GPUfit/resume proof. **Máy hiện tại không NVIDIA; chuyển máy theo agri-soybean-vla/GPU_HANDOFF.md và làm bước7 tiếp.** MODEL_TRAINING.md/status là mô tả thực hiện; các đoạn v0 dưới đây giữ lịch sử. 6000steps/batch1 hiện ≈1.67 lượt3600records; 64token không đủ, phải dùng96 và savedprocessor đã audit.

## 1. Hai đầu ra cần đạt

1. **Dataset v0 tái lập được, song ngữ Việt–Anh:** ảnh nguồn có provenance, nhãn rõ nguồn gốc, split chống rò rỉ, manifest dễ đọc, bản xem mẫu, báo cáo QA và LeRobot v3 đọc ngược đúng.
2. **Gói sẵn sàng train:** môi trường có phiên bản xác định, fresh LoRA trên SmolVLA base, cấu hình ba camera/action ba chiều, smoke test train/save/reload/resume trên máy GPU 4 GB, script chạy dài và đánh giá sau train.

“Sẵn sàng train 4 GB” chỉ được xác nhận sau khi chạy bài đo trên đúng máy train. Code/config chuẩn bị xong trên CPU là một mốc riêng. Không cam kết model hội tụ, đạt một accuracy cụ thể hoặc hoàn tất train trong đêm trước khi đo tốc độ.

Phạm vi đêm đầu là học bắt chước policy tổng hợp từ biểu hiện héo và câu lệnh. Phần cứng ba van, lượng nước được hiệu chuẩn và feedback cảm biến thuộc giai đoạn tiếp theo của đồ án.

## 2. Phát hiện mới phải đưa vào thiết kế

Phần này ghi phát hiện lúc lập plan (trước bước 1): khi đó mới tải **CSV nhỏ**, chưa tải ZIP ảnh. Bước 1–4 hiện đã QA ZIP/ảnh và sinh scene; số liệu hiện tại xem status. CSV có MD5 khớp bản ghi Zenodo:

```text
MD5:    538bbf95130b3bc637acfcbed503de64
SHA256: 5c4eb0dd55c164ebf8c83fc0d211fb44dbf4647824a2bbeac05988aa9ef02e4f
Header: ,treatment_camera,fullID,,Annotation
```

| Annotation thực tế | Số dòng |
|---|---:|
| 0 | 524 |
| 1 | 473 |
| 2 | 369 |
| 3 | 216 |
| 4 | 206 |
| 5 — ngoài miền nhãn v0 | 104 |
| Tổng | 1.892 |

Có 1.892 `fullID` khác nhau và 10 nhóm `treatment_camera`. Toàn bộ nhãn 5 nằm ở nhóm `11-46`. **1.788 dòng nằm trong miền 0–4**; đây chưa phải số ảnh đã kiểm tra đọc được. Số này trùng con số tập con trong handoff, nhưng chưa đủ bằng chứng để kết luận lý do tác giả chọn tập con đó. Nguồn kiểm tra: [CSV gốc](https://zenodo.org/records/8256382/files/dataAnns_goodFiles.csv?download=1), [bản ghi dataset](https://zenodo.org/records/8256382).

Quyết định v0: giữ nguyên raw CSV, ghi 104 dòng vào `quarantine.csv` với lý do `annotation_outside_v0_domain`; không tự đổi 5 thành 4, không xóa ảnh. Nếu xác minh được ý nghĩa nhãn 5 sau này, thay đổi bằng phiên bản dataset mới.

Nhãn 4 chỉ có ở `4-228` (4 ảnh), `10-87` (136), `11-46` (66). Vì vậy không chia ngẫu nhiên 80/10/10 theo ảnh hoặc theo scene đã sinh.

Các vấn đề repo cần xử lý trong module mới: `.gitignore` gốc dùng allowlist; installer đèn tham chiếu `requirements.txt` không tồn tại; generator đèn có đường dẫn cứng và `rmtree`. Không tái sử dụng trực tiếp những hành vi này.

## 3. Hợp đồng dữ liệu và policy v0

### Model nhận gì

- Ba ảnh RGB riêng biệt, thứ tự cố định camera1↔zone1, camera2↔zone2, camera3↔zone3.
- `task`: một câu lệnh **tiếng Việt hoặc tiếng Anh**, UTF-8; mỗi record chỉ có một ngôn ngữ, không ghép hai bản dịch vào cùng task. Không thêm điểm héo hoặc intent vào prompt.
- `observation.state`: float32[3] = `[0,0,0]`, diễn giải **giả định mọi van ban đầu đóng**, không phải đo cảm biến. Chuẩn hóa STATE bằng IDENTITY.
- Target: `action`: float32[3], mỗi chiều thuộc `{0,1,2,3}`. Chuẩn hóa ACTION bằng thống kê **train**, hoàn nguyên trước khi lượng tử hóa lúc inference.

State là cần thiết với đường code SmolVLA hiện tại. Feature action có ba chiều nhưng giữ kích thước padding nội bộ tương thích checkpoint; không tự đổi `max_action_dim`/`max_state_dim` thành 3. [Source SmolVLA đã pin](https://github.com/huggingface/lerobot/blob/7e241bd630a3719a56157a497ce5d08f244784f1/src/lerobot/policies/smolvla/modeling_smolvla.py).

### Nhãn hành động được tạo thế nào

`WILT_TO_LEVEL = {0:0, 1:0, 2:1, 3:2, 4:3}`. Đây là giả định policy của nhóm, không phải công thức lượng nước nông học.

| Intent | Quy tắc |
|---|---|
| AUTO | Áp dụng bảng cho cả ba zone. |
| WORST_ONLY | Nếu max wilt < 2: toàn 0. Nếu ≥ 2: chỉ tưới các zone cùng đạt max; **tất cả zone hòa điểm** được giữ mức. Template phải nói rõ trường hợp hòa điểm. |
| ZONE_ONLY(k) | Chỉ giữ mức proxy của zone k; các zone khác bằng 0. |
| CAP_LEVEL(c) | Mỗi zone lấy min(mức proxy, c), c thuộc {1,2}. CAP_LOW là trường hợp c=1. |
| EXCLUDE_ZONE(k) | Zone k bằng 0; các zone khác theo bảng. |
| STOP_ALL | Luôn `[0,0,0]`. |

Điểm héo 1 vẫn OFF trong policy này: tài liệu/templates phải giải thích ngưỡng hành động từ mức héo 2, tránh ngầm tuyên bố mọi biểu hiện héo đều được tưới. Các câu “nhẹ”, “vừa”, “nặng” không được dùng như mô tả nhãn thật nếu chưa định nghĩa rõ thang điểm. Không dùng “đất khô nhất” để diễn giải ảnh.

V0 sinh từng intent rõ ràng; câu mâu thuẫn, ngoài phạm vi hoặc phối hợp nhiều intent chưa hỗ trợ không được tự gán target. Bài test negative phải từ chối cấu hình intent/parameter không hợp lệ. STOP_ALL trong dataset khác với nút dừng độc lập ở hệ thống thật sau này.

### Manifest dễ đọc

Mỗi record có `schema_version`, `sample_id`, `scene_id`, `scene_type=synthetic_three_zone`, `split`, `images.zone1/2/3`, `source_image_ids`, `source_groups`, `instruction`, `language=vi|en`, `template_id`, `template_family_id`, `translation_pair_id`, `wording_partition=seen|unseen`, `intent`, `intent_params`, `teacher_annotations.wilt_scores`, `target_action`, `label_source=wilt_proxy_policy_v0`, `label_type=synthetic_policy_action`, `policy_version`, `seed`.

`language` là ngôn ngữ câu lệnh; `wording_partition` là cách diễn đạt đã thấy/chưa thấy khi train. Không dùng tên mơ hồ `language_partition` cho hai ý nghĩa này. `template_family_id` liên kết các bản Việt–Anh tương ứng; `translation_pair_id` liên kết hai record cùng scene/intent/params và target. Các field này chỉ dùng metadata/QA/eval, không là model input.

Source manifest chứa đường dẫn tương đối, camera/plot, timestamp gốc và kết quả parse, kích thước ảnh, checksum, nhãn nguyên bản, trạng thái QA, URL nguồn. Không để đường dẫn tuyệt đối của máy cá nhân trở thành dữ liệu bắt buộc.

Ví dụ `wilt=[0,2,4]`: AUTO→`[0,1,3]`, WORST_ONLY→`[0,0,3]`, CAP_LOW→`[0,1,1]`, ZONE_ONLY(2)→`[0,1,0]`, STOP_ALL→`[0,0,0]`. Các điểm héo chỉ ở metadata cho teacher/evaluation.

Ví dụ cặp dịch STOP_ALL: “Dừng tưới tất cả các vùng.” ↔ “Stop watering all zones.” → cùng `[0,0,0]` trên cùng scene. Những intent phụ thuộc ảnh cũng phải có cặp dịch và paraphrase đầy đủ; vài câu STOP tiếng Anh không đủ để gọi dataset song ngữ.

## 4. Split và quy mô khởi đầu

Phương án cố định **đề xuất từ CSV**, khóa trước train; phải kiểm tra lại sau kiểm tra ZIP, duplicate và ảnh gần trùng:

| Split | Nhóm nguồn | Số dòng 0–4 | Phân bố 0 / 1 / 2 / 3 / 4 |
|---|---|---:|---|
| Train | 8 nhóm còn lại, gồm `10-87` | 1.258 | 466 / 343 / 248 / 65 / 136 |
| Val | `4-228` | 189 | 21 / 48 / 49 / 67 / 4 |
| Test | `11-46`, loại riêng nhãn 5 khỏi tập dùng | 341 | 37 / 82 / 72 / 84 / 66 |

Train groups: `3-106`, `5-180`, `5-27`, `7-130`, `10-87`, `14-198`, `15-29`, `16-60`.

- Tỷ lệ ảnh hợp lệ theo CSV khoảng 70.4%/10.6%/19.1%. Camera 5 có hai plot, cả hai cùng train. Xác minh ý nghĩa camera/plot qua nguồn trước khi đóng băng split.
- Val chỉ có bốn ảnh mức 4; báo cáo sự hạn chế này. Ghép lại ảnh không tăng số quan sát độc lập. Một test plot chưa chứng minh tổng quát cho mọi ruộng; group cross-validation là mở rộng sau đêm đầu.
- Mỗi scene có ba **ảnh nguồn khác nhau**, có thể cùng plot nhưng khác thời điểm. Không yêu cầu ba plot khác nhau vì val/test chỉ có một plot; luôn gọi là scene tổng hợp.
- SHA256 trùng phải thuộc cùng một split hoặc được xử lý bằng bản ghi loại trùng; ảnh gần trùng cần nhóm/kiểm tra trước sinh scene. Nếu không thể giữ split dự kiến, cập nhật có lý do trước train, không nới điều kiện chống leakage.
- Mục tiêu pilot: **3.000 mẫu train, 600 val, 600 test**, có config để tăng sau. Train gồm 1.500 cặp tương đương Việt–Anh → 1.500 record mỗi ngôn ngữ, mỗi cặp cùng scene/intent/params/target. Đây là số record tổng hợp, không phải số ảnh hoặc quyết định độc lập; chỉ sinh đủ nếu các ràng buộc dữ liệu cho phép.
- Train đề xuất quota intent: AUTO 30%, WORST_ONLY 20%, ZONE_ONLY 15%, CAP_LEVEL 15%, EXCLUDE_ZONE 10%, STOP_ALL 10%. Cân bằng zone/parameter, đảm bảo mọi mức có cơ hội xuất hiện; công bố phân bố action thực tế thay vì ép tất cả vector đồng đều.
- Val/test dùng panel đánh giá cố định có phủ các intent/edge case. Tách kết quả theo mức và intent; phân bố được thiết kế phục vụ kiểm thử, không gọi là tần suất vận hành thực tế.
- Trong mỗi val/test: **150 trường hợp scene+intent+params × 2 ngôn ngữ × 2 loại wording (seen/unseen) = 600 record**. Cùng một trường hợp có bốn câu lệnh tương đương và cùng target. Mỗi ngôn ngữ có 300 record. Các bản dịch/paraphrase là quan sát phụ thuộc, không tính như bốn scene độc lập.
- Khóa template families đồng thời cho cả Việt–Anh: unseen-val families không xuất hiện trong train; unseen-test families không xuất hiện trong train hoặc val. Các bản dịch tương ứng không được chia family giữa seen và unseen. Đây là đánh giá wording chưa gặp trong **hai ngôn ngữ đều được train**, không phải zero-shot language transfer.
- Kiểm tra quota từng language × intent × zone/cap; không để tiếng Anh chủ yếu STOP còn tiếng Việt chủ yếu AUTO. Kiểm tra tương đương bản dịch về phủ định, số zone, cap, threshold và tie bằng template metadata, unit test và review mẫu.
- Tạo contrastive cases: cùng ảnh khác lệnh; cùng lệnh khác ảnh; hoán vị zone đúng cả tham số zone trong câu lệnh khi cần. Có all-OFF, all-high, tie, STOP và vùng bị loại.

## 5. Tổ chức code dự kiến

```text
agri-soybean-vla/
  README.md
  DATASET_SPEC.md
  DATASET_CARD.md
  TRAINING_4GB.md
  pyproject.toml
  requirements-data.lock.txt
  requirements-train.lock.txt
  configs/
    dataset_v0.yaml
    policy_v0.yaml
    split_v0.yaml
    language_v0.yaml
    train_v0_4gb.json
  src/agri_vla/
    cli.py
    ingest.py
    split.py
    policy.py
    language.py
    scenes.py
    validate.py
    preview.py
    lerobot_io.py
    model_io.py
    train.py
    evaluate.py
  scripts/
    setup.sh
    run_training.sh
  tests/
    test_ingest.py
    test_split.py
    test_policy.py
    test_generation.py
    test_lerobot_roundtrip.py
    test_model_io.py
  .venv/                   # local, ignored
  data/raw/                # local, ignored
  data/manifests/v0/        # generated, ignored
  data/lerobot/v0/          # generated, ignored
  reports/                 # generated, ignored
  outputs/                 # checkpoints/logs, ignored
```

Có thể gộp module nếu code ít; không tạo file rỗng để đủ cây. Tests dùng ảnh fixture tự tạo, không phụ thuộc tải toàn bộ dataset. Khi đổi `.gitignore`, chỉ allowlist source/config/docs/tests và giữ dữ liệu, weights, cache, venv, outputs ngoài Git.

Mỗi bước phải kết thúc bằng: **file thay đổi → giải thích input/output và lý do → lệnh chạy → kết quả thật → artifact xem được → hạn chế còn lại**. Giải thích thuật ngữ lần đầu, kèm ít nhất một ví dụ. Không ghi “PASS” khi chỉ mới viết code.

## 6. Các bước triển khai và yêu cầu có thể giao lần lượt

Tất cả lệnh dưới đây chạy từ root repo, sau khi CLI đã được hiện thực. Dùng virtualenv riêng của module để không làm thay đổi môi trường demo cũ:

```bash
AGRI_PY=agri-soybean-vla/.venv/bin/python
```

### Bước 0 — Khung module, môi trường và spec

**Cần làm:** tạo package/CLI dùng được; spec theo mục 3; environment setup hai chế độ `data` và `train`; preflight; allowlist Git. Mode data không import Torch/LeRobot. Mode train pin LeRobot đúng commit hiện có, cài đủ extras `smolvla,peft,training`, kiểm tra resolver, `pip check`, lưu freeze và revision của base model/tokenizer. Không tự nâng gói trong env demo cũ. [Extras ở phiên bản đã pin](https://github.com/huggingface/lerobot/blob/7e241bd630a3719a56157a497ce5d08f244784f1/pyproject.toml).

**Lệnh dự kiến:**

```bash
git branch --show-current
bash agri-soybean-vla/scripts/setup.sh --mode data
agri-soybean-vla/.venv/bin/python -m agri_vla.cli preflight --mode data
```

**Đạt khi:** đúng nhánh; CLI `--help` dùng được; spec/schema có ví dụ; Python 3.12 trong env; docs/code không bị ignore, raw/weights có bị ignore. Allowlist cả context/plan/prompt/status để chuyển giao qua Git, không tự commit/push. Dependency lock chỉ gọi là đã xác minh khi cài và import thành công.

**Prompt giao việc:** “Implement bước 0 theo plan. Giải thích rõ observation/target/teacher metadata, bốn mức hành động và giả định state. Ghi file, lệnh thiết lập, kết quả kiểm tra môi trường và những điểm còn chưa kiểm chứng.”

### Bước 1 — Nhập dữ liệu và đối chiếu ảnh

**Cần làm:** downloader có checksum, tải tạm rồi rename khi hoàn tất; dùng local path nếu đã có ZIP; không ghi đè/xóa dữ liệu. Parse CSV chấp nhận cột rỗng nhưng kiểm tra các cột bắt buộc. Giữ nhãn gốc, quarantine 5. Kiểm tra ZIP traversal, filename trùng/basename mơ hồ, thiếu/thừa ảnh, ảnh hỏng, RGB, kích thước, hash, timestamp. Không suy ngày/tháng trái mẫu `18_09_2019`.

**Lệnh dự kiến:**

```bash
$AGRI_PY -m agri_vla.cli fetch-source --config agri-soybean-vla/configs/dataset_v0.yaml --dry-run
$AGRI_PY -m agri_vla.cli fetch-source --config agri-soybean-vla/configs/dataset_v0.yaml
$AGRI_PY -m agri_vla.cli inspect-source --config agri-soybean-vla/configs/dataset_v0.yaml
```

**Artifact:** `sources.jsonl`, `quarantine.csv`, `source_report.json`, contact sheet theo nhãn. Lệnh tải ZIP chỉ thực hiện khi triển khai có quyền tải dữ liệu; phiên lập plan chưa tải ZIP.

**Đạt khi:** giải thích đủ 1.892 dòng = dùng được + quarantine + các vấn đề ảnh; không silently drop; counts 1.788 chỉ được nâng thành “ảnh hợp lệ” sau khi đối chiếu file thật. MD5 ZIP kỳ vọng từ Zenodo: `73e83e3abed5680d82d03d46984ad2a6`.

**Prompt:** “Implement bước 1. Giải thích phát hiện nhãn 5, cách bảo toàn dữ liệu gốc, số ảnh thực sự đọc được. Xuất báo cáo và ví dụ lỗi; không đổi nhãn 5 thành 4.”

### Bước 2 — Khóa split theo nguồn

**Cần làm:** thực hiện split mục 4 bằng manifest, seed và cấu hình tường minh; kiểm tra ảnh trùng/gần trùng; không scene synthesis trước khi split nguồn được khóa.

```bash
$AGRI_PY -m agri_vla.cli split --config agri-soybean-vla/configs/split_v0.yaml --dry-run
$AGRI_PY -m agri_vla.cli split --config agri-soybean-vla/configs/split_v0.yaml
```

**Artifact:** split manifest, `split_report.json`, bảng lớp theo group, hash của quyết định split.

**Đạt khi:** giao source IDs/hash giữa splits rỗng; nhóm nguồn không giao nhau; số lượng đối chiếu ingest; thiếu lớp hoặc thay đổi split phải được báo rõ. Tuyệt đối không chuyển ảnh test vào train để làm đẹp metric.

**Prompt:** “Implement bước 2. Giải thích tại sao split theo nguồn trước khi ghép cảnh, vì sao val chỉ có bốn ảnh mức 4 và hệ quả đánh giá. In bảng phân bố và kết quả test chống leakage.”

### Bước 3 — Policy labeler và câu lệnh song ngữ Việt–Anh

**Cần làm:** policy thuần deterministic; config version/hash; template registry **Việt–Anh** theo intent/params/language/template family/wording partition. Sinh các cặp lệnh tương đương, cùng target từ policy, mỗi record một ngôn ngữ. Mặc định không dùng LLM tự gán action. Các mẫu câu ở cả hai ngôn ngữ phải phản ánh đúng threshold/tie/cap/exclusion.

```bash
$AGRI_PY -m pytest agri-soybean-vla/tests/test_policy.py -q
$AGRI_PY -m agri_vla.cli explain-policy --wilt 0 2 4
```

**Đạt khi:** kiểm tra toàn bộ 125 tổ hợp wilt với intent/params; kiểm tra STOP, hòa điểm, cap, zone range, monotonic AUTO và permutation; ví dụ kỳ vọng độc lập với implementation. Câu mâu thuẫn/cấu hình ngoài miền bị từ chối.

**Prompt:** “Implement bước 3 song ngữ Việt–Anh. Giải thích từng intent bằng cặp câu Việt/Anh→cùng action, đặc biệt wilt 1 vẫn OFF và WORST_ONLY hòa điểm. Kiểm tra tương đương bản dịch, language balance và held-out template families, kèm test results.”

### Bước 4 — Sinh scene, QA và bản xem dataset

**Input đã khóa:** đọc ba `*.sources.jsonl` từ `data/manifests/source_splits/v0/` và kiểm tra từng SHA256 trong `split_lock.json` trước compose. Không đọc `excluded.sources.jsonl` vào dữ liệu v0, không tự chia lại source manifest gốc hoặc chuyển source giữa split.

Đọc thêm configs/policy_v0.yaml, configs/language_v0.yaml và reports/policy/v0/policy_lock.json; xác minh hash config/version. Dùng policy.action và registry.render_pair, không tự viết lại labeler. Phân phối family có seed tái lập/coverage, không gọi choose_pair với cùng seed cho mọi sample. Val/test seen phải dùng family đã thực sự xuất hiện trong train; unseen-val/test không giao nhau hoặc train. Tạo translation_pair_id theo scene/intent/params/family và kiểm hai record cùng target.

**Cần làm:** generator seed cố định, ba source IDs khác nhau, cân bằng zone và reuse có thống kê; giữ quan hệ contrastive pairs. JSONL schema strict. Đường dẫn resolve từ dataset root. Preview HTML tĩnh mở local, không cần server/web framework; hiển thị ba ảnh, lệnh, điểm héo teacher, target và provenance. Teacher metadata không được in trực tiếp lên ảnh dùng train.

```bash
$AGRI_PY -m agri_vla.cli compose --config agri-soybean-vla/configs/dataset_v0.yaml --dry-run
$AGRI_PY -m agri_vla.cli compose --config agri-soybean-vla/configs/dataset_v0.yaml
$AGRI_PY -m agri_vla.cli validate --manifests agri-soybean-vla/data/manifests/v0
$AGRI_PY -m agri_vla.cli preview --manifests agri-soybean-vla/data/manifests/v0 --count 60
```

**Đạt khi:** mọi record đúng schema; label tái tính đúng; không cross-split leakage; cùng seed/config/source cho cùng manifest semantic hash; không annotation 5 vào model data; không ảnh nguồn bị chọn lặp trong một scene. Báo số ảnh độc lập, scene, record, tỷ lệ all-zero, reuse, mức hành động, intents và language × wording × template coverage. Kiểm tra mỗi cặp dịch cùng source/intent/params/target, hai ngôn ngữ cân bằng và family isolation. Xem ít nhất 60 mẫu phủ mọi intent/edge case ở cả Việt và Anh.

**Prompt:** “Implement bước 4. Tạo dataset có thể xem và giải thích từng field. Xuất preview/report, giải thích sự khác nhau giữa 3.000 mẫu tổng hợp và 1.258 ảnh train nguồn. Chỉ báo hoàn thành khi QA thật đã chạy.”

### Bước 5 — Chuyển LeRobot và đọc ngược

**Bổ sung đã hoàn thành trước bước 5 — bước 4.5:** trang HTML localhost duyệt toàn bộ 4.200 record và 1.892 ảnh nguồn, với filters/pagination/detail/translation/ảnh gốc/thư mục local. Chạy `bash agri-soybean-vla/scripts/browse.sh`, mở http://127.0.0.1:8765/. Hướng dẫn [BROWSE_DATASET.md](agri-soybean-vla/BROWSE_DATASET.md); 343 tests PASS và Chromium desktop/mobile đã kiểm. Đây là read-only browser, không thay dataset lock và không phải conversion/train.

**Input đã có:** `data/manifests/v0/train.jsonl`, val.jsonl, test.jsonl và dataset_lock.json. Chạy validator hiện tại trước conversion; đối chiếu byte/semantic hashes, config/schema/model contract/implementation hashes và hai upstream locks. Giữ split/pair/wording metadata để đánh giá; tuyệt đối không đưa case/provenance/generation_metadata/teacher/structured intent vào features. Counts 3.000/600/600 là record, không phải ảnh độc lập. Visual review và DATASET_CARD không thay thế round-trip.

Conversion và round-trip chạy trên CPU dù đã cài dependencies train. Preflight ở bước này báo GPU availability; chỉ bước 7 bắt buộc CUDA để đạt tiêu chí GPU. Chọn 64 mẫu smoke phủ hai ngôn ngữ, intents và mức action từng chiều, không lấy tùy tiện 64 dòng đầu có thể chỉ toàn OFF.

**Cần làm:** môi trường train; tạo bản nhỏ 64 sample train trước, rồi full train/val/test ở ba root riêng. Một scene+instruction là một episode một frame, `fps=1` chỉ là quy ước lưu trữ. Ba image features, `use_videos=false`, state[3], action[3], task. Lưu mapping episode/sample ID. Gọi `finalize()` và đọc lại. [API dataset ở commit đã pin](https://github.com/huggingface/lerobot/blob/7e241bd630a3719a56157a497ce5d08f244784f1/src/lerobot/datasets/lerobot_dataset.py).

```bash
bash agri-soybean-vla/scripts/setup.sh --mode train
AGRI_TRAIN_PY=agri-soybean-vla/.venv-train/bin/python
$AGRI_TRAIN_PY -m agri_vla.cli preflight --mode train
$AGRI_TRAIN_PY -m agri_vla.cli convert --config agri-soybean-vla/configs/lerobot_v0.yaml --profile smoke
$AGRI_TRAIN_PY -m agri_vla.cli verify-lerobot --root agri-soybean-vla/data/lerobot/v0/smoke
$AGRI_TRAIN_PY -m agri_vla.cli convert --config agri-soybean-vla/configs/lerobot_v0.yaml --profile full
```

**Đạt khi:** đọc ngược toàn bộ record đối chiếu instruction, action, state, camera order và pixel với phép resize/pad đã định nghĩa; stats finite. PNG/Parquet có thể nhúng ảnh, không bắt buộc thư mục images tồn tại. Với `chunk_size=1`, kiểm tra action batch `[B,1,3]`, `action_is_pad` không che mất frame, sampler không drop toàn bộ singleton episodes. Train-only stats được lưu cho inference/eval, không thay bằng stats val/test.

**Prompt:** “Implement bước 5. Giải thích JSONL→LeRobot, vì sao dùng episode một frame và fps=1. In shape batch thật, stats, số episode/frame, kết quả round-trip và nơi lưu ba split.”

### Bước 6 — Fine-tune và checkpoint cho schema nông nghiệp

**Input bàn giao bước 5:** dùng .venv-train, full/train là train duy nhất, val/test roots riêng; không dùng tên container nội bộ info.splits.train để suy split nguồn. Nạp full/train_stats.json và normalization_contract.json, không dùng quantiles aggregate singleton hoặc stats val/test. Factory thêm time axis cho observations: camera [B,1,3,256,256], state [B,1,3], action [B,1,3]; cần nối processor/model đúng axes và không tự resize lại 512. Converterconfig riêng lerobot_v0.yaml giữ input dataset config/schema/model-contract hashes bất biến. GPU smoke/model weights chưa thực hiện.

**Cần làm:** dùng `lerobot/smolvla_base` với revision được ghi cố định, fresh LoRA; không nạp `light_lora_1000`. Wrapper mỏng gọi trainer upstream, không viết lại toàn bộ loop. Khôi phục đúng feature schema ngay lúc tạo policy. Cấu hình đề xuất ở mục 7; lưu resolved config đầy đủ.

Đặt `dataset.eval_split=0` và train chỉ đọc train root. Val/test dùng evaluator riêng. Lý do: factory upstream tách episode theo task, không biết camera/plot nguồn. [Source split của trainer](https://github.com/huggingface/lerobot/blob/7e241bd630a3719a56157a497ce5d08f244784f1/src/lerobot/datasets/factory.py).

Tokenizer phải đo toàn bộ prompt **Việt và Anh** sau tiền xử lý; báo length distribution/truncation theo ngôn ngữ, không cắt mất từ phủ định hoặc zone ở cuối. Nếu đổi max length, kiểm tra cả **processor thực sự được tạo/lưu**; sửa policy config riêng chưa đủ. Model loader dùng saved schema, tokenizer/preprocessor và train normalization stats, không phụ thuộc dataset đèn.

```bash
$AGRI_PY -m agri_vla.cli prepare-training --profile 4gb --dry-run
$AGRI_PY -m agri_vla.cli check-model-inputs --profile 4gb
```

**Đạt khi:** đúng ba camera/state/action; token truncation bằng 0 cho dataset; chỉ các adapter/projection được chủ định mới trainable; log tên và số tham số trainable; forward nhận input đúng dtype/shape; checkpoint đóng gói đủ adapter, config, processor/stats và base revision.

**Prompt:** “Implement bước 6. Giải thích cái gì được freeze, cái gì được LoRA học, vì sao không dùng adapter đèn. In resolved config, trainable parameters, tokenizer coverage và những thay đổi so với demo cũ.”

### Bước 7 — Kiểm chứng train trên đúng GPU 4 GB

**Cần làm:** chạy 50–100 optimizer steps ở batch 1 với cấu hình và prompt dài nhất thật; full forward/backward/update; đo torch allocated/reserved và tổng GPU từ nvidia-smi. Kiểm tra loss/gradient/weights finite; adapter thay đổi; checkpoint save→fresh-process reload→inference→resume training. Ghi kết quả JSON/Markdown.

Smoke dùng subset train nhưng giữ schema, processor và thống kê normalization của full train sẽ dùng ở run dài. Nếu kết quả kiểm tra đòi đổi chúng, cập nhật phiên bản/config rồi rerun smoke; không dùng checkpoint smoke có stats khác để mặc nhiên xác nhận pipeline full train.

```bash
$AGRI_PY -m agri_vla.cli smoke-train --profile 4gb --steps 100 --check-resume
```

**Đạt khi:** không OOM, không NaN/Inf, có update thật, load/resume đúng schema và optimizer step, dự đoán shape `[3]`; reset policy queue cho từng sample độc lập. Cùng seed/noise cho reload comparison trong tolerance số học đã công bố. Báo VRAM peak/ảnh thực đi vào encoder/token count/thời gian mỗi step. Headroom mục tiêu khoảng 10–15% tổng VRAM sau tính các process khác; không coi số này là bảo đảm trên mọi máy.

Sau smoke kỹ thuật, thử overfit 32–64 mẫu train có lệnh đối lập và nhãn đa dạng trong 200–300 steps. Đây là bài chẩn đoán: theo dõi loss và predictions, không dùng accuracy train làm kết quả đồ án. Nếu không học được khác biệt cơ bản, debug data/processor/LoRA trước run dài.

**Prompt:** “Implement và chạy bước 7 trên máy train. Giải thích forward/backward/optimizer/checkpoint, báo peak VRAM thật và ETA. Nếu thiếu GPU, đánh dấu BLOCKED cho kiểm chứng GPU; không gọi cấu hình là đã chạy được 4 GB.”

### Bước 8 — Runbook chạy tối nay

**Cần làm:** run script kiểm tra QA/smoke/config hash, output mới, log đầy đủ, save định kỳ, resume không ghi đè run khác. Số bước chốt bằng tốc độ đo thực tế và thời gian người dùng có.

```bash
bash agri-soybean-vla/scripts/run_training.sh --profile 4gb --steps 6000 --dry-run
bash agri-soybean-vla/scripts/run_training.sh --profile 4gb --steps 6000
```

Profile đầu đề xuất 6.000 optimizer steps, batch 1, tương đương khoảng hai lượt qua 3.000 record nếu sampler không thay đổi. 1.000 steps chỉ đi qua khoảng một phần ba epoch ở batch 1; không gọi đó là đã học hết dataset.

`ETA = số steps còn lại × seconds/step đo sau warm-up + thời gian save/eval`. Ví dụ minh họa 2 s/step × 6.000 ≈ 3 giờ 20 phút, **không phải tốc độ đã đo**. Nếu không đủ thời gian, giảm số bước run đầu và ghi rõ coverage; giữ checkpoint để resume, không âm thầm giảm độ nghiêm ngặt của split/evaluation.

**Đạt khi:** `TRAINING_4GB.md` có lệnh install/data/smoke/train/resume/eval, cách tìm checkpoint/log, xử lý OOM và lỗi thường gặp. Phân biệt bắt đầu run mới và resume; chỉ gọi resume khi đã lưu đủ optimizer/scheduler/RNG/step state. Save checkpoint mỗi 500 steps, giữ điểm kiểm tra được lựa chọn rõ ràng; không tự xóa checkpoint có sẵn.

**Prompt:** “Implement bước 8. Viết runbook từng lệnh và giải thích tham số. Từ smoke report tính ETA thật, ghi số epoch tương đương và bàn giao lệnh để tôi tự bắt đầu run dài tối nay.”

### Bước 9 — Đánh giá và demo inference sau train

**Cần làm:** chọn checkpoint bằng validation; test chỉ dùng sau khi chốt lựa chọn. Ban đầu đánh giá một số checkpoint cố định (ví dụ 1.000/3.000/6.000) trên dev subset val đã khóa, rồi đánh giá đầy đủ val cho ứng viên; không cho hai process model cùng chiếm GPU 4 GB. Dùng cùng processor/stats từ train.

```bash
$AGRI_PY -m agri_vla.cli evaluate --checkpoint <checkpoint_path> --split val
$AGRI_PY -m agri_vla.cli evaluate --checkpoint <selected_checkpoint_path> --split test --ablations
$AGRI_PY -m agri_vla.cli infer --checkpoint <selected_checkpoint_path> --images <zone1.jpg> <zone2.jpg> <zone3.jpg> --instruction 'Tưới các vùng có biểu hiện héo theo mức quy định.'
```

**Metrics:** exact vector accuracy, per-zone accuracy, ordinal MAE, confusion matrix, per-intent/per-level, STOP_ALL correctness, WORST_ONLY/tie correctness, zone permutation consistency. Tách kết quả **vi/en × seen/unseen wording**; thêm tỷ lệ hai bản dịch cho cùng vector dự đoán và tỷ lệ cả hai cùng đúng. Agreement không đồng nghĩa accuracy: hai bản dịch có thể cùng sai. Dùng cùng noise seed khi so cặp để hạn chế nhiễu sampling. Báo số source độc lập và lỗi mẫu bằng preview.

**Baselines tối thiểu:** luôn OFF/majority từ train; reference-policy oracle để kiểm tra generator; inference với ảnh blank/ảnh tráo và với lệnh bỏ trống/đổi lệnh. Ablation là kiểm tra mức phụ thuộc input, không tự nó chứng minh hiểu ngữ nghĩa. Baseline classifier dự đoán wilt rồi áp dụng policy là mốc nghiên cứu tiếp theo nếu chưa kịp đêm đầu; không tuyên bố VLA vượt classifier khi chưa so sánh.

**Postprocess:** continuous action → inverse normalization → kiểm tra finite → clamp [0,3] → lượng tử hóa với tie rule công bố, ví dụ half-up `floor(x+0.5)` → output JSON. Báo cả continuous/unclipped values và tỷ lệ ngoài miền; không che lỗi bằng clip. Không bịa confidence. Lệnh tự do ngoài tập phân bố vẫn chưa có bảo đảm xử lý đúng.

**Đạt khi:** lưu prediction từng sample, model/config/dataset hashes, metrics và failure cases; không gọi model đạt yêu cầu nhận biết Vision chỉ dựa loss giảm. Output v0 là đề xuất action tổng hợp; chưa gửi van thật.

**Prompt:** “Implement bước 9. Giải thích từng metric bằng ví dụ, so baseline và ablation, hiển thị trường hợp sai. Phân biệt đúng policy tổng hợp với hiệu quả tưới thực tế.”

## 7. Profile fine-tune 4 GB đề xuất và thứ tự xử lý OOM

| Tham số | Khởi đầu đề xuất | Điều cần kiểm chứng |
|---|---|---|
| Base | SmolVLA base, revision khóa lúc chuẩn bị | Không khởi tạo expert ngẫu nhiên khi định fine-tune |
| Adapter | Fresh LoRA r=8, alpha=8 | Target modules và số tham số trainable |
| Vision/VLM | Freeze vision, train expert bằng LoRA | Không vô tình mở full VLM training |
| Camera | Ba ảnh RGB, giữ tỷ lệ, resize/pad 256×256 | Encoder/connector chấp nhận độ phân giải; xem chất lượng lá sau resize |
| Model resize | Explicit `(256,256)` thử trước | Kiểm tra tensor thật, không để mặc định tăng lại 512 |
| State/action | 3 / 3 feature dimensions | Giữ padding nội bộ tương thích pretrained weights |
| Horizon | `chunk_size=1`, `n_action_steps=1` | Cả forward/backward/sampler/inference |
| Batch | 1 | Đây là microbatch thực; không hứa effective batch lớn hơn |
| Precision | AMP, lựa chọn dtype theo GPU và path được hỗ trợ | AMP không có nghĩa toàn bộ weights lưu FP16 |
| LR | 3e-4 thử đầu; warmup 100; decay tới 3e-5 ở step 6000 | Đây là lựa chọn thử nghiệm, điều chỉnh từ smoke/pilot/val |
| Workers | 0 ban đầu | Tránh multiprocessing tăng RAM hoặc lỗi WSL |
| Compile | false | Đơn giản hóa smoke và memory profile |
| Augmentation | Tắt trước | Không làm mất dấu hiệu màu/tán lá khi chưa kiểm chứng |
| Token length | Đề xuất 64, tăng nếu audit Việt–Anh cần | Không truncation ở cả hai ngôn ngữ; processor thực khớp config |
| Denoising | 10 inference steps ban đầu | Giảm `num_steps` không thay thế tối ưu backward memory |
| Eval | batch 1, process riêng sau khi giải phóng train process | Dùng saved train stats và held-out nguồn |

Source config cho phép n_action_steps≤chunk_size và action timestamps bắt đầu tại 0, nên horizon một bước có cơ sở ở mức code; chưa thay thế runtime test. Resize mặc định là 512. [Config SmolVLA đã pin](https://github.com/huggingface/lerobot/blob/7e241bd630a3719a56157a497ce5d08f244784f1/src/lerobot/policies/smolvla/configuration_smolvla.py). LoRA có hỗ trợ chính thức và target expert/projections, nhưng mức r/LR ở bảng là cấu hình thử nghiệm của dự án. [Hướng dẫn PEFT v0.6.1](https://huggingface.co/docs/lerobot/v0.6.1/en/peft_training).

Nếu OOM:

1. Dùng nvidia-smi xác định process khác; đóng ứng dụng GPU không cần thiết với sự chủ động của người dùng. Không tự kill process lạ.
2. Kiểm tra batch 1, horizon 1, adapter-only, AMP thực bật và kích thước ảnh thực. Không giải quyết nhầm bằng giảm số record dataset.
3. Thử kích thước thấp hơn được encoder/connector hỗ trợ, ghi rõ quality tradeoff; rerun round-trip/input/smoke và cập nhật config. Không âm thầm bỏ một camera.
4. Nếu weights/activations vẫn vượt giới hạn, xem xét frozen-backbone dtype hoặc checkpointing/offload qua API đã kiểm chứng. Đây là nhánh kỹ thuật riêng, cần gradient/reload test; không thêm flag không tồn tại vào CLI.
5. QLoRA/4-bit không là mặc định đêm đầu; chỉ thử khi xác nhận kiến trúc/PEFT/version hỗ trợ và so sánh reload/prediction. Nếu không có cấu hình pass trên 4 GB, báo rõ cần môi trường GPU lớn hơn để train; dataset vẫn hoàn thành độc lập.

Trainer đã pin chưa cho thấy flow gradient accumulation trong entrypoint được đọc; không thêm `--gradient_accumulation_steps` theo suy đoán. Chỉ triển khai accumulation trong wrapper nếu cần và có kiểm tra scaling loss, optimizer/scheduler step và checkpoint state. [Training entrypoint](https://github.com/huggingface/lerobot/blob/7e241bd630a3719a56157a497ce5d08f244784f1/src/lerobot/scripts/lerobot_train.py).

## 8. Thứ tự ưu tiên và tiêu chí bàn giao

- **Cần trước run dài:** bước 0–8; tập nhỏ 64 mẫu cho bước 5–7 được ưu tiên kiểm tra sớm trước khi tốn thời gian fine-tune. Viết evaluator cơ bản trong bước 6–8 để checkpoint đầu đã có cách kiểm tra; chạy đầy đủ bước 9 sau train.
- **Có thể rút gọn hình thức:** preview HTML đơn giản, ít biểu đồ, không Web UI mới, không video, không tích hợp van trong v0.
- **Không rút gọn:** quarantine nhãn 5; provenance; group split; chống label leakage; train-only normalization; checkpoint reload/resume; GPU smoke; báo cáo đúng kết quả.
- **Đầu ra dataset:** spec/card, raw report, nguồn và split manifest, train/val/test JSONL, preview, QA report, LeRobot roots và mapping.
- **Đầu ra train:** dependency/model revisions, resolved config, smoke report thật, VRAM/ETA, launcher và runbook, checkpoint/evaluation/inference contract.
- **Đầu ra đồ án về sau:** sensor/flow calibration, action duration/volume có cơ sở, controller ba van với fail-safe/STOP, dữ liệu thời gian thật và thử tưới có đo kết quả. Các mục này không được đánh dấu hoàn thành bởi dataset tổng hợp.

## 9. Cách giao việc từng lượt

Copy đúng Prompt N trong `AGRIVLA_IMPLEMENTATION_PROMPTS_2026-11-05.md`. Model chỉ thực hiện bước N, đọc artifact/status bước trước, ghi kết quả thật và dừng. Người dùng kiểm tra bàn giao rồi gửi prompt kế tiếp. Nếu một tiêu chí chưa đạt, sửa hoặc giải quyết blocker của bước đó trước khi gọi bước phụ thuộc.
