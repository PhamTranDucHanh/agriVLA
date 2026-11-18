# Chẩn đoán học trên tập nhỏ — 2026-11-12 đến 2026-11-15

**Cập nhật16/11:** xem [báo cáo hiện hành](OPTIMIZATION_2026-11-16.md). TRAIN60 cao không đủ: VAL112 còn kém OFF; chưa full-ready. Runner mới yêu cầu thêm `--validation-evidence` hợp lệ trước execute. Các kết quả/lệnh15/11 bên dưới là lịch sử.

**Lịch sử tiến trình ngày15:** candidate r8/alpha8, batch4, LR3e-4, native Euler2bước đã đạt40/40 predictions ở micro8 (5noise seeds); đang train panel60 đủ7intents. Chưa gọi config đạt toàn bộ trước khi panel60 pass và reload/resource checks tương ứng hoàn tất. Full train3600 và val/test mới chưa chạy.

**Hiện hành: tạm dừng theo yêu cầu người dùng, web model300 đang chạy.** Không chạy thêm training; fresh-process full60/five-seed evaluation đã dừng theo yêu cầu và chưa hoàn tất. Screen cuối80%/98,33%, không đạt100%. Web8presets gồm7case VI đúng và1case EN còn sai đã kiểm API. Hướng dẫn: DEPLOY_IMPROVED.md; báo cáo web-improved-check.json.

## Phạm vi và tiêu chí

Thí nghiệm này kiểm tra khả năng học các quy tắc trên **train**, trước khi cân nhắc train toàn bộ. Nó không đo test accuracy, không chứng minh tổng quát hóa sang ảnh/cách nói mới, và không thay checkpoint step6000 đang được giữ nguyên.

Giữ dataset WORDING_V1, ba camera, tokenizer96, state IDENTITY, action MEAN_STD và stats của toàn bộ train3600. Không tính stats lại theo tập nhỏ. Không đưa intent, mức héo, teacher annotations hoặc target vào input inference. Không sửa nhãn hoặc hardcode STOP/WATER_ALL.

Đã kiểm lại target của toàn bộ3600 train records bằng số học độc lập: mức tưới cơ sở `max(0, wilt-1)` với wilt0..4, rồi áp điều kiện chọn/loại/cap/tie/STOP hoặc force-all. Kết quả0 target mismatch; receipt `reports/small-cases/train-label-arithmetic-audit.json`. Phép kiểm xác nhận logic synthetic labels, không xác minh annotation mức héo bằng chuyên gia ngoài bộ dữ liệu.

| Panel | Phép kiểm | Số records |
|---|---|---:|
| micro8 | Cùng ba ảnh, STOP và bắt buộc tưới mức1/2/3; mỗi câu có VI/EN. Đáp án000/111/222/333. | 8 |
| cases32 | Thêm11 lựa chọn lệnh/tham số trên cùng scene: AUTO, WORST, ZONE1/2/3, CAP1/2, EXCLUDE1/2/3, STOP; thêm đồng hạng WORST. | 32 |
| cases60 | Thêm11 controls hoán vị vị trí ảnh, WORST dưới ngưỡng, AUTO tất cả khỏe và tất cả héo nặng. Đủ7 intents, VI30/EN30,15 vector target. | 60 |

Các panel lồng nhau và chọn cố định từ full/train. Danh sách đầy đủ gồm câu, ảnh nguồn, target và episode ID ở `reports/small-cases/panels/{micro8,cases32,cases60}.{json,csv}`. Teacher labels trong CSV chỉ giúp con người kiểm kết quả, không phải input model.

Panel60 bao phủ các nhánh quy tắc được liệt kê, không bao phủ toàn bộ125 tổ hợp mức héo, mọi ảnh hoặc mọi wording. Năm seed ở đây là seed **inference noise**, không phải năm lần train độc lập.

Ngưỡng đã chốt trong `configs/small_cases_v1.yaml`: micro8 phải đúng100% vector và từng zone; cases32/60 cần exact>=95%, zone>=98%; STOP phải100% ở mọi panel. Screen dùng seed0/42, batch4; chỉ khi screen đạt mới chứng nhận batch1 ở cả5seeds0/1/2/3/42. **Mọi seed phải đạt**, không lấy trung bình hoặc chọn seed đẹp. Continuous MAE và output ngoài miền được lưu riêng, không phải một ngưỡng MAE đã được chứng nhận.

Trước khi bắt đầu bất kỳ training cases32/60 nào, đã tạo riêng **`configs/small_cases_v3.yaml`** với ngưỡng100% exact/zone/STOP ở mọi panel và mọi seed, để việc chọn cấu hình đáp ứng yêu cầu đúng toàn bộ bộ nhỏ. V1/v2 và raw receipts không sửa. Đây là siết điều kiện cho các run lớn hơn chưa chạy, không phải hạ ngưỡng để hợp thức hóa kết quả cũ.

## Những thay đổi phục vụ thí nghiệm

- `src/agri_vla/small_cases.py`: chọn panel train cố định, kiểm cặp dịch, coverage và gate từng seed.
- `src/agri_vla/small_train.py`: gọi nguyên loop upstream đã pin; đo loss, gradient, thay đổi trọng số, VRAM và dự đoán sau sampling. Lưu checkpoint bằng native API kèm optimizer/scheduler/RNG/step, processor/tokenizer và contract gốc. Loader riêng hỗ trợ các cấu hình thử nghiệm; loader deploy r8 cũ giữ nguyên.
- `configs/small_cases_v1.yaml`: LoRA r8/alpha8 LR3e-4 làm đối chứng; các candidate LR1e-4, r16/alpha16, hoặc train đầy đủ bốn action/time projections nhỏ. VLM/vision vẫn frozen. Candidate được khai báo không đồng nghĩa đã chạy hoặc đã đạt.
- Schedule thí nghiệm: warmup20 rồi constant LR, tối đa1000optimizer steps mỗi run, đánh giá tại100/300/600/1000 và dừng sớm nếu đạt. Schedule cũ cosine6000/warmup100 không bị sửa. Đây là khác biệt giữa diagnostic mới và run cũ; không quy mọi thay đổi chất lượng riêng cho một yếu tố khi chưa có đối chứng tương ứng.
- Nếu dùng `--init-checkpoint`, đó là **weights-only curriculum**, optimizer mới. Không gọi thao tác này là resume đầy đủ. Native training_state được lưu không tự chứng minh đã kiểm thử resume.
- `small_resume.py`: phép kiểm một optimizer update bằng native resume API, đối chiếu với checkpoint của nhánh chạy liên tục. Các run mới lưu checkpoint99 và100 thật, bao gồm thêm AMP GradScaler state mà native save API chưa lưu. Công cụ ghi riêng kết quả restore state và so sánh sample/RNG/weights sau update; chỉ báo đạt sau khi chạy thật.
- `small_memory.py`: lấy các train records có prompt dài nhất, chạy forward/backward với batch của profile, saved processor/stats và native optimizer state đã nạp. Không optimizer step; kiểm trọng số không đổi, actual encoder shapes/token counts và VRAM. Đây là kiểm tài nguyên, không phải điểm chất lượng trên các samples đó.

## Cách tái lập

Từ root repo, trong terminal có GPU access:

```bash
agri-soybean-vla/.venv/bin/python -m pytest agri-soybean-vla/tests/test_small_cases.py -q

agri-soybean-vla/.venv-train/bin/python -u -m agri_vla.small_train \
  --stage micro8 --profile r8_lr3e4 \
  --output agri-soybean-vla/outputs/small-cases/micro8-r8-lr3e4-reproduction
```

Mỗi lệnh cần output mới; runner từ chối ghi đè. Để đánh giá checkpoint trong process mới:

```bash
agri-soybean-vla/.venv-train/bin/python -u -m agri_vla.small_train \
  --stage micro8 \
  --evaluate-checkpoint agri-soybean-vla/outputs/small-cases/RUN/checkpoints/STEP/pretrained_model \
  --reference-evaluation agri-soybean-vla/reports/small-cases/RUN/evaluation-STEP.json \
  --output agri-soybean-vla/reports/small-cases/reload-RUN.json
```

So reload với đúng batch size, seed và noise của report gốc: continuous phải thỏa `abs(new-old) <= 0.001 + 0.001*abs(old)` ở từng thành phần và quantized vector phải trùng hoàn toàn. Đây là tolerance kiểm round-trip FP16 đã chọn trước khi chạy, không phải khoảng sai cho target. Certificate chất lượng vẫn dùng batch1 và năm seeds như trên.

Trọng số học được nằm trong `outputs/small-cases/RUN/checkpoints/STEP/pretrained_model/adapter_model.safetensors`. Với profile full projections, file này còn chứa các module nhỏ được train đầy đủ, vì vậy phải dùng loader thí nghiệm tương ứng. Model tập nhỏ không thay thế model đã train toàn bộ.

## Kết quả

Mỗi run có `experiment.json`, `selection.json`, `resolved_config.json`, snapshot runner, `parameters.json`, `updates.jsonl`, `evaluation-STEP.json` và `result.json` khi kết thúc, trong `reports/small-cases/RUN/`.

| Run / panel / bước | Exact seed0 | Exact seed42 | Zone seed0 /42 | STOP seed0 /42 | Kết luận |
|---|---:|---:|---:|---:|---|
| Always OFF / micro8 | 25% | 25% | 25% /25% | 100% /100% | Baseline, không học |
| r8_lr3e4 / micro8 /1000 | 75% | 87,5% | 75% /95,83% | 100% /100% | Không đạt micro8; chưa chạy certificate vì screen không đạt |
| r8_heads_lr3e4 / micro8 /1000 | 25% | 25% | 33,33% /37,5% | 100% /0% | Không đạt; kém r8 thường, không chọn |
| r8_b2_lr3e4 / micro8 /1000, Euler10 | 75% | 100% | 83,33% /100% | 100% /100% | Không đạt;5seeds có37/40 đúng |
| r8_b4_lr3e4 / micro8 /1000, Euler10 | 87,5% | 100% | 95,83% /100% | 100% /100% | Không đạt;5seeds có39/40 đúng |
| Cùng weights batch4, Euler2 | 100% | 100% | 100% /100% | 100% /100% | Micro8:40/40 đúng; chưa suy ra panel60/full train |

Run đối chứng `micro8-r8-lr3e4-v1`: 1000updates finite, không AMP skip; gradient khác0 ở expert và action/time projections, state projection gradient0 do input state luôn0. Trọng số adapter thay đổi. Tại seed0, hai câu VI/EN mức3 còn ra[2,2,2]; seed42, câu EN mức1 ra[1,2,1]. Đây là lỗi prediction thực, không thể nghiệm thu từ loss.

Update trung bình1,117s, median1,104s trên run này. Torch max allocated1.586GB, reserved1.606GB; hai số đầu tiên lấy từ counter tích lũy của phiên bản runner đầu, có thể bao gồm evaluation trước đó. NVML peak toàn GPU3067/4096MiB, headroom quan sát1029MiB. Các run sau reset torch peak trước mỗi update và ghi riêng peak evaluation; snapshot runner ghi chính xác cách đo. Headroom phụ thuộc cả các allocation khác trên máy và không phải cam kết cho mọi workload.

Saved-state audit ở checkpoint600:74 optimizer parameter steps đều600, scheduler last_epoch600, toàn bộ optimizer tensors finite, có NumPy/Python/CPU/CUDA RNG. Đây mới là kiểm file state đã lưu, chưa phải test resume.

Run `micro8-r8-heads-lr3e4-v1` đã hoàn thành1000updates,1933968trainable parameters, VLM/vision frozen. Các projection được update thật nhưng exact chỉ25% ở cả hai seeds. Ở bước1000, gradient sau clip tập trung nhiều vào action_out projection; đây là quan sát diagnostic, chưa chứng minh cơ chế tối ưu duy nhất. Việc unfreeze thêm layers với cùng LR3e-4 **không cải thiện** trong phép thử này. Không chọn nó chỉ vì có nhiều tham số học hơn.

`configs/small_cases_v2.yaml` khai báo thêm batch thực2/4 để có thể đối chứng ảnh hưởng của số mẫu/noise mỗi update. V1 giữ nguyên hash. Đây là cấu hình candidate; chỉ các run có raw reports mới được xem là đã chạy/đo fit, không tự suy rằng batch lớn fit GPU4GB.

**Fresh-process reload hai run batch1:** cùng16 predictions tham chiếu (8records ×2seeds), batch size4 như lúc train-time screen: continuous max error0 và quantized trùng hoàn toàn ở cả hai run. R8 thường khi đánh giá batch1 trên5seeds có exact75%/100%/100%/100%/87,5% tương ứng0/1/2/3/42, nên vẫn không đạt. Reports `reload-micro8-r8-lr3e4-v1.json`, `reload-micro8-r8-heads-lr3e4-v1.json`.

**Baseline model6000 trên cùng panel**, batch1, seeds0/1/2/3/42:

| Panel | Exact theo năm seed | Always OFF exact |
|---|---|---:|
| micro8 | 25% /0% /50% /25% /0% | 25% |
| cases32 | 18,75% /0% /18,75% /15,625% /0% | 18,75% |
| cases60 | 23,33% /0% /18,33% /11,67% /0% | 23,33% |

Ở seed42 trên60records, zone29,44%, STOP0%, false-watering77/100 target-OFF components. Đây là baseline được đo lại trên máy hiện tại; không dùng số test1,81% để so với overfit train. Reports `baseline-step6000-cases60.json` (300rawpredictions), `baseline-step6000-panels.json`, `panels/always-off.json`.

## Native sampling được đối chứng ngày15

Giữ nguyên checkpoint micro8 batch4/step1000, input và nămnoise seeds; chỉ đổi `SmolVLAConfig.num_steps`, là tham số Euler có thật của upstream:

| Euler steps | Exact trên40predictions | Continuous MAE trung bình5seeds |
|---|---:|---:|
| 1 | 40/40 | 0,13759 |
| 2 | 40/40 | 0,06566 |
| 4 | 40/40 | 0,10593 |
| 10 | 39/40 | 0,13867 |
| 20 | 38/40 | 0,15159 |

Chọn2bước làm candidate, không thay noise để chọn kết quả đẹp. Đây là kết quả trên model/tập nhỏ đã kiểm, không phải định luật rằng ít Euler steps luôn tốt hơn. Learned velocity field có thể tích lũy lỗi qua sampling; phép thử chưa xác định duy nhất cơ chế gây lỗi ở mọi model. Native forward/loss khi train vẫn giữ nguyên. Không dùng analytic-padding sampler: đối chứng trên model batch2 không sửa3 lỗi còn lại, nên đã loại biến thể đó.

Report đầy đủ: `reports/small-cases/sampling-numsteps-micro8-r8-b4-v3-v2.json` và các file `.steps-N.json`; script `audit_num_steps.py`. Attempt trước tên không có `-v2` dừng do writer exclusive của repo, chỉ là receipt chưa hoàn tất; không dùng nó làm kết luận cuối. Checkpoint không thay bytes.

Model6000 với Euler2 trên cùng60records vẫn chỉ có exact26,67/3,33/18,33/18,33/5% theo seeds0/1/2/3/42. Vì vậy đổi sampler riêng không cứu model cũ. Report: `baseline-step6000-cases60-n2.json`.

Config **`small_cases_v4.yaml`** ghi rõ native `num_steps`; loader thí nghiệm kiểm saved config đúng sampling steps của profile. Run `cases60-r8-b4-n2-lr3e4-v4` bắt đầu bằng weights của micro8 batch4, optimizer mới, rồi train trực tiếp60records. Panel32 nằm trong60 sẽ được tính điểm riêng; không coi đó là một run train32 độc lập. Các giới hạn/gates vẫn100% ở từngseed. Chưa chốt kết quả của run60 khi nó còn đang chạy.

**Run60 q/v đã kết thúc:** tại1000updates, screen exact43,33%/55% theo seed0/42; zone73,89%/81,11%, STOP100%/83,33%, MAE0,36766/0,32697. Không đạt gate100%, nên chưa dùng làm cấu hình full train. Giữ kết quả thất bại, không thay seed hoặc ngưỡng nghiệm thu.

## LoRA mở rộng trong action expert — tiếp tục ngày15/11

`small_cases_v6.yaml` giữ r8/alpha8, batch4, LR3e-4, warmup20/constant và native Euler2/Beta-time loss. Mở target LoRA thêm k/o và gate/up/down của expert; VLM/vision vẫn frozen. Native PEFT regex thực, không thêm flag giả. Có1.756.800trainable parameters; expansion giữ nguyên74tensors cũ, thêm160tensors trong expert với80LoRA B tensors bằng0. Vì vậy không loại bỏ trọng số đã học. Run này weights-only từ run60 narrow/step1000 và resetoptimizer, được ghi rõ trong experiment.json.

Step100 exact43,33%/53,33%; step30061,67%/70%, STOP100% ở cảhai seeds. Chưa quy toàn bộ cải thiện cho scope: model có thêm updates và lịch sử warm-start khác; chưa chạy narrow với cùng tổng budget làm đối chứng cô lập.

Run `cases60-r8-wide-b4-n2-lr3e4-v6` dừng ở step520 vì **CUDA unknown error**, không có bằng chứng đó là OOM. Checkpoint cuối an toàn300; report lỗi và520update receipts giữ nguyên. CUDA tensor kernel trong fresh process hoạt động lại. Không đổi driver/dependencies hoặc killprocess lạ.

Sau GPU resume proof99→100 wide PASS, `small_train --resume-checkpoint .../000300` tiếp tục vào **output mới** `cases60-r8-wide-b4-n2-resume300-v6`. Dùng native `load_training_state` khôi phục702optimizer tensors/8RNG tensors, scheduler/step và AMPscaler; tất cả exact. First resumed step301 khớp episodes/RNG và loss0,13624699413776398 của run liên tục trước khi gián đoạn. Wrapper dùng loader continuation đã được kiểm, không resetoptimizer; --init-checkpoint vẫn chỉ weights-only.

Resume chỉ hỗ trợ cùng panel/profile/batch/world1, workers0, không stochastic augmentation và constant scheduler. Mốc cuối1000 là số optimizer step tích lũy của state, **không phải thêm1000sau300**.220steps sau checkpoint300 của attempt lỗi bị bỏ khi khôi phục; không tính chúng hai lần vào model cuối.14testsPASS gồm CPU continuation20updates qua epoch trên8/B2 và60/B4.

Native pixel audit mới `native-pixel-and-subset-audit.json`:3600actions khớp source,10800camera tensors được hash,60subsetinputs/targets/tasks khớp cùng fulltrainrows,0image-path-ID mismatch và0actual-input→different-target conflicts. Không tìm thấy lỗi ghép input/target giải thích accuracy thấp. Phép kiểm này không xác minh mức héo bằng chuyên gia hoặc đảm bảo neural model sẽ học100%.

## Resume thật đã đối chiếu với nhánh liên tục

Dùng checkpoint99 và100 của run batch2 đã train thật; native `lerobot_train.train` gọi `load_training_state` và chạy một optimizer update. Saved optimizer/scheduler/RNG/step đều nạp đúng, nhưng prepared loader mới của Accelerate đặt lại epoch và tạo thêm CPU RNG draw giữa epoch. Bản native không điều chỉnh lấy episodes[3401,64] thay vì[3201,3200], loss0,3746077 thay vì0,03065127; adapter maxdifference0,00015527.

`small_resume.py` dùng public `loader.set_epoch()` để nối đúng epoch, giữ native sampler offset và bảo toàn CPU RNG quanh lần tạo iterator đầu nếu đang giữa epoch. Không sửa vendor. Sau sửa: episodes, bốn RNG fingerprints trước forward, loss/gradient norm trùng; **74adapter tensors,222optimizer tensors và8RNG tensors đều exact, maxerror0**. Scheduler/step/AMPscaler JSON trùng hoàn toàn. Tensor tolerance đã khai báo `atol=1e-6,rtol=1e-5`, thực đo0. CPU regression còn kiểm cả giữa và ranh giới epoch.

Đây là proof một update GPU thật, không phải checkpoint fixture. Giới hạn đã kiểm: mộtGPU, cùngbatch/world-size, workers0, không stochastic augmentation, constant scheduler. Nó chứng minh tính tiếp nối trong điều kiện đó; không chứng minh sửa resume tự làm accuracy full tăng. Lệnh tái lập (output mới):

```bash
agri-soybean-vla/.venv-train/bin/python -u -m agri_vla.small_resume \
  --checkpoint agri-soybean-vla/outputs/small-cases/micro8-r8-b2-lr3e4-v2/checkpoints/000099 \
  --control agri-soybean-vla/outputs/small-cases/micro8-r8-b2-lr3e4-v2/checkpoints/000100 \
  --restore-loader-continuation \
  --output agri-soybean-vla/outputs/small-cases/resume-fixed-reproduction
```

Reports: `resume-native-b2-99to100-v1/resume_report.json` (mismatch giữ làm đối chứng), `resume-fixed-b2-99to100-v1/resume_report.json` (passed). Lệnh này là công cụ proof một update, không phải CLI run dài. `--init-checkpoint` của small_train vẫn chỉ là weights-only, không thay thế full-state resume.

## Resource checks

Batch4 run micro: torch train peak allocated1.944GB/reserved1.965GB; NVML observed peak2214/4096MiB, headroom1882MiB; mean update2,699s. Poll NVML mỗi1giây; đây là peak quan sát toàn GPU trong run, có cả allocation của hệ thống. Chưa suy ra mọi workload tương lai đều có cùng headroom.

Longest-train-prompt probe batch2 đã PASS: actual74tokens, capacity96, ba encoder inputs[2,3,256,256], forward/backward finite, native optimizer state step1000 có trong GPU, weights không đổi và0optimizer updates. Torch peak allocated1.702GB/reserved1.732GB. Report `memory-longest-r8-b2-v2.json`. Candidate batch4/Euler2 cần resource check tương ứng sau khi có checkpoint của run mới.

## Liệu có tăng accuracy khi train tất cả?

Một cấu hình đạt các panel này sẽ chứng minh có thể học những phân biệt cụ thể đã kiểm và giúp loại cấu hình không fit được ngay trên train. Nó chưa đảm bảo accuracy trên3600train hoặc720val/test: độ đa dạng ảnh/câu và phân bố target lớn hơn nhiều. Sau khi đạt tập nhỏ mới chạy full train có validation định kỳ, so baseline luôn OFF và checkpoint6000 trên cùng protocol, báo riêng từng intent/ngôn ngữ/seen-unseen wording. Chọn config/checkpoint bằng validation; test để đánh giá cuối, không dùng để tuning.

## Kết quả tiếp tục ngày15/11 — sau các run hoàn tất

Wide LR3e-4 full-state resume300→1000 đã hoàn tất700updates mới, finite và adapter thay đổi. Step600 screen exact71,67%/91,67%; step1000 là71,67%/88,33%, STOP100% cả hai. Không chọn last checkpoint chỉ vì step lớn. Torch train peak allocated1.971.778.048bytes/reserved2.009.071.616bytes (1880,43/1916MiB), NVML peak3750/4096MiB; headroom quan sát346MiB. Meanupdate3,306s; có allocation hệ thống thay đổi theo thời điểm, không gọi3750MiB là riêng PyTorch.

Thử profile đã khai báo r8_wide_b4_n2_lr1e4, weights-only từ wide/step600, optimizer mới, tối đa300updates. Step100 exact83,33% ở cả seed0/42, STOP100%. Lệnh bị ngắt ở111 khi tương tác bị ngắt; không có result cuối, checkpoint an toàn100. Đã resume đầy đủ100→300 vào output mới; first resumed101 sameepisodes/RNG/loss0,12659825384616852. Step300 exact80%/98,33%, zone92,22%/99,44%, STOP100% cả hai; targetOFF false-watering0/1 trên100components. Chưa đạt strict gate100% mọiseed. Không chạy thêm training sau mốc này trong lượt chốt kết luận.

Bản99/100 là checkpoint đã optimizer train thật. Các báo cáo fresh-process reload100/300 ghi riêng batch1 ở5seeds, cùng120predictions B4 của screen gốc để kiểm sai số save/reload. Kết quả cuối và checkpoint dùng demo xem reports/small-cases/selected-model.json; không lấy riêng98,33% để gọi model đạt.

Full-run preparation: candidate_train.py tạo native config full/train3600, batch4, scope/LR/num_steps từ YAML, saved full-train stats và processor. Fresh full run là weights-only curriculum với optimizer mới; resume full chỉ từ checkpoint do runner này tạo, cùng batch/profile/data/world1/workers0/noaugmentation, AMPscaler và loader continuation. Nhánh fullGPU chưa chạy; dry-run CPU và actual fullnative data view3600/4 PASS, normalization maxerror0. Full runner yêu cầu evidence TRAIN60/five-seed/B1/reload đạt100%; không tự coi config preview là quality/long-run proof.

Hướng dẫn model, lệnh kiểm, full smoke/resume/VAL và tài nguyên: **RUN_SMALL_MODEL_AND_FULL_TRAIN.md**. CPU regression20PASS gồm sampler3600 trước/sau ranh giới epoch; đây không phải full GPU optimizer test. Configv5 endpoint là prototype chưa train, r16/LR và các YAML candidate khác không tự có kết quả nếu không có raw receipts. Không thay schema/stats/nhãn/camera, không train3600 hoặc tuning VAL/TEST, không commit/push.
