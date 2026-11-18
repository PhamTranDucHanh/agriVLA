# Prompt4 — thiết kế trước triển khai: LoRA K/V ở VLM block15

**Cập nhật lượt GPU được người dùng cho chạy sau Prompt4:** xem [LATE_VLM_GPU_2026-11-16.md](LATE_VLM_GPU_2026-11-16.md). Đã chạy matched300updates/arm, genuine resume và freshreload PASS; VALID mean8,57%→8,75% chưa đủ improvement/full-ready. Phần “chưa train/GPU” bên dưới mô tả đúng điểm dừng Prompt4 trước lượt GPU này, không phải trạng thái mới nhất. Lệnh full/web mới: [FULL_LATE_VLM_GUIDE.md](FULL_LATE_VLM_GUIDE.md).

## Quyết định và giới hạn bằng chứng

Chọn **nhánh A, đúng một thay đổi: thêm LoRA r8/alpha8 vào K/V của block VLM15 đang được dùng**. Giữ expert-wide LoRA, endpoint objective, action head, LR/batch/data/normalization/inference hiện hành. Không thêm auxiliary supervision, không thay action head, không mở vision encoder. Chưa train trong Prompt4.

Đối chứng mới nhất: VALID112 mean5noise seeds fresh5,89%, warm8,57%; AUTO/EXCLUDE0%, STOP và WATER_ALL cũng thấp; chưa fit TRAIN280. Chưa có probe chứng minh representation đủ thông tin. Vì thế nhánh B chưa có tiền đề; chỉ đổi decoder/noise có thể che nút thắt conditioning. Nhánh A là **giả thuyết cần kiểm**, không phải kết luận đã chứng minh frozenVLM là nguyên nhân.

## Module thực tế và số tham số

Pinned SmolVLA dùng16 VLM blocks0..15, cross_attn, self_attn_every_n_layers=2. Backbone có32blocks nhưng policy chỉ dùng16. Paths trước PEFT:

```
model.vlm_with_expert.vlm.model.text_model.layers.15.self_attn.k_proj
model.vlm_with_expert.vlm.model.text_model.layers.15.self_attn.v_proj
```

Header weights thực xác nhận cảhai matrices[320,960]. Native finalcross-attention dùng K/V này vào expert.layer15; Q/output prefix cuối không nằm trên đường trực tiếp tới suffix action cuối. Không targetlm_head hoặc32blocks backbone không được dùng.

Mỗi projection thêm r×(in+out)=8×(960+320)=10240params. Hai projection thêm **20480**, tổngcandidate **1777280** soexpertcontrol1756800;4tensor LoRA mới. BaseVLM/vision vẫn frozen; chỉLoRAK/V mới cógrad. PrefixVLM.eval() không đồng nghĩa no_grad; giữ train_expert_only=True, PEFT bậtrequires_grad cho các adapter được allowlist.

## Loss và gradient

Native flow velocity MSE chỉ3actiondims đang dùng; endpointt=1: x=noise, velocitytarget=noise−normalized_action. MộtEulerstep cho action=noise−v. Không đổi loss hoặc lượng tử clip0..3/floor(x+.5). Gradient từ action loss→expert cross-attn15→VLM K/V15 LoRA. Không đưa teacher/intent/wilt vào features; ba camera, câu, zerostate vẫn là input. Không ghi đè output STOP/WATER_ALL.

Khởi tạo mới copy **đúng234expert adapter tensors** từ endpoint60 checkpoint600; thêm K/V LoRA với B=0, giữ hàm ban đầu. Warm source chỉ là reference cố định cho matched control, **không được gọi là init thắng Prompt3**. Cảhai arm phải train lại cùng RNG barrier/dataorder/optimizer mới; không tái dùng sốB cũ khi RNG/protocol mới khác.

## Loader / save / resume contract

Configtyped`late_vlm_kv_v1`; loader riêng kiểm arm,targetregex,layercount,shapes,paramcount,sourcehash,config/codehash và mọi checkpointcontract. Metadata`late_vlm.json` cùng nativePEFTadapter/config/tokenizer/processors/stats/schema/provenance. Checkpoint train còn giữ nativeoptimizer/scheduler/RNG/step và AMPscaler. Generic small_cases v9 loader không được coi là loader của variant này.

CPU scope-expansion fixture không có optimizer step mới và **không phải resume proof**. Resume chỉ nhận checkpoint train thật, cùngarm/recipe/panel/world1/workers0/noaugmentation; native state+AMP+preparedloader continuation. Fresh RNG barrier chỉ áp dụng fresh run, không reset RNG sau restore. Phải so actual trained save→fresh-process reload→resume với nhánh liên tục trước khi gọi resume verified; Prompt4 không thực hiện trainingproof đó.

## Protocol đối chứng và điều kiện tiếp tục/bỏ

OriginalTRAIN280/fixedVALID112, batch4/LR1e-4/warmup20/constant, trainingseed42; cùng warm source, native endpointEuler1, cùng thứ tựdata/noise. Control expert-only vs candidate expert+KV15. Mốc100/300, tốiđa300updates/arm; nămnoise seeds0/1/2/3/42 khi xác nhậnVALID; không dùngTEST. Không traincounterfactual cùng lúc. Native save/loop/update/optimizer; không sửa vendor, không bịa accumulation/4bitflags.

Smoke tương lai2updates kiểm finite grads, adapterdelta và **grad củaKV15.B khôngzero**; zeroB khiếnLoRA.A grad đầu có thểzero bình thường. Đo longestTRAINprompt, Torchallocated/reserved,totalGPU/NVML,step/RAM/disk; OOM dừng, không bỏcamera. Không hứa fit4GB: tăngweights rất nhỏ không đủ để suy memorybackward.

Tiếp tục nếu candidate cải thiện meanVALIDexact≥5điểm%,zone≥2điểm%, thắng≥4/5noise seeds, conditionalpairedCI positive, khôngregressSTOP/falsewatering/new220 và old60drop≤10điểm%. Giữ readinessgates cũ cho full. Bỏ hoặc chưa kết luận nếu chỉTRAINloss/auxmetric tốt, KV không ảnh hưởngaction, nhiềugrad ngoàiallowlist, khôngfitmemory, hoặcVALIDkhôngcải thiện. Mộttrainingseed không đo training-seed variability.

Tài liệu này được tạo trước implementation. Phần dưới ghi implementation và kiểm chứng đã hoàn tất; chưa có sốquality của biến thể.

## Implementation reviewable

-`src/agri_vla/late_vlm.py`: scopeallowlist, source-copy, typedloader, checkpoint/fixture và resumeguards.
-`late_vlm_experiment.py`: defaultplan, bounded native runner qua scopedprojecthooks; không sửa vendor hoặc small_train.py. Native loop/update/optimizer/scheduler/save vẫn là upstream thực. Hooks khôi phục trongfinally; freshRNGbarrier sau khi dựngpolicy/optimizer áp dụng giốngnhau cho haiarm, resume khôngresetbarrier.
-`late_vlm_inputs.py`: raw3ảnh theo namedzones→native-sized tensor, zerostate, instruction; không nhận intent/wilt/target.
-`late_vlm_evaluate.py`: checkpoint đãtrain thật→originalTRAIN280 seed42 vàfixedVALID112 seed42 hoặc5noise; từ chối CPUfixture. Không đọcTEST.
-`late_vlm_checks.py`: CPUforward/reload; wrappers cho longest-prompt memory vàactual99→100resumeproof, chỉ nhận native-trained checkpoint. GPUchecks chưa được chạy.
-`scripts/late_vlm_v1.sh`: defaultplan, `set -e` dừng script nếu một arm fail/OOM. Smoke2→resume100→resume300 giữ tổng300updates/arm, không cộngthành402. Không kill process khác hoặc tự bỏcamera.

Typedmetadata pin recipe/modelscope/runner/legacyengine hashes, sourceadapter/statshashes và loạicheckpoint. SourcefullTRAINstats rawSHA b0684934…; bản saved JSON có byteSHA f0c7435f… doformatindent, **parsedJSON bằng nhau hoàn toàn**. Loader phân biệt hai hash, không thay stats hay tính stats mới. Test/guard banđầu đã phát hiện việc nhầm rawSHA với savedSHA; đã sửa trong loader mới, không sửa bất kỳ statsfile nào.

## CPU evidence thực, không train

Canonical receipts: **`reports/late-vlm-kv-v1-cpu-v3/`**. Những folderCPUv1/v2 là kiểm trước khi chốt configacceptance; chỉv3 có recipe cuối. Fixturev3 là copy scopeexpansion từ trained expert cũ, **0optimizerupdates mới**, không phải model mới đãtrain.

-13unit/nativeCPUtestsPASS: targetlevels/domain, cameraorder bằng ảnh đỏ/xanh, inputallowlist, typedmetadata, fixture-resume rejection, drybudget và hookrestoration/RNGbarrier khiresume.
-Header và model thực: K/Vblock15[320,960], control1756800/candidate1777280params; thêm4LoRAtensors.234expert tensors copyexact; mớiB=0.
-ForwardCPUactualpinnedSmolVLA/ảnhTRAIN: outputcontrol vàcandidatezeroB khớp **maxerror0**. Hai K/V thật được gọi mỗi cái một lần ởprefixprefill.
-Perturb B củaK rồiV **tạm trongRAM**, không optimizer/backward: actioncontinuous thay đổi max≈0,01763/0,04848. Saukhikhôi phụcB=0 outputtrởlạireferenceexact. Điều này kiểm đườnghàm thực sự ảnh hưởngaction, **chưa chứng minh gradientGPU hữu ích**.
-Poison metadata teacher/intent/target không thayprediction. Rawinputbuilder cho ba ảnhTRAIN khớpnative tensors **maxpixelerror0 cả3camera**.
-Savedprocessor round-trip targetuniformlevels0/1/2/3 đúng trong1e-6; `train()` không vô tình refreeze adapters, baseVLM vẫn eval.
-Savefixture→**processCPU mới** reload→nativeinference khớp maxerror0, tolerance1e-5, quantizedexact. Khôngcó optimizer/scheduler/RNG của một run mới để chứng nhậnresume.

Chưa train/backward, chưa GPUmemorytest, chưa VALIDbenchmark cho variant. Không đổi split/sourceimages/targets/tokenizer/normalization, modeldeploy hoặc readiness. Một ví dụCPU đúngtarget vốn có thể đã được expertwarm cũ học; không tính đó là accuracy cải thiện.

## Lệnh để người dùng chạy sau

Từ repo root, giữ mộtTAG cho cảcácphase. LượtPrompt4 hiện tại **chỉ chạy plan/CPU**, chưa chạy các lệnhGPU bên dưới.

```bash
cd /home/hanhpham/agriVLA
AGRI_TAG=latekv_try1
AGRI_PY="$PWD/agri-soybean-vla/.venv-train/bin/python"
AGRI_OUT="$PWD/agri-soybean-vla/outputs/small-cases/latekv-$AGRI_TAG"
export HF_DATASETS_CACHE="$PWD/agri-soybean-vla/.cache/hf-datasets"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Chỉ inplan, không tạo model/run:
bash agri-soybean-vla/scripts/late_vlm_v1.sh plan "$AGRI_TAG"

# GPU thật:2optimizerupdates mỗiarm, native checkpointstep2:
bash agri-soybean-vla/scripts/late_vlm_v1.sh smoke "$AGRI_TAG"

# LongestTRAINprompt B4 +longestVALinference, load nativeoptimizer thật,
# backward nhưng0updates. NVML trongprobe làsnapshotcuối; lấy totalpeak
# từ telemetry native run, không nhầm snapshot sauinference làtrainingpeak.
"$AGRI_PY" -m agri_vla.late_vlm_checks memory \
  --checkpoint "$AGRI_OUT-candidate-s2/checkpoints/000002/pretrained_model" \
  --output "agri-soybean-vla/reports/latekv-$AGRI_TAG-memory.json"

# Tiếp nốistate2→100 cho cảhai, thêm98updates/arm:
bash agri-soybean-vla/scripts/late_vlm_v1.sh 100 "$AGRI_TAG"
```

Smoke2 còn có TRAIN60screen/init/save nên không phải jobchỉvàigiây. Dừng ởmốc100 để đọcfinite/KVgradient/VRAM/predictions, không tựchochạy300 nếu lỗi. Sau100 cócheckpoints99/100 thật để kiểmresume:

```bash
"$AGRI_PY" -m agri_vla.late_vlm_checks resume \
  --checkpoint "$AGRI_OUT-candidate-s100/checkpoints/000099" \
  --control "$AGRI_OUT-candidate-s100/checkpoints/000100" \
  --output "agri-soybean-vla/outputs/small-cases/latekv-$AGRI_TAG-candidate-proof99"

# FrozenoriginalTRAIN280+VALID112, khôngTEST:
"$AGRI_PY" -m agri_vla.late_vlm_evaluate \
  --checkpoint "$AGRI_OUT-candidate-s100/checkpoints/000100/pretrained_model" \
  --output "agri-soybean-vla/reports/latekv-$AGRI_TAG-candidate100.json" --seeds 42
```

Chạy cùng lệnhresume/evaluate cho`control`, thaycandidate trongpaths. Resumeproof phải PASS thật mới gọi resumeverified. Proof99→100 thêm1update chẩnđoán ởoutputriêng, không dùng weightsproof thay main run. Nếu tiếp tụcthửmốc300 sauđánhgiá100:

```bash
# Native100→300, thêm200updates/arm; tổngmaintraining300/arm:
bash agri-soybean-vla/scripts/late_vlm_v1.sh 300 "$AGRI_TAG"

"$AGRI_PY" -m agri_vla.late_vlm_evaluate \
  --checkpoint "$AGRI_OUT-candidate-s300/checkpoints/000300/pretrained_model" \
  --output "agri-soybean-vla/reports/latekv-$AGRI_TAG-candidate300.json" \
  --seeds 0 1 2 3 42
```

Đánhgiácontrol300tươngtự, cùngseeds/panels. Đối chiếuactualepisodeorder/RNG ởupdates giữaarms, TRAINold60/new220 vàVALIDintent/VI/EN/seen/unseen. Các APIgiữrawpredictions, không chọnseedđẹp. Không dùngB lịch sử ởPrompt3 làm controlcủaprotocol này vì freshRNGbarrier đãđược địnhnghĩa khác; phải rerunmatchedcontrol.

Tái kiểmCPU vào **foldermới**:

```bash
"$AGRI_PY" -m agri_vla.late_vlm_checks cpu \
  --output /tmp/agrivla-late-kv-cpu-recheck

"$AGRI_PY" -m pytest -q \
  agri-soybean-vla/tests/test_late_vlm.py \
  agri-soybean-vla/tests/test_late_vlm_native.py
```

Không ghiđèoutput. RerunTAGcũ sẽbị từchối; chọnTAGmới hoặc đúngresumephase. Không dùngCPUfixture làmtrainedcheckpoint để vượtgates.

## Chi phí và giới hạn

Thêm20480FP32params chỉ≈80KiBweights,≈160KiBAdam moments và≈80KiBgrad; **không bao gồm activation/backward attention**. Không thể suy VRAMtổng chỉ từ20kparams. GPU4GB chưa được chứng nhận cho candidate; longest-prompt backward/native optimizer phải đo thật. Baselinecũ≈2,85s/update,300updates gồmsetup/screens≈18phút chỉ là mốc tham khảo, không phảiETAđãđo củacandidate. Không chốtETAfull khi chưa cópilotGPU mới.

Chỉadapter/tokenizer/contract được lưu, dùngchungbase/backbone cache. Budgetguard1GiB/run,300steps/arm; footprintdựkiến vài–vàichụcMiB/checkpoint, không nhânbảnảnh. Native runner hiện giữ2/99/100/300 ởcácphase để phục vụproof; saukhi kiểm hãy giữbest+latest thuộcrunmới, không dọnmodelcũ. Khôngcoi giới hạnlogicalfile này là giới hạnVHDX/swapWindows.

Fullrecipe/fullaccuracy chưa có, không tạofull_ready.json. Nếu hypothesisKV15không giúpVALID hoặc chỉ tăngmemorization thì bỏnhánh, không tăngrank/LR/stepsđồngloạt để cứu một kết luận đãđịnhtrước.
