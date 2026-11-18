"""Scope/labels/allowlist/contract tests; no optimizer or backward calls."""
import json
import re

import pytest

from agri_vla.late_vlm import (KV,action_target,ordered_images,target_regex,metadata,
                              validate_metadata,resume_contract,recipe,model_batch)
from agri_vla.late_vlm_experiment import plan


def test_only_last_active_KV_is_added():
    pattern=target_regex('candidate');control=target_regex('control')
    for name in KV:assert re.fullmatch(pattern,name) and not re.fullmatch(control,name)
    for layer in (14,16,31):
        assert not re.fullmatch(pattern,f'model.vlm_with_expert.vlm.model.text_model.layers.{layer}.self_attn.v_proj')
    assert not re.fullmatch(pattern,KV[0].replace('k_proj','q_proj'))
    assert re.fullmatch(pattern,'model.vlm_with_expert.lm_expert.layers.15.self_attn.q_proj')
    assert recipe()['trainable_parameters']['candidate']-recipe()['trainable_parameters']['control']==2*8*(960+320)


def test_action_levels_keep_integer_mapping_and_zone_order():
    for level in (0,1,2,3):assert action_target([level]*3)==[level]*3
    assert action_target([3,0,2])==[3,0,2]
    assert ordered_images({'zone3':'c','zone1':'a','zone2':'b'})==['a','b','c']


@pytest.mark.parametrize('values',[[False,0,1],[1.,2,3],[0,1,4],[0,1],[-1,0,1]])
def test_reject_invalid_targets(values):
    with pytest.raises(ValueError):action_target(values)


def test_metadata_is_not_inference_input():
    allowed=['observation.images.camera1','observation.images.camera2','observation.images.camera3','observation.state','task']
    raw={k:object() for k in [*allowed,'teacher_annotations','intent','target_action','action','wilt_notes','case_id']}
    result=model_batch(raw)
    assert list(result)==allowed and all(result[k] is raw[k] for k in allowed)


def test_typed_metadata_and_fixture_resume_fail_closed(tmp_path):
    value=metadata('candidate');assert validate_metadata(value)==value
    folder=tmp_path/'pretrained_model';folder.mkdir();(folder/'late_vlm.json').write_text(json.dumps(value))
    with pytest.raises(ValueError):resume_contract(tmp_path,'candidate')
    value['target_modules']=target_regex('control')
    with pytest.raises(ValueError):validate_metadata(value)


def test_default_plan_has_no_training_and_enforces_budget():
    from agri_vla.preflight import project_root
    out=project_root()/'outputs/small-cases/test-candidate-never-executed'
    result=plan('candidate',2,out)
    assert result['mode']=='plan_only' and not result['training_executed'] and not out.exists()
    with pytest.raises(ValueError):plan('candidate',301,out)
