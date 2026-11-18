import json
from pathlib import Path
import pytest
from agri_vla import candidate_train


def test_report_excludes_runtime_processors_but_keeps_actual_metrics():
    state={'step':100,'pre':object(),'post':object(),'train_peak_allocated':123,
           'training_time_sampling':{'mode':'endpoint_t1'},'adapter_weight_squared_delta':.5}
    report=candidate_train.reportable_state(state)
    assert json.loads(json.dumps(report))=={k:v for k,v in state.items() if k not in ('pre','post')}
    assert 'pre' in state and 'post' in state


def test_unknown_runtime_value_or_nan_does_not_silently_disappear():
    with pytest.raises(TypeError):candidate_train.reportable_state({'step':100,'unexpected':object()})
    with pytest.raises(ValueError):candidate_train.reportable_state({'loss':float('nan')})


def test_resume_only_accepts_exact_audited_report_fix_predecessor(monkeypatch,tmp_path):
    original=tmp_path/'runner-source.py';original.write_text('old exact source fixture')
    old=candidate_train.REPORT_SERIALIZATION_PREDECESSOR
    monkeypatch.setattr(candidate_train,'file_sha',lambda p:old if Path(p)==original else 'current')
    candidate_train.check_resume_runner({'runner_sha256':old,'research_run':True},original)
    candidate_train.check_resume_runner({'runner_sha256':'current'},tmp_path/'absent')
    for meta in ({'runner_sha256':old,'research_run':False},{'runner_sha256':'unreviewed','research_run':True}):
        with pytest.raises(ValueError):candidate_train.check_resume_runner(meta,original)
    with pytest.raises(ValueError):candidate_train.check_resume_runner({'runner_sha256':old,'research_run':True},tmp_path/'missing')
