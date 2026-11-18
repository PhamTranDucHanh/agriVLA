from types import SimpleNamespace
from pathlib import Path
import json
import pytest


@pytest.mark.parametrize('research',[False,True])
def test_native_plan_keeps_quality_gate_false_without_evidence(monkeypatch,tmp_path,research):
    from agri_vla import candidate_train
    monkeypatch.setattr(candidate_train,'resolve',lambda *a,**kw:({}, {'_sha256':'config'},
        {'batch_size':4},SimpleNamespace(to_dict=lambda:{},batch_size=4),tmp_path/'model',0))
    monkeypatch.setattr(candidate_train,'runtime',lambda *a,**kw:None)
    args=SimpleNamespace(research_run=research,experiment_config=Path('config.json'),profile='control',
                         output=tmp_path/'out',steps=6000,init_checkpoint=tmp_path/'model',resume_from=None,
                         execute=False,evidence=None,validation_evidence=None,readiness_policy=None)
    plan=candidate_train.run(args)
    assert plan['research_run'] is research
    assert plan['full_execution_gate_passed'] is False
    assert plan['optimizer_steps_performed']==0


def research_fixture(tmp_path,monkeypatch):
    from agri_vla import full_research
    from agri_vla.training import file_sha
    monkeypatch.setattr(full_research,'project_root',lambda:tmp_path)
    cp=tmp_path/'outputs/small-cases/latekv-gpu20261116-control-s300/checkpoints/000300/pretrained_model'
    cp.mkdir(parents=True);(cp/'adapter_model.safetensors').write_bytes(b'unit fixture only')
    rep=tmp_path/'reports/latekv-gpu-20261116';rep.mkdir(parents=True)
    (rep/'control-final.json').write_text(json.dumps({'adapter_sha256':file_sha(cp/'adapter_model.safetensors')}))
    (rep/'smoke_report.json').write_text(json.dumps({'resume_status':{'control':'passed'}}))
    (tmp_path/'configs').mkdir();(tmp_path/'configs/late_vlm_kv_v1.json').write_text('{}')
    monkeypatch.setattr(full_research,'gpu_headroom',lambda *a:pytest.fail('Must not touch GPU'))
    return SimpleNamespace(output=tmp_path/'outputs/train/new',steps=6000,max_new_gib=1.,patience=3,
                           execute=False,research_run=False,resume=False)


def test_research_plan_does_not_train_or_claim_readiness(monkeypatch,tmp_path):
    from agri_vla.full_research import run
    args=research_fixture(tmp_path,monkeypatch);p=run(args)
    assert p['milestones']==[100,1000,2000,3000,4000,5000,6000]
    assert not p['readiness_gate_passed'] and not p['TEST_opened'] and not p['full_training_executed']
    assert not args.output.exists()


def test_explicit_research_flag_is_required_before_gpu(monkeypatch,tmp_path):
    from agri_vla.full_research import run
    args=research_fixture(tmp_path,monkeypatch);args.execute=True
    with pytest.raises(ValueError,match='Explicit --research-run'):run(args)


def test_workflow_preserves_unqualified_status_and_never_launches_test(monkeypatch,tmp_path):
    """Subprocess/file fixtures only: not GPU or trained-resume evidence."""
    from agri_vla import full_research,full_run
    args=research_fixture(tmp_path,monkeypatch);args.execute=args.research_run=True;args.steps=100
    monkeypatch.setattr(full_research.sys,'prefix',str(tmp_path/'.venv-train'))
    monkeypatch.setattr(full_research,'gpu_headroom',lambda *a:None)
    monkeypatch.setattr(full_research,'complete_checkpoint',lambda p:True)
    monkeypatch.setattr(full_run,'project_root',lambda:tmp_path)
    calls=[]
    def fake_subprocess(argv,**kw):
        calls.append(argv)
        def value(flag):return argv[argv.index(flag)+1]
        if 'memory' in argv:
            Path(value('--output')).write_text(json.dumps({'status':'pass'}))
        elif 'phase' in argv:
            assert '--research-run' in argv
            cp=Path(value('--output'))/'checkpoints/000100/pretrained_model';cp.mkdir(parents=True)
            (cp/'adapter_model.safetensors').write_bytes(b'fake trained tensor for routing test only')
        elif 'evaluate' in argv:
            assert value('--split')=='val'
            out=Path(value('--output'));out.mkdir()
            (out/'metrics.json').write_text(json.dumps({'partial':False,'metrics':{'overall':{
                'records':720,'exact_action_accuracy':.4,'zone_accuracy':.6,'continuous_mae':.3}}}))
        else:pytest.fail('Unexpected command')
    monkeypatch.setattr(full_research.subprocess,'run',fake_subprocess)
    result=full_research.run(args)
    assert result['status']=='research_completed_not_qualified'
    assert result['readiness_gate_passed'] is False and result['TEST_opened'] is False
    assert len(calls)==3 and (args.output/'deploy.sh').exists()
