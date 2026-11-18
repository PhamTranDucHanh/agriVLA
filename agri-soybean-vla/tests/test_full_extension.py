import json
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager
import pytest
from agri_vla import full_research as runner
from agri_vla.training import file_sha
from test_full_research import research_fixture


def setup_extension(tmp_path,monkeypatch):
    args=research_fixture(tmp_path,monkeypatch)
    parent=tmp_path/'outputs/train/completed6000'
    cp=parent/'completed6000-phase-006000/checkpoints/006000'
    (cp/'pretrained_model').mkdir(parents=True)
    (cp/'pretrained_model/adapter_model.safetensors').write_bytes(b'unit test only, not trained proof')
    (cp/'pretrained_model/agri_full_candidate.json').write_text(json.dumps({'research_run':True,
        'runner_sha256':file_sha(Path(runner.__file__).with_name('candidate_train.py'))}))
    plan={'profile':'control','batch_size':4,'lr':1e-4,'objective':'endpoint_t1','scheduler':'constant_with_warmup',
          'patience':3,'TEST_opened':False,'config_sha256':file_sha(tmp_path/'configs/late_vlm_kv_v1.json'),
          'code_sha256':{'full_research.py':'old_driver','candidate_train.py':file_sha(Path(runner.__file__).with_name('candidate_train.py'))}}
    metrics={'overall':{'records':720,'exact_action_accuracy':.3,'zone_accuracy':.5,'continuous_mae':.7}}
    progress={'status':'research_completed_not_qualified','completed_steps':6000,'best':str(cp),'latest':str(cp),
              'validations':[{'step':6000,'checkpoint':str(cp),'metrics':metrics}]}
    (parent/'run-plan.json').write_text(json.dumps(plan));(parent/'progress.json').write_text(json.dumps(progress))
    monkeypatch.setattr(runner,'complete_checkpoint',lambda p:True)
    args.extend_from=parent;args.steps=12000
    return args,cp


def test_extension_plan_starts_after6000_and_preserves_optimizer(tmp_path,monkeypatch):
    args,cp=setup_extension(tmp_path,monkeypatch);p=runner.run(args)
    assert p['milestones']==[7000,8000,9000,10000,11000,12000]
    assert p['extension']['optimizer_reset'] is False and p['extension']['native_start_step']==6000
    assert p['source_checkpoint']==str(cp/'pretrained_model') and not args.output.exists()


def test_reject_incomplete_parent_and_changed_training_source(tmp_path,monkeypatch):
    args,cp=setup_extension(tmp_path,monkeypatch)
    path=args.extend_from/'progress.json';p=json.loads(path.read_text());p['completed_steps']=5000;path.write_text(json.dumps(p))
    with pytest.raises(ValueError,match='normally completed'):runner.run(args)
    p['completed_steps']=6000;path.write_text(json.dumps(p))
    path=args.extend_from/'run-plan.json';p=json.loads(path.read_text());p['code_sha256']['candidate_train.py']='drift';path.write_text(json.dumps(p))
    with pytest.raises(ValueError,match='source changed'):runner.run(args)


def test_parent_is_not_reused_as_output(tmp_path,monkeypatch):
    args,cp=setup_extension(tmp_path,monkeypatch);args.output=args.extend_from
    with pytest.raises(ValueError,match='NEW output'):runner.run(args)


def test_extension_workflow_routes_native_resume_and_early_stops(tmp_path,monkeypatch):
    """Mock orchestration only; not actual GPU/model proof."""
    from agri_vla import full_run
    args,original=setup_extension(tmp_path,monkeypatch);args.execute=args.research_run=True
    monkeypatch.setattr(runner.sys,'prefix',str(tmp_path/'.venv-train'))
    monkeypatch.setattr(runner,'gpu_headroom',lambda *a:None)
    monkeypatch.setattr(full_run,'project_root',lambda:tmp_path)
    calls=[]
    def invoke(argv,**kw):
        calls.append(argv)
        val=lambda flag:argv[argv.index(flag)+1]
        if 'memory' in argv:Path(val('--output')).write_text(json.dumps({'status':'pass'}))
        elif 'phase' in argv:
            assert 'agri_vla.full_research_extension' in argv and '--resume-from' in argv
            assert '--init-checkpoint' not in argv
            p=Path(val('--output'))/'checkpoints'/f'{int(val("--steps")):06d}'/'pretrained_model'
            p.mkdir(parents=True);(p/'adapter_model.safetensors').write_bytes(b'fake routing checkpoint')
        elif 'evaluate' in argv:
            assert val('--split')=='val'
            out=Path(val('--output'));out.mkdir()
            score=.4 if out.name=='val-007000' else .2
            (out/'metrics.json').write_text(json.dumps({'partial':False,'metrics':{'overall':{
                'records':720,'exact_action_accuracy':score,'zone_accuracy':.5,'continuous_mae':.7}}}))
    monkeypatch.setattr(runner.subprocess,'run',invoke)
    result=runner.run(args)
    assert result['completed_steps']==10000 and result['early_stop']
    assert result['TEST_opened'] is False and original.exists()
    progress=json.loads((args.output/'progress.json').read_text())
    assert progress['validations'][0]['step']==6000
    assert int(Path(progress['best']).name)==7000


def test_phase_requires_native6000_resume(monkeypatch,tmp_path):
    from agri_vla import full_research_extension as ext
    a=SimpleNamespace(research_run=True,execute=True,profile='control',steps=6000,resume_from=tmp_path)
    with pytest.raises(ValueError,match='6001..12000'):ext.phase(a)
