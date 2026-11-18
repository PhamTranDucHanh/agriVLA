from pathlib import Path
import pytest
from agri_vla.dataset import DatasetError
from agri_vla import full_run


def test_validation_ranking_uses_exact_then_zone_then_mae():
    def score(exact, zone, mae):
        return full_run.ranking(dict(exact_action_accuracy=exact, zone_accuracy=zone, continuous_mae=mae))
    assert score(.9,.9,.2) > score(.8,1.,.01)
    assert score(.9,.95,.3) > score(.9,.9,.1)
    assert score(.9,.95,.1) > score(.9,.95,.3)
    with pytest.raises(DatasetError):score(float('nan'),1.,0.)


def test_retention_keeps_best_latest_and_never_prunes_source(tmp_path, monkeypatch):
    monkeypatch.setattr(full_run,'project_root',lambda:tmp_path)
    root=tmp_path/'outputs/train/unit-run';root.mkdir(parents=True)
    (root/'run-plan.json').write_text('{}')
    source=tmp_path/'original-source';source.mkdir();(source/'keep').write_text('source')
    cps=[]
    for step in (100,500,1000):
        cp=root/'unit-run-phase-001000/checkpoints'/f'{step:06d}'
        (cp/'pretrained_model').mkdir(parents=True)
        (cp/'pretrained_model/adapter_model.safetensors').write_bytes(b'prune test only, not a model')
        cps.append(cp)
    deleted=full_run.prune(root,[cps[0],cps[2]])
    assert len(deleted)==1 and not cps[1].exists() and cps[0].exists() and cps[2].exists()
    assert (source/'keep').read_text()=='source'
    with pytest.raises(DatasetError):full_run.prune(root,[])
    unsafe=tmp_path/'outputs/train/not-owned';unsafe.mkdir()
    with pytest.raises(DatasetError):full_run.prune(unsafe,[cps[0]])


def test_retention_rejects_checkpoint_symlink(tmp_path,monkeypatch):
    monkeypatch.setattr(full_run,'project_root',lambda:tmp_path)
    root=tmp_path/'outputs/train/unit';root.mkdir(parents=True);(root/'run-plan.json').write_text('{}')
    outside=tmp_path/'source';outside.mkdir()
    cp=root/'unit-phase-000100/checkpoints/000100';cp.parent.mkdir(parents=True)
    cp.symlink_to(outside,target_is_directory=True)
    with pytest.raises(DatasetError):full_run.prune(root,[cp])


def workflow_fixture(tmp_path, monkeypatch, *, fail_after_save=False):
    """Simulate subprocess files only; this is NOT trained/GPU resume evidence."""
    import json,sys,subprocess
    from types import SimpleNamespace
    from agri_vla import candidate_train
    from agri_vla import readiness
    root=tmp_path;monkeypatch.setattr(full_run,'project_root',lambda:root)
    monkeypatch.setattr(sys,'prefix',str(root/'.venv-train'))
    source=root/'small/pretrained_model';source.mkdir(parents=True)
    (source/'adapter_model.safetensors').write_bytes(b'fixture source, not NN')
    (source/'small_experiment.json').write_text(json.dumps({'profile':{'rank':8}}))
    config=root/'config.yaml';config.write_text('unit test')
    evidence=root/'evidence.json';evidence.write_text('{}')
    memory=root/'memory.json';memory.write_text(json.dumps({'status':'pass','checkpoint':str(source),'torch_peak_reserved':1000}))
    control=root/'control/pretrained_model';control.mkdir(parents=True)
    (control/'small_experiment.json').write_text(json.dumps({'profile':{'rank':8}}))
    resume=root/'resume.json';resume.write_text(json.dumps({'status':'passed','control_checkpoint':str(control.parent)}))
    recipe=root/'recipe.json';recipe.write_text(json.dumps({'version':1,'checkpoint':str(source),'config':str(config),
        'config_sha256':full_run.file_sha(config),'adapter_sha256':full_run.file_sha(source/'adapter_model.safetensors'),
        'profile':'unit','evidence':str(evidence),'validation_evidence':str(evidence),
        'memory_evidence':str(memory),'memory_evidence_sha256':full_run.file_sha(memory),
        'resume_evidence':str(resume),'resume_evidence_sha256':full_run.file_sha(resume)}))
    monkeypatch.setattr(candidate_train,'check_evidence',lambda *a:{'scope':'unit only'})
    monkeypatch.setattr(readiness,'check_validation_evidence',lambda *a:{'scope':'unit only'})
    monkeypatch.setattr(full_run,'complete_checkpoint',lambda cp:(cp/'pretrained_model/adapter_model.safetensors').is_file())
    monkeypatch.setattr(full_run,'baseline',lambda split:{'overall':{'exact_action_accuracy':.2}})
    monkeypatch.setattr(full_run,'web_smoke',lambda *a:{'technical_web_passed':True,'example_correct':True})
    monkeypatch.setattr(full_run,'gpu_headroom',lambda *a:None)
    calls=[];fail=[fail_after_save]
    def execute(argv,**kwargs):
        calls.append(argv)
        def value(flag):return argv[argv.index(flag)+1]
        out=Path(value('--output'));out.mkdir(parents=True,exist_ok=True)
        if 'agri_vla.candidate_train' in argv:
            step=int(value('--steps'))
            cp=out/'checkpoints'/f'{step:06d}'
            (cp/'pretrained_model').mkdir(parents=True)
            (cp/'pretrained_model/adapter_model.safetensors').write_bytes(str(step).encode())
            (cp/'pretrained_model/agri_checkpoint_lock.json').write_text('{}')
            (cp/'training_state').mkdir();(cp/'training_state/amp_scaler_state.json').write_text('{}')
            if fail[0] and step==2000:
                fail[0]=False
                raise subprocess.CalledProcessError(1,argv)
        else:
            split=value('--split');step=int(Path(value('--checkpoint')).parent.name)
            exact=.92 if split=='test' else .8 if step==1000 else .7
            report={'partial':False,'metrics':{'overall':{'records':720,'exact_action_accuracy':exact,
                    'zone_accuracy':.97,'continuous_mae':.1,'stop_exact_accuracy':1.}}}
            (out/'metrics.json').write_text(json.dumps(report))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(full_run.subprocess,'run',execute)
    args=SimpleNamespace(recipe=recipe,steps=2000,max_new_gib=.25,output=root/'outputs/train/unit',execute=True,resume=False)
    return args,calls,source


def test_workflow_selects_by_val_and_tests_only_after_selection(tmp_path,monkeypatch):
    args,calls,source=workflow_fixture(tmp_path,monkeypatch)
    report=full_run.run(args)
    assert report['best_checkpoint'].endswith('/001000')
    assert (args.output/'final_model/adapter_model.safetensors').read_bytes()==b'1000'
    assert (source/'adapter_model.safetensors').read_bytes()==b'fixture source, not NN'
    evaluations=[a[a.index('--split')+1] for a in calls if '--split' in a]
    assert evaluations==['val','val','test']
    assert len(list(args.output.glob('*-phase-*/checkpoints/[0-9]*')))==2


def test_workflow_recovers_saved_phase_without_training_it_twice(tmp_path,monkeypatch):
    import subprocess
    args,calls,_=workflow_fixture(tmp_path,monkeypatch,fail_after_save=True)
    with pytest.raises(subprocess.CalledProcessError):full_run.run(args)
    args.resume=True
    report=full_run.run(args)
    train_calls=[a for a in calls if 'agri_vla.candidate_train' in a]
    assert [a[a.index('--steps')+1] for a in train_calls]==['100','1000','2000']
    assert report['status']=='completed'


def test_workflow_stops_before_next_phase_if_evaluation_code_changes(tmp_path, monkeypatch):
    args, calls, _ = workflow_fixture(tmp_path, monkeypatch)
    original_run, original_sha = full_run.subprocess.run, full_run.file_sha
    changed = [False]
    def execute(argv, **kwargs):
        result = original_run(argv, **kwargs)
        changed[0] = True
        return result
    monkeypatch.setattr(full_run.subprocess, 'run', execute)
    monkeypatch.setattr(full_run, 'file_sha', lambda p: 'changed' if changed[0] and Path(p).name == 'evaluation.py' else original_sha(p))
    with pytest.raises(DatasetError, match='code changed during'):
        full_run.run(args)
    assert len(calls) == 1 and 'agri_vla.candidate_train' in calls[0]
