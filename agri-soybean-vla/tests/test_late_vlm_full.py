from types import SimpleNamespace
import pytest
from agri_vla.late_vlm_full import describe,run


def test_full_plan_has_original_dataset_and_no_execution():
    p=describe(6000)
    assert p['train_records']==3600 and p['val_records']==720 and p['epochs']==pytest.approx(6.6666666667)
    assert p['counterfactual_used'] is False and p['GPU_full_workflow_executed'] is False
    assert '/wording_v1/full/train' in p['train_root']


def test_missing_qualification_blocks_before_gpu_or_training(tmp_path):
    with pytest.raises(ValueError,match='Full run blocked'):
        run(SimpleNamespace(execute=True,recipe=tmp_path/'missing.json',steps=6000))


def test_full_bridge_routes_native_commands_and_deploy_script(monkeypatch,tmp_path):
    pytest.importorskip('torch')
    import json
    from pathlib import Path
    from agri_vla import full_run,late_vlm_full
    from agri_vla.training import file_sha
    calls=[]
    original=SimpleNamespace(run=lambda argv,*a,**kw:calls.append(argv),Popen=lambda argv,*a,**kw:calls.append(argv))
    monkeypatch.setattr(full_run,'subprocess',original)
    model=tmp_path/'final_model';model.mkdir()
    def fake_workflow(args):
        full_run.subprocess.run(['python','-m','agri_vla.candidate_train','--steps','100'])
        full_run.subprocess.Popen(['python','-u','-m','agri_vla.cli','demo-web','--checkpoint','X'])
        return {'deploy_model':str(model)}
    monkeypatch.setattr(full_run,'run',fake_workflow)
    decision=tmp_path/'decision.json';decision.write_text(json.dumps({'relative_improvement':{'passed':True}}))
    recipe=tmp_path/'recipe.json';recipe.write_text(json.dumps({'bridge':'late_vlm_full_bridge_v1','bridge_sha256':file_sha(Path(late_vlm_full.__file__)),
        'relative_decision':str(decision),'relative_decision_sha256':file_sha(decision)}))
    run(SimpleNamespace(execute=True,recipe=recipe,steps=6000,final_test=True))
    assert calls[0][2:4]==['agri_vla.late_vlm_full','phase']
    assert calls[1][3:5]==['agri_vla.late_vlm_full','web']
    assert 'agri_vla.late_vlm_full web' in (tmp_path/'deploy.sh').read_text()
    assert full_run.subprocess is original


def test_web_uses_the_loader_called_by_demo_runtime(monkeypatch,tmp_path):
    pytest.importorskip('torch')
    import json
    from agri_vla import evaluation,demo_server,late_vlm_full
    original=evaluation.load_model
    cp=tmp_path/'model';cp.mkdir();(cp/'late_vlm.json').write_text(json.dumps({'arm':'candidate'}))
    policy=object()
    monkeypatch.setattr(late_vlm_full,'load_policy',lambda checkpoint,device:(policy,'pre','post'))
    monkeypatch.setattr(late_vlm_full,'audit_parameters',lambda p,arm:{'trainable':1777280})
    def fake_serve(**kwargs):
        assert evaluation.load_model({},None,checkpoint=cp,device='cpu')==(policy,'pre','post',{'trainable':1777280})
        return 'served'
    monkeypatch.setattr(demo_server,'serve_demo',fake_serve)
    assert late_vlm_full.web(SimpleNamespace(checkpoint=cp,device='cpu',port=8000))=='served'
    assert evaluation.load_model is original


def test_qualification_refuses_a_failed_relative_comparison(tmp_path):
    import json
    from agri_vla.late_vlm_full import qualify
    decision=tmp_path/'decision.json';decision.write_text(json.dumps({'relative_improvement':{'passed':False}}))
    with pytest.raises(ValueError,match='Relative K/V improvement'):
        qualify(SimpleNamespace(checkpoint=tmp_path/'model',decision=decision))
