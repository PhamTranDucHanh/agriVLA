"""Native hook wiring on CPU, without model construction, backward or optimizer."""
import pytest

pytest.importorskip('torch')
pytest.importorskip('lerobot')


def test_scoped_hooks_restore_even_on_exception():
    from lerobot.scripts import lerobot_train as upstream
    from agri_vla import small_train
    from agri_vla.late_vlm_experiment import scoped_engine
    from agri_vla.preflight import project_root
    names=('load_experiment','expert_target_modules','parameter_audit','load_checkpoint','write_json')
    before={name:getattr(small_train,name) for name in names}
    factory,cycle,update=upstream.make_policy,upstream.cycle,upstream.update_policy
    with pytest.raises(RuntimeError):
        with scoped_engine('candidate',project_root()/'outputs/small-cases/unit-candidate-unused'):
            assert upstream.make_policy is not factory
            assert upstream.cycle is not cycle
            assert small_train.load_experiment(project_root()/'configs/late_vlm_kv_v1.json')['profiles']['candidate']['late_vlm_arm']=='candidate'
            raise RuntimeError('test restoration')
    assert all(getattr(small_train,name) is value for name,value in before.items())
    assert (upstream.make_policy,upstream.cycle,upstream.update_policy)==(factory,cycle,update)


def test_resume_does_not_install_fresh_rng_barrier():
    from lerobot.scripts import lerobot_train as upstream
    from agri_vla.late_vlm_experiment import scoped_engine
    from agri_vla.preflight import project_root
    cycle=upstream.cycle
    with scoped_engine('candidate',project_root()/'outputs/small-cases/unit-candidate-unused',resuming=True):
        assert upstream.cycle is cycle


def test_raw_input_camera_order_and_allowlist(tmp_path):
    import torch
    from PIL import Image
    from agri_vla.late_vlm_inputs import raw_batch
    for name,color in [('r',(255,0,0)),('g',(0,255,0)),('b',(0,0,255))]:
        Image.new('RGB',(256,256),color).save(tmp_path/(name+'.png'))
    batch=raw_batch({'zone3':tmp_path/'b.png','zone1':tmp_path/'r.png','zone2':tmp_path/'g.png'},'Tắt hết.')
    for i in (1,2,3):
        image=batch[f'observation.images.camera{i}']
        assert image.shape==(1,1,3,256,256) and image[0,0,i-1].eq(1).all()
        assert image.sum().item()==256*256
    assert torch.equal(batch['observation.state'],torch.zeros(1,1,3))
    assert set(batch)=={'observation.images.camera1','observation.images.camera2','observation.images.camera3','observation.state','task'}
