"""Policy semantics, TRAIN isolation, grouping and bounded source sampling."""
from copy import deepcopy
import itertools

import pytest

from agri_vla.counterfactual import (REQUESTS, independent_action, model_inputs,
                                     build_view, sampling_plan, request_key)
from agri_vla.revision import POLICY, revision_action


def fixture_rows():
    rows=[]
    for scene in range(4):
        requests=REQUESTS if scene==0 else [('AUTO',{})]
        for intent,params in requests:
            for family in range(2):
                pair=f'{scene}-{request_key(intent,params)}-{family}'
                for lang in ('vi','en'):
                    rows.append({'sample_id':pair+'-'+lang,'scene_id':str(scene),
                                 'images':{f'zone{i}':f'{scene}-{i}.jpg' for i in (1,2,3)},
                                 'source_image_ids':[f'{scene}-{i}.jpg' for i in (1,2,3)],
                                 'source_groups':['g0' if scene<3 else 'g1']*3,
                                 'teacher_annotations':{'wilt_scores':[0,2,4]},
                                 'split':'train','wording_partition':'seen','policy_version':POLICY,
                                 'translation_pair_id':pair,'template_family_id':f'{intent}-{family}',
                                 'instruction':f'{intent} {params} {family} {lang}',
                                 'language':lang,'intent':intent,'intent_params':params,
                                 'target_action':independent_action([0,2,4],intent,params)})
    return rows


def test_independent_arithmetic_all_125_wilt_triplets_and_14_requests():
    for wilt in itertools.product(range(5),repeat=3):
        for intent,params in REQUESTS:
            assert independent_action(wilt,intent,params)==revision_action(wilt,intent,params)
    assert independent_action([0,0,0],'WATER_ALL',{'level':3})==[3,3,3]
    assert independent_action([4,4,1],'WORST_ONLY',{})==[3,3,0]


def test_view_retains_originals_and_reuses_only_train_text_and_images():
    original=fixture_rows();snapshot=deepcopy(original)
    view,scenes=build_view(original)
    assert original==snapshot
    assert {r['sample_id'] for r in original}=={r['sample_id'] for r in view if r['origin']=='original'}
    assert {r['instruction'] for r in view}<={r['instruction'] for r in original}
    for sid in scenes:
        rr=[r for r in view if r['scene_id']==sid]
        assert {'STOP_ALL','WATER_ALL'}<={r['intent'] for r in rr}
        assert len({tuple(r['target_action']) for r in rr})>=2
        assert all(r['images']==scenes[sid]['images'] for r in rr)
    assert build_view(original)==(view,scenes)


@pytest.mark.parametrize('field,value',[('split','val'),('split','test'),('wording_partition','unseen'),('policy_version','wilt_proxy_policy_v0')])
def test_reject_nontrain_or_wrong_policy(field,value):
    rows=fixture_rows();rows[0][field]=value
    with pytest.raises(ValueError):build_view(rows)


def test_reject_translation_semantics_and_inconsistent_scene():
    rows=fixture_rows();rows[0]['images']['zone1']='other.jpg'
    with pytest.raises(ValueError):build_view(rows)
    rows=fixture_rows();rows[0]['target_action']=[3,3,3]
    with pytest.raises(ValueError):build_view(rows)


def test_sampling_is_reproducible_covers_every_source_and_caps_scene_visits():
    view,_=build_view(fixture_rows())
    plan,stats=sampling_plan(view,8,3,42)
    assert (plan,stats)==sampling_plan(view,8,3,42)
    control,_=sampling_plan([r for r in view if r['origin']=='original'],8,3,42)
    assert [(r['scene_id'],r['anchor_image']) for r in plan]==[(r['scene_id'],r['anchor_image']) for r in control]
    assert stats['covered_scenes']==4 and stats['covered_images']==12
    assert max(stats['scene_visit_histogram'])<=3
    mapping={r['sample_id']:r for r in view}
    for visit in plan:
        rr=[mapping[s] for s in visit['sample_ids']]
        assert [r['language'] for r in rr]==['vi','en']
        assert rr[0]['target_action']==rr[1]['target_action']
    with pytest.raises(ValueError):sampling_plan(view,13,3,42)
    with pytest.raises(ValueError):sampling_plan(view,8,3,42,max_image_visits=1)
    limited,limited_stats=sampling_plan(view,8,3,42,max_image_visits=3)
    assert limited_stats['image_camera_exposure_min_max'][1]<=3


def test_inputs_contain_no_teacher_target_or_structured_intent():
    row=fixture_rows()[0];row.update(intent='STOP_ALL',target_action=[9,9,9],teacher='secret')
    inputs=model_inputs(row)
    assert set(inputs)=={'observation.images.camera1','observation.images.camera2',
                         'observation.images.camera3','observation.state','task'}
    assert inputs['task']==row['instruction'] and inputs['observation.state']==[0,0,0]
