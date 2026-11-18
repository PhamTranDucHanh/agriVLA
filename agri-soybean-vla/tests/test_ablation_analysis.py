from copy import deepcopy

from agri_vla.ablation_analysis import select_initialization, conditional_scene_interval


CRITERIA={'minimum_mean_validation_exact_margin':.05,'minimum_mean_validation_zone_margin':.02,
          'minimum_seeds_with_positive_validation_exact_margin':4,'train280_exact_max_drop':.02,'old60_exact_max_drop':.10}


def arm(exact):
    return {'validation_mean':{'exact':exact,'zone':exact,'STOP':1.,'false_watering_rate':.1},
            'validation_by_seed':{str(i):{'exact':exact} for i in (0,1,2,3,42)},
            'train42':{'exact':.8},'old60_42':{'exact':.9},'new220_42':{'exact':.7}}


def test_small_gain_or_STOP_regression_does_not_select():
    assert not select_initialization(arm(.42),arm(.4),CRITERIA,{'lower':.01})['passed']
    a=arm(.6);a['validation_mean']['STOP']=.9
    assert not select_initialization(a,arm(.4),CRITERIA,{'lower':.01})['passed']
    assert select_initialization(arm(.6),arm(.4),CRITERIA,{'lower':.01})['passed']


def test_old60_and_new220_guards_are_independent():
    a=arm(.6);a['old60_42']['exact']=.5
    assert not select_initialization(a,arm(.4),CRITERIA,{'lower':.01})['passed']
    a=arm(.6);a['new220_42']['exact']=.6
    assert not select_initialization(a,arm(.4),CRITERIA,{'lower':.01})['passed']


def test_bootstrap_keeps_translations_and_seeds_in_the_same_scene():
    a=[{'sample_id':s,'seed':seed,'target_action':[0,0,0],'quantized_action':[0,0,0]} for s in ('vi','en') for seed in (0,42)]
    b=deepcopy(a)
    for r in b:r['quantized_action']=[1,0,0]
    interval=conditional_scene_interval(a,b,{s:{'scene_id':'same'} for s in ('vi','en')},draws=100)
    assert interval['clusters']==1 and interval['lower']==interval['upper']==1.
