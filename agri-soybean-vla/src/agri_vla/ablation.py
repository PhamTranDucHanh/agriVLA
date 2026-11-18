"""Freeze a bounded initialization experiment; audit historical B and frozen inference."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import time

from .dataset import require, read_json, read_jsonl
from .preflight import project_root
from .training import file_sha, load_training_config, write_json

NAME='initialization-ablation-v1-20261116'


def freeze():
    root=project_root();out=root/'reports'/NAME;out.mkdir(exist_ok=False)
    config=root/'configs/initialization_ablation_v1.json'
    spec=read_json(config)
    c=load_training_config(root/'configs/training_v0.yaml')
    from .small_cases import selection
    from .evaluation import fixed_validation_selection
    panel=selection(c,'cases280');validation=fixed_validation_selection(c,112)
    write_json(out/'panel.json',panel);write_json(out/'validation-panel.json',validation)
    tracked=[config,root/'configs/small_cases_v9.yaml',root/'configs/training_v0.yaml',
             root/'configs/readiness_balanced_v1.json',root/'data/manifests/wording_v1/dataset_lock.json',
             root/'data/manifests/wording_v1/train.jsonl',root/'data/lerobot/wording_v1/full/train_stats.json',
             root/'models/prepared/wording_v1/bundle_lock.json',root/'reports/counterfactual-train-v1/artifact-lock.json']
    tracked += [root/f'src/agri_vla/{n}.py' for n in ('small_train','small_cases','small_resume','evaluation','inference_sampling','augmentation','counterfactual','ablation')]
    baseline={p.name:sum(f.stat().st_size for f in p.rglob('*') if f.is_file()) for p in [root/'outputs',root/'reports',root/'.cache/hf-datasets']}
    warm=root/'outputs/small-cases/cases60-endpoint-resume300-20261116/checkpoints/000600/pretrained_model'
    old_reports=[root/'reports/small-cases'/n for n in ('pilot280-warm-20261116','pilot280-resume100-20261116')]
    a,b=[read_json(p/'experiment.json') for p in old_reports]
    require(a['episodes']==b['episodes']==panel['episodes'],'Historical B panel mismatch')
    require(a['profile']==b['profile'] and a['profile_name']==spec['profile'],'Historical B profile mismatch')
    require(a['initialization']['optimizer_reset'] is True and a['initialization']['adapter_sha256']==file_sha(warm/'adapter_model.safetensors'),'Historical B initialization mismatch')
    require(b['initialization']['start_step']==100 and b['initialization']['optimizer_reset'] is False,'Wrong B continuation')
    require(all(v.get('exact',False) if isinstance(v,dict) else v for v in read_json(old_reports[1]/'restored-state.json').values()),'Historical B restore failed')
    updates=[r for p in old_reports for r in read_jsonl(p/'updates.jsonl')]
    require([u['step'] for u in updates]==list(range(1,301)),'Historical B steps incomplete')
    require(all(math.isfinite(u['loss']) and math.isfinite(u['grad_norm']) for u in updates),'Historical B nonfinite')
    cp=root/spec['historical_B_candidate']
    for name,sha in read_json(cp/'agri_checkpoint_lock.json')['files'].items():require(file_sha(cp/name)==sha,'Historical B checksum mismatch')
    write_json(out/'historical-B-audit.json',{'status':'conditionally_reusable_pending_A_order_RNG_comparison','conditions_static_passed':True,
               'checkpoint':str(cp),'adapter_sha256':file_sha(cp/'adapter_model.safetensors'),
               'source_reports':[str(p) for p in old_reports], 'updates':300,
               'runner_difference':'current wrapper adds telemetry and a no-op augmentation hook for this profile; upstream loop remains pinned',
               'pending':['all300 A episode batches match B','recorded A RNG fingerprints match B'],
               'source_file_hashes':{str(p.relative_to(root)):file_sha(p) for folder in old_reports for p in folder.iterdir() if p.is_file()}})
    write_json(out/'historical-B-updates.json',updates)
    write_json(out/'protocol-lock.json',{'version':spec['version'],'frozen_utc_seconds':time.time(),
               'protocol':spec,'files':{str(p.relative_to(root)):file_sha(p) for p in tracked},
               'baseline_bytes':baseline,'warm_source_adapter_sha256':file_sha(warm/'adapter_model.safetensors'),
               'new_run_registry':[], 'A_output':str(root/'outputs/small-cases'/f'{NAME}-A'),
               'note':'B results were already known before protocol definition; no A results used to define selection thresholds'})
    print('Protocol frozen at',out,flush=True)


def assert_lock():
    root=project_root();lock=read_json(root/'reports'/NAME/'protocol-lock.json')
    for path,sha in lock['files'].items():require(file_sha(root/path)==sha,f'Protocol code/config drift: {path}')
    return lock


def train_a():
    root=project_root();lock=assert_lock();report=root/'reports'/NAME
    write_json(report/'new-run-1-started.json',{'arm':'A','maximum_optimizer_updates':300,'time':time.time()})
    from .small_train import run
    result=run('cases280','r8_wide_b4_n1_t1_lr1e4',Path(lock['A_output']),steps=300,
               experiment_config=root/'configs/small_cases_v9.yaml',evaluate_stage='cases60')
    require(result['status']!='error',f'Run A stopped: {result["failure"]}')
    reference=read_json(report/'historical-B-updates.json')
    actual=read_jsonl(root/'reports/small-cases'/f'{NAME}-A'/'updates.jsonl')
    same_order=len(actual)==len(reference) and all(a['episodes']==b['episodes'] for a,b in zip(actual,reference))
    rng_comparisons=[{'step':a['step'],'equal':a['rng_before']==b['rng_before']} for a,b in zip(actual,reference) if a.get('rng_before') is not None and b.get('rng_before') is not None]
    write_json(report/'A-B-fairness.json',{'all300_episode_batches_equal':same_order,'recorded_RNG':rng_comparisons,
               'historical_B_reusable':same_order and bool(rng_comparisons) and all(r['equal'] for r in rng_comparisons),
               'caution':'only initialization changes; no training-seed replication'})
    require(same_order,'A/B episode order mismatch')


def evaluate(checkpoint, output, seeds):
    """Native frozen prediction; TRAIN280 and original fixed VAL112, no augmentation/TEST."""
    root=project_root();assert_lock();out=Path(output).resolve();require(not out.exists(),'Output already exists')
    c=load_training_config(root/'configs/training_v0.yaml')
    from .small_train import load_checkpoint,panel_evaluate
    from .small_cases import selection
    from .evaluation import data_loader,predict_batch,metrics,fixed_validation_selection
    from lerobot.datasets.factory import make_train_eval_datasets
    from .training import train_config
    import torch
    cp=Path(checkpoint).resolve();before=file_sha(cp/'adapter_model.safetensors')
    policy,pre,post=load_checkpoint(c,cp)
    selected=selection(c,'cases280')
    cfg=train_config(c,root/c['bundle'],output=root/'outputs/unused-ablation-eval',device='cuda',episodes=selected['episodes'])
    ds,ev=make_train_eval_datasets(cfg);require(ev is None and len(ds)==280,'Wrong evaluation TRAIN')
    start=time.perf_counter();train=panel_evaluate(policy,pre,post,ds,selected,seeds,batch_size=4)
    validation=fixed_validation_selection(c,112)
    loader,mapping=data_loader(c,policy.config,'val',indices=validation['episodes']);rows=[]
    for seed in seeds:
        for batch in loader:
            raw=mapping[int(batch['index'].item())]['source_record']
            require(batch['task']==[raw['instruction']] and batch['action'].reshape(-1).tolist()==raw['target_action'],'VAL mapping drift')
            rows.append({k:raw[k] for k in ('sample_id','language','intent','wording_partition','target_action')}|{'seed':seed}|predict_batch(policy,pre,post,batch,seed=seed))
        print('VAL',cp.parent.name,seed,metrics([r for r in rows if r['seed']==seed])['overall'],flush=True)
    require(file_sha(cp/'adapter_model.safetensors')==before,'Frozen checkpoint changed')
    write_json(out,{'checkpoint':str(cp),'adapter_sha256':before,'seeds':seeds,'train':train,
                   'validation_selection':validation,'val_predictions':rows,
                   'val_by_seed':{str(seed):metrics([r for r in rows if r['seed']==seed]) for seed in seeds},
                   'inference_seconds':time.perf_counter()-start,'optimizer_updates':0,'test_opened':False})


def main():
    os.environ['HF_DATASETS_CACHE']=str(project_root()/'.cache/hf-datasets')
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=('freeze','train-a','evaluate'))
    parser.add_argument('--checkpoint',type=Path);parser.add_argument('--output',type=Path)
    parser.add_argument('--seeds',type=int,nargs='+',default=[0,1,2,3,42]);args=parser.parse_args()
    if args.command=='freeze':freeze()
    elif args.command=='train-a':train_a()
    else:
        require(args.checkpoint is not None and args.output is not None,'Provide checkpoint/output')
        require(args.seeds in ([42],[0,1,2,3,42]),'Noise seed protocol differs')
        evaluate(args.checkpoint,args.output,args.seeds)


if __name__=='__main__':main()
