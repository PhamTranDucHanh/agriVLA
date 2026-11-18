"""Bounded native experiment. Default only prints plan; --execute is explicit."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import random
import time

from .dataset import require,read_json
from .preflight import project_root
from .training import file_sha,write_json
from .late_vlm import (recipe,target_regex,initialize,audit_parameters,load_policy,
                       metadata,update_lock,resume_contract,VERSION,KV)


def plan(arm,steps,output,resume=None):
    r=recipe();require(arm in ('control','candidate'),'Unknown arm')
    require(type(steps) is int and steps in (2,100,300),'Allowed budgets are2,100,300updates')
    root=project_root();output=Path(output).resolve()
    require(output.is_relative_to(root/'outputs/small-cases') and not output.exists(),'Use a fresh outputs/small-cases directory')
    require(arm in output.name,'Name output with arm for clear provenance')
    if resume:
        old=resume_contract(resume,arm);require(old['new_optimizer_updates']<steps,'Resume must increase step')
    return {'mode':'plan_only','version':VERSION,'arm':arm,'target_modules':target_regex(arm),
            'trainable_parameters':r['trainable_parameters'][arm],'final_step':steps,
            'stage':'cases280','source':r['source'],'output':str(output),
            'resume':str(resume) if resume else None,'GPU_backward_verified':False,'training_executed':False,
            'limits':{'max_updates':300,'new_artifact_bytes':r['max_new_artifact_bytes'],'TEST':False,'full_train':False}}


@contextmanager
def scoped_engine(arm,output,*,resuming=False):
    """Adapt project hooks, keep vendor loop/update/optimizer/save APIs unchanged."""
    import torch
    import numpy as np
    from lerobot.scripts import lerobot_train as upstream
    from . import small_train as engine
    from .late_vlm import PREFIX
    root=project_root();r=recipe();cfg_path=root/'configs/late_vlm_kv_v1.json'
    saved={name:getattr(engine,name) for name in ('load_experiment','expert_target_modules','parameter_audit','load_checkpoint','write_json')}
    original_factory,original_cycle,original_update=upstream.make_policy,upstream.cycle,upstream.update_policy
    original_write=saved['write_json'];profile={'rank':8,'alpha':8,'lr':1e-4,'full_projections':False,'batch_size':4,'num_steps':1,'time_sampling':'endpoint_t1','expert_targets':'all_linear','late_vlm_arm':arm}
    spec={'version':VERSION,'seed':42,'screen_seeds':[0,42],'certificate_seeds':[0,1,2,3,42],
          'warmup_steps':20,'max_steps':300,'evaluate_at':[100,300],
          'profiles':{arm:profile},'gates':{stage:{'exact':1.,'zone':1.,'stop':1.} for stage in ('micro8','cases32','cases60','cases280')},
          '_sha256':file_sha(cfg_path),'_path':str(cfg_path)}
    dataset_cache=root/'.cache/hf-datasets'
    baseline_cache=sum(p.stat().st_size for p in dataset_cache.rglob('*') if p.is_file())
    output=Path(output);report=root/'reports/small-cases'/output.name
    calls={'step':0,'KV_gradient_checks':[]}
    def spec_loader(path):
        require(Path(path).resolve()==cfg_path,'Wrong recipe path');return spec
    def make_policy(*args,**kwargs):
        policy,_,_,audit=initialize(arm,device='cuda')
        original_write(report/'scope-initialization.json',audit)
        return policy
    def cycle(loader):
        # One shared barrier AFTER policy+optimizer construction for both fresh arms.
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        yield from original_cycle(loader)
    def typed_write(path,value):
        path=Path(path)
        if path.name in ('experiment.json','small_experiment.json'):
            if not resuming:
                value['initialization']={'weights_only_curriculum':str(root/r['source']),'source_adapter_sha256':r['source_adapter_sha256'],'optimizer_reset':True}
            value['fixed_warm_reference']=r['source_adapter_sha256']
            value['late_vlm_variant']=VERSION
        original_write(path,value)
        if path.name=='small_experiment.json':
            step=int(path.parent.parent.name)
            original_write(path.parent/'late_vlm.json',metadata(arm,step=step,kind='native_trained_checkpoint'))
            update_lock(path.parent)
    def update(*args,**kwargs):
        policy=args[1] if len(args)>1 else kwargs['policy'];optimizer=args[3] if len(args)>3 else kwargs['optimizer']
        opt=getattr(optimizer,'optimizer',optimizer);hook=None
        owned=sum(p.stat().st_size for folder in (output,report) if folder.exists() for p in folder.rglob('*') if p.is_file())
        cache_growth=max(0,sum(p.stat().st_size for p in dataset_cache.rglob('*') if p.is_file())-baseline_cache)
        require(owned+cache_growth<r['max_new_artifact_bytes'],'Artifact budget exceeded')
        if arm=='candidate':
            def grad_check(_opt,_args,_kwargs):
                found={}
                for name,p in policy.named_parameters():
                    if any(k+'.lora_B.' in name for k in KV):
                        require(p.grad is not None and bool(torch.isfinite(p.grad).all()),'Missing/nonfinite lastKV.B gradient')
                        found[name]=float(p.grad.float().square().sum())
                require(len(found)==2 and all(v>0 for v in found.values()),'LastKV.B has no action-loss gradient')
                calls['KV_gradient_checks'].append({'update':calls['step']+1,'squared_norms':found})
            hook=opt.register_step_pre_hook(grad_check)
        try:
            result=original_update(*args,**kwargs)
        finally:
            if hook is not None:hook.remove()
        calls['step']+=1
        return result
    try:
        engine.load_experiment=spec_loader
        engine.expert_target_modules=lambda profile:target_regex(arm)
        engine.parameter_audit=lambda policy:audit_parameters(policy,arm)
        engine.load_checkpoint=lambda c,path:load_policy(path,device='cuda')
        engine.write_json=typed_write
        upstream.make_policy=make_policy;upstream.update_policy=update
        if not resuming:upstream.cycle=cycle
        yield calls
    finally:
        for name,value in saved.items():setattr(engine,name,value)
        upstream.make_policy=original_factory;upstream.cycle=original_cycle;upstream.update_policy=original_update


def execute(arm,steps,output,resume=None):
    resolved=plan(arm,steps,output,resume)
    from .small_train import run
    root=project_root();before=time.time()
    with scoped_engine(arm,output,resuming=resume is not None) as probes:
        result=run('cases280',arm,Path(output),steps=steps,resume_checkpoint=resume,
                   experiment_config=root/'configs/late_vlm_kv_v1.json',evaluate_stage='cases60')
    report=root/'reports/small-cases'/Path(output).name
    write_json(report/'late-vlm-runtime.json',{'version':VERSION,'arm':arm,'wall_seconds':time.time()-before,
               'native_updates':result['optimizer_steps_performed'],'KV_gradient_checks':probes['KV_gradient_checks'],
               'trained_resume_proof':False,'note':'requires external interrupted-vs-continuous proof; this receipt alone does not certify resume'})
    require(result['status']!='error',result['failure'])
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--arm',choices=('control','candidate'),required=True)
    parser.add_argument('--steps',type=int,default=2);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--resume-checkpoint',type=Path);parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    result=execute(args.arm,args.steps,args.output,args.resume_checkpoint) if args.execute else plan(args.arm,args.steps,args.output,args.resume_checkpoint)
    print(json.dumps({k:v for k,v in result.items() if k in ('mode','version','arm','trainable_parameters','final_step','limits','GPU_backward_verified','status','step')},indent=2))


if __name__=='__main__':main()
