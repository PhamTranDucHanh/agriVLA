"""Forward-only CPU scope/reload checks; GPU checks require trained native states."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys

from .dataset import require,read_json
from .preflight import project_root
from .training import file_sha,load_training_config,write_json,CAMERAS
from .late_vlm import initialize,load_policy,save_fixture,recipe,resume_contract,KV,audit_parameters,metadata,update_lock


def observation(policy):
    from .evaluation import data_loader
    c=load_training_config(project_root()/'configs/training_v0.yaml')
    loader,mapping=data_loader(c,policy.config,'train',indices=[0])
    batch=next(iter(loader));return batch,mapping[0]['source_record']


def cpu(output):
    import torch
    from .evaluation import predict_batch
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    policy,pre,post,control_audit=initialize('control')
    batch,raw=observation(policy)
    reference=predict_batch(policy,pre,post,batch,seed=42)
    del policy,pre,post;gc.collect()
    policy,pre,post,candidate_audit=initialize('candidate')
    calls={path:0 for path in KV};handles=[]
    modules=dict(policy.get_base_model().named_modules())
    def make_hook(path):
        def hook(*_args):calls[path]+=1
        return hook
    for path in KV:handles.append(modules[path].register_forward_hook(make_hook(path)))
    poison=dict(batch,teacher_annotations={'wilt_scores':[4,4,4]},intent='STOP_ALL',target_action=[0,0,0],wilt_notes='not input')
    zero=predict_batch(policy,pre,post,poison,seed=42)
    require(zero['continuous_action']==reference['continuous_action'],'Zero-B expansion changed native output')
    require(all(v>0 for v in calls.values()),'New K/V modules not called')
    for handle in handles:handle.remove()
    sensitivities={}
    for path in KV:
        module=modules[path];weight=module.lora_B['default'].weight
        with torch.no_grad():weight.copy_(torch.linspace(-.2,.2,weight.numel()).reshape_as(weight))
        altered=predict_batch(policy,pre,post,batch,seed=42)
        with torch.no_grad():weight.zero_()
        difference=max(abs(a-b) for a,b in zip(altered['continuous_action'],zero['continuous_action']))
        require(difference>0,'K/V perturbation did not affect action')
        sensitivities[path]=difference
    restored=predict_batch(policy,pre,post,batch,seed=42)
    require(restored['continuous_action']==reference['continuous_action'],'Probe failed to restore zeroB')
    save_fixture(policy,output/'fixture','candidate')
    write_json(output/'reference.json',{'prediction':zero,'sample_id':raw['sample_id'],'target_for_review_only':raw['target_action']})
    write_json(output/'cpu-scope-report.json',{'status':'pass','control':control_audit,'candidate':candidate_audit,
               'same_initial_function_max_error':0.,'native_module_calls':calls,
               'forward_only_in_memory_B_perturbation_action_differences':sensitivities,
               'poison_metadata_does_not_change_prediction':True,
               'optimizer_updates_performed':0,'backward_performed':False,'GPU_fit_verified':False,
               'fixture_is_trained_resume_proof':False})
    del modules,module,weight,policy,pre,post;gc.collect()
    env=os.environ.copy();env['HF_DATASETS_CACHE']=str(project_root()/'.cache/hf-datasets')
    subprocess.run([sys.executable,'-m','agri_vla.late_vlm_checks','reload','--checkpoint',str(output/'fixture'),'--output',str(output/'fresh-reload.json'),'--reference',str(output/'reference.json')],check=True,env=env)
    print('CPU scope and fresh-process reload PASS; no training/backward/GPU fit proof.')


def reload_cpu(checkpoint,reference,output):
    from .evaluation import predict_batch
    policy,pre,post=load_policy(checkpoint,device='cpu');batch,_=observation(policy)
    predicted=predict_batch(policy,pre,post,batch,seed=42);expected=read_json(reference)['prediction']
    difference=max(abs(a-b) for a,b in zip(predicted['continuous_action'],expected['continuous_action']))
    require(difference<=1e-5 and predicted['quantized_action']==expected['quantized_action'],'Fresh CPU reload differs')
    write_json(output,{'status':'pass','max_absolute_error':difference,'atol':1e-5,'quantized_equal':True,'prediction':predicted,'optimizer_updates':0,'fixture_resume_proof':False})


def gpu_check(mode,checkpoint,output,control=None):
    folder=Path(checkpoint).resolve();cp=folder.parent if folder.name=='pretrained_model' else folder
    if (folder/'late_vlm.json').exists():
        require(read_json(folder/'late_vlm.json')['kind']=='native_trained_checkpoint','CPU scope fixture cannot prove GPU memory or trained resume')
    value=read_json(cp/'pretrained_model/late_vlm.json');arm=value['arm'];resume_contract(cp,arm)
    if mode=='memory':
        from . import small_memory as engine
        saved=engine.load_checkpoint,engine.parameter_audit
        try:
            engine.load_checkpoint=lambda c,path:load_policy(path,device='cuda')
            engine.parameter_audit=lambda policy:audit_parameters(policy,arm)
            return engine.run(cp/'pretrained_model',output)
        finally:engine.load_checkpoint,engine.parameter_audit=saved
    require(control is not None,'Adjacent actual trained control required')
    from . import small_resume as engine
    saved={key:getattr(engine,key) for key in ('load_checkpoint','parameter_audit','attach_checkpoint_contract','write_json')}
    original_write=saved['write_json']
    def attach(bundle,folder):
        saved['attach_checkpoint_contract'](bundle,folder)
        step=int(Path(folder).parent.name)
        original_write(Path(folder)/'late_vlm.json',metadata(arm,step=step,kind='native_trained_checkpoint'))
    def write(path,val):
        original_write(path,val)
        if Path(path).name=='small_experiment.json':update_lock(Path(path).parent)
    try:
        engine.load_checkpoint=lambda c,path:load_policy(path,device='cuda')
        engine.parameter_audit=lambda policy:audit_parameters(policy,arm)
        engine.attach_checkpoint_contract=attach;engine.write_json=write
        return engine.run(cp,control,output,restore_loader_continuation=True)
    finally:
        for key,val in saved.items():setattr(engine,key,val)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('cpu','reload','memory','resume'))
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--reference',type=Path);parser.add_argument('--control',type=Path)
    args=parser.parse_args()
    if args.mode=='cpu':cpu(args.output)
    elif args.mode=='reload':
        require(args.checkpoint is not None and args.reference is not None,'Provide fixture/reference')
        reload_cpu(args.checkpoint,args.reference,args.output)
    else:
        require(args.checkpoint is not None,'Provide real trained checkpoint')
        gpu_check(args.mode,args.checkpoint,args.output,args.control)


if __name__=='__main__':main()
