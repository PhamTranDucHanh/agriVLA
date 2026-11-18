"""Typed qualified workflow and explicit research phases that never claim readiness."""
import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import subprocess
import shlex

from .dataset import require,read_json,DatasetError
from .preflight import project_root
from .training import file_sha,load_training_config,write_json
from .late_vlm import recipe,metadata,update_lock,load_policy,audit_parameters


def describe(steps=6000):
    root=project_root();r=recipe();require(100<=steps<=6000,'Full projection budget100..6000')
    return {'version':'late_vlm_full_bridge_v1','mode':'plan_only','train_root':str(root/'data/lerobot/wording_v1/full/train'),
            'train_records':3600,'val_root':str(root/'data/lerobot/wording_v1/full/val'),'val_records':720,
            'inputs':'three RGB256 cameras + instruction/tokenizer96 + zero state3',
            'targets':'three original integer action levels0..3; full TRAIN normalization unchanged',
            'counterfactual_used':False,'steps':steps,'batch_size':4,'epochs':steps*4/3600,
            'init':'qualified small-checkpoint adapter weights only; reset optimizer for full TRAIN',
            'outputs':['native checkpoints/pretrained_model/adapter_model.safetensors',
                       'native checkpoints/training_state optimizer/scheduler/RNG/step/AMP',
                       'VAL720 metrics and predictions at milestones','best+latest checkpoint retention',
                       'VAL-selected final_model/ and deploy.sh; TEST only with explicit --final-test'],
            'recipe_required':str(root/'configs/late_vlm_full_ready.json'),
            'execution_ready':(root/'configs/late_vlm_full_ready.json').is_file(),
            'readiness':'existing capacity+VAL+memory+trained-resume gates, unchanged',
            'GPU_full_workflow_executed':False}


@contextmanager
def full_scope(arm,output,resuming=False):
    from . import candidate_train as engine
    from . import small_train
    from .late_vlm_experiment import scoped_engine
    saved={name:getattr(engine,name) for name in ('load_experiment','expert_target_modules','attach_checkpoint_contract')}
    with scoped_engine(arm,output,resuming=resuming) as probes:
        def attach(bundle,folder):
            saved['attach_checkpoint_contract'](bundle,folder)
            folder=Path(folder);step=int(folder.parent.name)
            value=metadata(arm,step=step,kind='native_trained_checkpoint')
            value['full_bridge_sha256']=file_sha(Path(__file__))
            write_json(folder/'late_vlm.json',value)
            full=read_json(folder/'agri_full_candidate.json')
            write_json(folder/'small_experiment.json',{'version':'late_vlm_kv_v1','profile':full['profile'],
                       'stage':'full3600','full_bridge':True,'arm':arm})
            update_lock(folder)
        try:
            engine.load_experiment=small_train.load_experiment
            engine.expert_target_modules=small_train.expert_target_modules
            engine.attach_checkpoint_contract=attach
            yield
        finally:
            for name,value in saved.items():setattr(engine,name,value)
            if probes['step']:
                report=project_root()/'reports/full-candidate'/Path(output).name
                write_json(report/'late-vlm-runtime.json',{'arm':arm,'native_updates':probes['step'],
                           'KV_gradient_checks':probes['KV_gradient_checks'],'bridge_sha256':file_sha(Path(__file__))})


def phase(args):
    from . import candidate_train
    require(100<=args.steps<=6000,'Full phase limit100..6000')
    if args.resume_from:
        saved=read_json(Path(args.resume_from)/'pretrained_model/late_vlm.json')
        require(saved['full_bridge_sha256']==file_sha(Path(__file__)),'Full bridge changed after save')
    with full_scope(args.profile,args.output,resuming=args.resume_from is not None):
        return candidate_train.run(args)


def evaluate(args):
    from . import evaluation
    root=project_root();c=load_training_config(root/'configs/training_v0.yaml')
    original=evaluation.load_model
    def typed(c,bundle,*,checkpoint=None,device='cpu'):
        p,pre,post=load_policy(checkpoint,device=device)
        arm=read_json(Path(checkpoint)/'late_vlm.json')['arm']
        return p,pre,post,audit_parameters(p,arm)
    try:
        evaluation.load_model=typed
        return evaluation.evaluate(c,root/c['bundle'],split=args.split,output=args.output,checkpoint=args.checkpoint,device=args.device)
    finally:evaluation.load_model=original


def web(args):
    from . import evaluation
    from .demo_server import serve_demo
    original=evaluation.load_model
    def typed(c,bundle,*,checkpoint=None,device='cpu'):
        p,pre,post=load_policy(checkpoint,device=device)
        arm=read_json(Path(checkpoint)/'late_vlm.json')['arm']
        return p,pre,post,audit_parameters(p,arm)
    try:
        evaluation.load_model=typed
        return serve_demo(checkpoint=args.checkpoint,device=args.device,port=args.port)
    finally:evaluation.load_model=original


def qualify(args):
    """Export a recipe only from passing, hash-bound measurements; never train."""
    from .candidate_train import check_evidence
    from .readiness import check_validation_evidence
    root=project_root();cp=Path(args.checkpoint).resolve();decision=read_json(args.decision)
    require(decision['relative_improvement']['passed'], 'Relative K/V improvement criteria not met')
    sha=file_sha(cp/'adapter_model.safetensors')
    require(Path(decision['candidate_checkpoint']).resolve()==cp and decision['candidate_adapter_sha256']==sha,
            'Decision belongs to another checkpoint')
    from .late_vlm import validate_metadata
    meta=validate_metadata(read_json(cp/'late_vlm.json'))
    require(meta['arm']=='candidate' and meta['kind']=='native_trained_checkpoint','Need a trained candidate')
    policy=root/'configs/readiness_balanced_v1.json'
    check_evidence(args.capacity,cp,policy)
    check_validation_evidence(args.validation,cp,policy)
    memory=read_json(args.memory);resume=read_json(args.resume_evidence)
    require(memory['status']=='pass' and Path(memory['checkpoint']).resolve()==cp,'Memory proof differs')
    require(resume['status']=='passed' and resume['optimizer_steps_performed']==1,'Actual trained resume proof required')
    control=Path(resume['control_checkpoint'])/'pretrained_model'
    require(read_json(control/'small_experiment.json')['profile']==read_json(cp/'small_experiment.json')['profile'],
            'Resume proof profile differs')
    value={'version':1,'bridge':'late_vlm_full_bridge_v1','bridge_sha256':file_sha(Path(__file__)),
           'checkpoint':str(cp),'adapter_sha256':sha,'config':str(root/'configs/late_vlm_kv_v1.json'),
           'config_sha256':recipe()['_sha256'],'profile':'candidate','readiness_policy':str(policy),
           'evidence':str(Path(args.capacity).resolve()),'validation_evidence':str(Path(args.validation).resolve()),
           'memory_evidence':str(Path(args.memory).resolve()),'memory_evidence_sha256':file_sha(args.memory),
           'resume_evidence':str(Path(args.resume_evidence).resolve()),'resume_evidence_sha256':file_sha(args.resume_evidence),
           'relative_decision':str(Path(args.decision).resolve()),'relative_decision_sha256':file_sha(args.decision)}
    write_json(args.output,value)
    print('Qualified recipe written:',args.output)


def capacity(args):
    """Fresh-process TRAIN60/five-noise/B1 evidence, without optimizer updates."""
    from .small_cases import selection,compare_predictions,passes
    from .small_train import panel_evaluate
    from .readiness import load_policy as readiness_policy
    from .training import train_config
    from lerobot.datasets.factory import make_train_eval_datasets
    root=project_root();cp=Path(args.checkpoint).resolve();output=Path(args.output).resolve()
    require(not output.exists(),'Use a fresh capacity receipt')
    value=read_json(cp/'late_vlm.json')
    require(value['kind']=='native_trained_checkpoint' and value['new_optimizer_updates']>0,'Trained checkpoint required')
    policy,pre,post=load_policy(cp,device='cuda');c=load_training_config(root/'configs/training_v0.yaml')
    selected=selection(c,'cases60')
    cfg=train_config(c,root/c['bundle'],output=root/'outputs/unused-capacity',device='cuda',episodes=selected['episodes'])
    ds,ev=make_train_eval_datasets(cfg);require(ev is None and len(ds)==60,'Wrong TRAIN60 panel')
    result=panel_evaluate(policy,pre,post,ds,selected,[0,1,2,3,42],batch_size=1)
    ref=read_json(args.reference_evaluation)
    require(Path(ref['checkpoint']).resolve()==cp.parent,'Reference belongs to another checkpoint')
    panel=ref.get('certificate',ref['screen'])
    matched=result if panel['batch_size']==1 and panel['seeds']==result['seeds'] else panel_evaluate(
        policy,pre,post,ds,selected,panel['seeds'],batch_size=panel['batch_size'])
    result.update(stage='cases60',checkpoint=str(cp),adapter_sha256=file_sha(cp/'adapter_model.safetensors'),
                  optimizer_steps_performed=0,reload_comparison=compare_predictions(panel['predictions'],matched['predictions']))
    result['gate_passed']=passes([r['overall'] for r in result['by_seed'].values()],readiness_policy()[0]['capacity_train60_each_seed'])
    write_json(output,result)
    print('TRAIN60 capacity gate:',result['gate_passed'],'fresh reload:',result['reload_comparison']['passed'])


def run(args):
    require(100<=args.steps<=6000,'Full workflow limit100..6000')
    if not args.execute:
        print(json.dumps(describe(args.steps),ensure_ascii=False,indent=2));return
    root=project_root();path=Path(args.recipe)
    require(path.is_file(),'Full run blocked: no qualified late_vlm_full_ready.json; do not train an unqualified model')
    r=read_json(path);require(r['bridge']=='late_vlm_full_bridge_v1' and r['bridge_sha256']==file_sha(Path(__file__)),'Wrong typed full recipe')
    require(file_sha(Path(r['relative_decision']))==r['relative_decision_sha256']
            and read_json(Path(r['relative_decision']))['relative_improvement']['passed'],'Relative comparison no longer qualifies')
    require(args.final_test,'Existing full end-to-end workflow includes final TEST; explicitly pass --final-test only after freezing selection')
    from . import full_run
    original=full_run.subprocess
    locked={p:file_sha(p) for p in Path(__file__).parent.glob('late_vlm*.py')}
    class TypedCommands:
        def __getattr__(self,name):return getattr(original,name)
        def rewrite(self,argv):
            for p,sha in locked.items():require(file_sha(p)==sha,'Typed workflow source changed during run')
            argv=list(argv)
            if '-m' in argv:
                i=argv.index('-m')+1
                if argv[i]=='agri_vla.candidate_train':argv[i:i+1]=['agri_vla.late_vlm_full','phase']
                elif argv[i]=='agri_vla.cli' and len(argv)>i+1 and argv[i+1] in ('evaluate','demo-web'):
                    mode='evaluate' if argv[i+1]=='evaluate' else 'web';argv[i:i+2]=['agri_vla.late_vlm_full',mode]
            return argv
        def run(self,argv,*a,**kw):return original.run(self.rewrite(argv),*a,**kw)
        def Popen(self,argv,*a,**kw):return original.Popen(self.rewrite(argv),*a,**kw)
    try:
        full_run.subprocess=TypedCommands()
        result=full_run.run(args)
        model=Path(result['deploy_model'])
        (model.parent/'deploy.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\n'+shlex.join([
            str(root/'.venv-train/bin/python'),'-m','agri_vla.late_vlm_full','web','--checkpoint',str(model),'--device','cuda','--port','8000'])+'\n')
        return result
    finally:full_run.subprocess=original


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('run');p.add_argument('--steps',type=int,default=6000);p.add_argument('--recipe',type=Path,default=project_root()/'configs/late_vlm_full_ready.json')
    p.add_argument('--execute',action='store_true');p.add_argument('--final-test',action='store_true');p.add_argument('--output',type=Path)
    p.add_argument('--max-new-gib',type=float,default=1.);p.add_argument('--resume',action='store_true');p.add_argument('--patience',type=int,default=3)
    p=sub.add_parser('phase');p.add_argument('--experiment-config',type=Path,required=True);p.add_argument('--profile',choices=('control','candidate'),required=True)
    p.add_argument('--research-run',action='store_true',help='Train as unqualified research without claiming readiness')
    for key in ('init-checkpoint','resume-from','evidence','validation-evidence','readiness-policy'):p.add_argument('--'+key,type=Path)
    p.add_argument('--steps',type=int,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--execute',action='store_true')
    p=sub.add_parser('evaluate');p.add_argument('--split',choices=('val','test'),required=True);p.add_argument('--device',default='cuda');p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('web');p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--port',type=int,default=8000)
    p=sub.add_parser('qualify')
    for key in ('checkpoint','decision','capacity','validation','memory','resume-evidence'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--output',type=Path,default=project_root()/'configs/late_vlm_full_ready.json')
    p=sub.add_parser('capacity')
    for key in ('checkpoint','reference-evaluation','output'):p.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args()
    try:
        {'run':run,'phase':phase,'evaluate':evaluate,'web':web,'qualify':qualify,'capacity':capacity}[args.mode](args)
    except DatasetError as error:
        parser.exit(2,f'Late VLM stopped: {error}\n')


if __name__=='__main__':main()
