"""Explicit unqualified full-TRAIN experiment; never opens TEST or claims readiness."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from .dataset import require, read_json, DatasetError
from .preflight import project_root
from .training import file_sha
from .full_run import atomic_json, tree_bytes, ranking, prune, complete_checkpoint, gpu_headroom


def extension_source(parent, config):
    """Accept a completed6000 run, preserving its original artifacts and protocol."""
    parent=Path(parent).resolve();root=project_root()
    require(parent.parent==root/'outputs/train' and not parent.is_symlink(),'Use a completed outputs/train run')
    plan=read_json(parent/'run-plan.json');progress=read_json(parent/'progress.json')
    require(progress['status']=='research_completed_not_qualified' and progress['completed_steps']==6000,
            'Extension requires a normally completed6000 run')
    require(progress['best']==progress['latest'],'This extension requires best and latest both at6000')
    cp=Path(progress['latest']).resolve()
    require(cp.is_relative_to(parent) and int(cp.name)==6000 and complete_checkpoint(cp),'Incomplete checkpoint6000')
    require(plan['profile']=='control' and plan['batch_size']==4 and plan['lr']==1e-4
            and plan['objective']=='endpoint_t1' and plan['scheduler']=='constant_with_warmup'
            and plan['patience']==3 and not plan['TEST_opened'],'Parent protocol differs')
    require(plan['config_sha256']==file_sha(config),'Parent training config changed')
    for name,sha in plan['code_sha256'].items():
        if name!='full_research.py':
            require(file_sha(Path(__file__).with_name(name))==sha,f'Parent training/evaluation source changed: {name}')
    baseline=next(v for v in progress['validations'] if v['step']==6000)
    meta=read_json(cp/'pretrained_model/agri_full_candidate.json')
    require(meta['research_run'] and meta['runner_sha256']==file_sha(Path(__file__).with_name('candidate_train.py')),
            'Checkpoint native runner mismatch')
    return cp,baseline,{'parent_run':str(parent),'parent_plan_sha256':file_sha(parent/'run-plan.json'),
                       'source_checkpoint':str(cp),'adapter_sha256':file_sha(cp/'pretrained_model/adapter_model.safetensors'),
                       'native_start_step':6000,'optimizer_reset':False,'TEST_opened':False}


def run(args):
    root=project_root();out=Path(args.output).resolve()
    require(out.parent==root/'outputs/train' and not out.is_symlink(),'Use outputs/train/NEW_RUN')
    extending=getattr(args,'extend_from',None) is not None
    require(100<=args.steps<=12000 and (args.steps<=6000 or extending)
            and args.max_new_gib>=.25 and args.patience>=1,'Over6000 requires --extend-from completed run')
    config=root/'configs/late_vlm_kv_v1.json';extension=None
    if extending:
        require(args.steps>6000 and args.patience==3,'Extension requires cumulative steps>6000 and patience3')
        initial_cp,baseline,extension=extension_source(args.extend_from,config)
        require(out!=Path(args.extend_from).resolve(),'Use a NEW output to preserve completed6000 run')
    source=root/'outputs/small-cases/latekv-gpu20261116-control-s300/checkpoints/000300/pretrained_model'
    frozen=read_json(root/'reports/latekv-gpu-20261116/control-final.json')
    require(file_sha(source/'adapter_model.safetensors')==frozen['adapter_sha256'],'Source control changed')
    proof=read_json(root/'reports/latekv-gpu-20261116/smoke_report.json')
    require(proof['resume_status']['control']=='passed','Trained resume proof missing')
    files=['full_research.py','candidate_train.py','late_vlm_full.py','late_vlm.py','late_vlm_experiment.py',
           'small_train.py','small_resume.py','training.py','evaluation.py','full_run.py',
           'small_cases.py','augmentation.py','inference_sampling.py','readiness.py']
    if extending:files.append('full_research_extension.py')
    lock={n:file_sha(Path(__file__).with_name(n)) for n in files}
    config=root/'configs/late_vlm_kv_v1.json'
    milestones=sorted({100,*range(1000,args.steps+1,1000),args.steps})
    if extending:milestones=[s for s in milestones if s>6000]
    plan={'mode':'unqualified_full_research','profile':'control','readiness_gate_passed':False,
          'source_checkpoint':str(source),'source_adapter_sha256':frozen['adapter_sha256'],
          'config_sha256':file_sha(config),'code_sha256':lock,'train_records':3600,'val_records':720,
          'train_root':str(root/'data/lerobot/wording_v1/full/train'),
          'batch_size':4,'lr':1e-4,'rank':8,'alpha':8,'warmup':20,'scheduler':'constant_with_warmup',
          'objective':'endpoint_t1','native_inference_steps':1,'new_VLM_KV':False,
          'initialization':'control300 weights only; fresh optimizer/scheduler/RNG for full TRAIN',
          'milestones':milestones,'patience':args.patience,'max_new_gib':args.max_new_gib,
          'TEST_opened':False,'output':str(out),'full_training_executed':False}
    if extending:
        source=initial_cp/'pretrained_model'
        plan.update(extension=extension,source_checkpoint=str(source),source_adapter_sha256=extension['adapter_sha256'],
                    initialization='native6000 full-state continuation; optimizer/scheduler/RNG/step/AMP preserved')
    if not args.execute:
        print(json.dumps(plan,ensure_ascii=False,indent=2));return plan
    require(args.research_run,'Explicit --research-run required; current model has not passed readiness')
    require(sys.prefix==str(root/'.venv-train'),'Use .venv-train')
    gpu_headroom(2400)
    if out.exists():
        require(args.resume and read_json(out/'run-plan.json')==plan,'Resume requires unchanged plan/code/source/budget')
        progress=read_json(out/'progress.json')
    else:
        require(not args.resume,'No existing run to resume')
        out.mkdir(parents=True);atomic_json(out/'run-plan.json',plan)
        progress={'completed_steps':0,'latest':None,'best':None,'best_rank':None,'validations':[],
                  'cache_start_bytes':tree_bytes(root/'.cache/hf-datasets'),'stale':0}
        if extending:
            copied=out/(out.name+'-phase-006000-seed')/'checkpoints/006000'
            shutil.copytree(initial_cp,copied)
            require(complete_checkpoint(copied),'Copied native checkpoint incomplete')
            inherited=dict(baseline,checkpoint=str(copied))
            progress.update(completed_steps=6000,latest=str(copied),best=str(copied),
                            best_rank=list(ranking(inherited['metrics']['overall'])),validations=[inherited])
        atomic_json(out/'progress.json',progress)
    (out/'logs').mkdir(exist_ok=True)
    env=os.environ.copy();env.update(HF_DATASETS_CACHE=str(root/'.cache/hf-datasets'),HF_HUB_OFFLINE='1',
                                     TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    def budget():
        used=tree_bytes(out)+sum(tree_bytes(root/'reports/full-candidate'/p.name) for p in out.glob('*-phase-*'))
        used+=max(0,tree_bytes(root/'.cache/hf-datasets')-progress['cache_start_bytes'])
        require(used<args.max_new_gib*2**30,'Artifact budget exceeded; stopped before next phase')
        require(shutil.disk_usage(out).free>256*2**20,'Insufficient filesystem headroom')
        return used
    def command(argv,label):
        require(all(file_sha(Path(__file__).with_name(n))==h for n,h in lock.items()),'Source changed during run')
        require(file_sha(config)==plan['config_sha256'],'Config changed during run')
        print(label,flush=True)
        with (out/'logs'/f'{label}.log').open('a') as log:
            subprocess.run([sys.executable,'-u','-m',*argv],cwd=root.parent,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    try:
        if not (out/'longest-prompt-memory.json').exists():
            command(['agri_vla.late_vlm_checks','memory','--checkpoint',str(source),
                     '--output',str(out/'longest-prompt-memory.json')],'memory')
        require(read_json(out/'longest-prompt-memory.json')['status']=='pass','Longest-prompt memory did not pass')
        for step in milestones:
            if progress.get('early_stop') or step<=progress['completed_steps']:continue
            budget();gpu_headroom(2400)
            prefix=out.name+f'-phase-{step:06d}'
            phases=list(out.glob(prefix+'*'))
            cps=[cp for p in phases for cp in (p/'checkpoints').glob('[0-9]*') if not cp.is_symlink() and complete_checkpoint(cp)]
            target=max(cps,key=lambda p:int(p.name)) if cps else None
            latest=target or (Path(progress['latest']) if progress['latest'] else None)
            if not target or int(target.name)<step:
                phase=out/(prefix+(f'-recovery{len(phases)}' if phases else ''))
                argv=['agri_vla.full_research_extension' if extending else 'agri_vla.late_vlm_full','phase','--experiment-config',str(config),'--profile','control',
                      '--steps',str(step),'--output',str(phase),'--execute','--research-run']
                argv+=['--resume-from',str(latest)] if latest else ['--init-checkpoint',str(source)]
                command(argv,f'train-{step:06d}-{len(phases)}')
                target=phase/'checkpoints'/f'{step:06d}'
            require(complete_checkpoint(target),'Training checkpoint incomplete')
            val=out/f'val-{step:06d}'
            if not (val/'metrics.json').exists():
                command(['agri_vla.late_vlm_full','evaluate','--split','val','--device','cuda',
                         '--checkpoint',str(target/'pretrained_model'),'--output',str(val)],f'val-{step:06d}')
            result=read_json(val/'metrics.json');require(not result['partial'] and result['metrics']['overall']['records']==720,'Incomplete VAL')
            score=ranking(result['metrics']['overall'])
            progress['validations'].append({'step':step,'checkpoint':str(target),'metrics':result['metrics']})
            if progress['best_rank'] is None or score>tuple(progress['best_rank']):
                progress.update(best=str(target),best_rank=list(score),stale=0)
            else:progress['stale']+=1
            progress.update(completed_steps=step,latest=str(target),status='research_training')
            if progress['stale']>=args.patience:progress['early_stop']='No VAL improvement for declared patience'
            atomic_json(out/'progress.json',progress)
            prune(out,[Path(progress[k]) for k in ('best','latest')]);budget()
        final=out/'final_model';best=Path(progress['best'])/'pretrained_model'
        if not final.exists():
            partial=out/'final_model.partial'
            if partial.exists():
                require(not partial.is_symlink(),'Unsafe staging path');shutil.rmtree(partial)
            shutil.copytree(best,partial);partial.replace(final)
        require(file_sha(final/'adapter_model.safetensors')==file_sha(best/'adapter_model.safetensors'),'Final weights differ')
        (out/'deploy.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(root.parent))+'\n'+
            'export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n'+
            shlex.join([sys.executable,'-m','agri_vla.late_vlm_full','web','--checkpoint',str(final),'--device','cuda','--port','8000'])+'\n')
        report={'status':'research_completed_not_qualified','readiness_gate_passed':False,'TEST_opened':False,
                'best_checkpoint':progress['best'],'deploy_model':str(final),'completed_steps':progress['completed_steps'],
                'best_VAL_metrics':max(progress['validations'],key=lambda v:ranking(v['metrics']['overall']))['metrics'],
                'artifact_bytes':budget(),'early_stop':progress.get('early_stop')}
        atomic_json(out/'final-report.json',report);progress['status']=report['status'];atomic_json(out/'progress.json',progress)
        print(json.dumps(report,ensure_ascii=False,indent=2));return report
    except BaseException as error:
        progress.update(status='interrupted_or_failed',failure=f'{type(error).__name__}: {error}')
        atomic_json(out/'progress.json',progress);raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=6000)
    p.add_argument('--research-run',action='store_true');p.add_argument('--execute',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--max-new-gib',type=float,default=1.)
    p.add_argument('--patience',type=int,default=3)
    p.add_argument('--extend-from',type=Path,help='Completed6000 run; use a new output and preserve native state')
    try:run(p.parse_args())
    except DatasetError as error:p.exit(2,f'Research run stopped: {error}\n')


if __name__=='__main__':main()
