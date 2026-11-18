"""Bounded-disk full training workflow. Default is a read-only plan, not training.

Fresh subprocesses release GPU between training/VAL/TEST. Only VAL ranks models.
Keep best and latest native checkpoints; never prune outside this run's marker.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import socket
import base64
import urllib.request
from datetime import datetime
from .dataset import require, read_json, DatasetError
from .preflight import project_root
from .training import file_sha


def tree_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob('*') if p.is_file() and not p.is_symlink()) if Path(path).exists() else 0


def atomic_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def ranking(overall):
    values = (overall['exact_action_accuracy'], overall['zone_accuracy'], -overall['continuous_mae'])
    require(all(math.isfinite(v) for v in values), 'Nonfinite validation metric')
    return values


def complete_checkpoint(cp):
    """Ignore interrupted saves rather than treating an existing folder as resumable."""
    try:
        cp=Path(cp);model=cp/'pretrained_model';state=cp/'training_state'
        lock=read_json(model/'agri_checkpoint_lock.json')
        if not lock['files'] or any(file_sha(model/n)!=h for n,h in lock['files'].items()):return False
        if read_json(state/'training_step.json')['step']!=int(cp.name):return False
        for name in ('scheduler_state.json','amp_scaler_state.json'):
            read_json(state/name)
        from safetensors import safe_open
        for name in ('optimizer_state.safetensors','rng_state.safetensors'):
            with safe_open(state/name,framework='np') as f:
                if not list(f.keys()):return False
        return True
    except Exception:
        return False


def prune(root, keep):
    root = Path(root).resolve()
    require((root / 'run-plan.json').is_file() and root.parent == project_root() / 'outputs/train', 'Unowned prune root')
    keep = {Path(p).resolve() for p in keep}
    require(keep and all(p.is_relative_to(root) for p in keep), 'Retain at least one run-owned checkpoint')
    deleted = []
    for phase in root.glob('*-phase-*'):
        if not phase.is_dir() or phase.is_symlink():
            continue
        for cp in (phase / 'checkpoints').glob('[0-9]*'):
            require(not cp.is_symlink() and cp.resolve().is_relative_to(root), 'Unsafe checkpoint path')
            if cp.resolve() not in keep:
                weight=cp/'pretrained_model/adapter_model.safetensors'
                receipt = {'path': str(cp), 'bytes': tree_bytes(cp),
                           'adapter_sha256': file_sha(weight) if weight.is_file() else None}
                # Persist the record before freeing the run-owned duplicate.
                with (root / 'pruned.jsonl').open('a') as f:
                    f.write(json.dumps(receipt) + '\n')
                shutil.rmtree(cp); deleted.append(receipt)
        restored = phase / 'restored_before_update'
        if restored.is_dir() and not restored.is_symlink():
            shutil.rmtree(restored)
    return deleted


def baseline(split):
    from .evaluation import metrics, quantize_action
    source = project_root() / 'data/lerobot/wording_v1/full' / split / 'sample_metadata.jsonl'
    rows = []
    for line in source.read_text().splitlines():
        r = json.loads(line)['source_record']
        rows.append({k: r[k] for k in ('sample_id', 'language', 'intent', 'wording_partition', 'target_action')} | quantize_action([0.,0.,0.]))
    return metrics(rows)


def web_smoke(model, output, env):
    """Actual HTTP upload -> saved policy -> action; no target is sent to inference."""
    root=project_root()
    with socket.socket() as port_socket:
        port_socket.bind(('127.0.0.1',0));port=port_socket.getsockname()[1]
    argv=[str(root/'.venv-train/bin/python'),'-u','-m','agri_vla.cli','demo-web',
          '--checkpoint',str(model),'--device','cuda','--port',str(port)]
    with (output/'logs/web-smoke.log').open('a') as log:
        process=subprocess.Popen(argv,cwd=root.parent,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            url=f'http://127.0.0.1:{port}'
            deadline=time.monotonic()+180
            while True:
                require(process.poll() is None,'Web process exited while loading')
                try:
                    status=json.load(urllib.request.urlopen(url+'/api/status',timeout=3))
                except (OSError,ValueError):status={'status':'loading'}
                require(status['status']!='error',f'Web load failed: {status}')
                if status['status']=='ready':break
                require(time.monotonic()<deadline,'Web loading timeout')
                time.sleep(1)
            row=json.loads((root/'data/lerobot/wording_v1/full/train/sample_metadata.jsonl').open().readline())['source_record']
            payload={'instruction':row['instruction'],'wilt_notes':[None,None,None],
                     'images':['data:image/jpeg;base64,'+base64.b64encode((root/row['images'][z]).read_bytes()).decode()
                               for z in ('zone1','zone2','zone3')]}
            request=urllib.request.Request(url+'/api/infer',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
            start=time.perf_counter()
            with urllib.request.urlopen(request,timeout=120) as response:
                prediction=json.load(response)
            action=prediction['quantized_action']
            require(len(action)==3 and all(type(x) is int and 0<=x<=3 for x in action),'Bad web output')
            report={'technical_web_passed':True,'checkpoint':str(model),'sample_id':row['sample_id'],
                    'input_fields':list(payload),'target_sent_to_model':False,'target':row['target_action'],
                    'prediction':prediction,'example_correct':action==row['target_action'],'http_seconds':time.perf_counter()-start}
            atomic_json(output/'web-smoke.json',report)
            return report
        finally:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)


def gpu_headroom(required_mib):
    # Do not load a second policy alongside the local GPU demo on a4GiB card.
    for entry in Path('/proc').glob('[0-9]*/cmdline'):
        try:parts=entry.read_bytes().decode(errors='replace').split('\0')
        except OSError:continue
        if '-m' in parts and parts[parts.index('-m')+1:parts.index('-m')+2]==['agri_vla.cli'] and 'demo-web' in parts:
            cpu='--device' in parts and parts[parts.index('--device')+1:parts.index('--device')+2]==['cpu']
            require(cpu,f'GPU demo PID{entry.parent.name} is active; stop your demo with Ctrl+C before full training')
    r=subprocess.run(['/usr/lib/wsl/lib/nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
    free=int(r.stdout.strip().splitlines()[0])
    require(free>=required_mib,f'GPU free{free}MiB < measured reserve+margin{required_mib}MiB; close your GPU apps, no process is killed')


def run(args):
    root = project_root(); recipe_path = Path(args.recipe).resolve()
    require(recipe_path.is_file(), 'Full run remains locked: no measured recipe. Complete TRAIN/VAL/reload/memory/resume checks first.')
    recipe = read_json(recipe_path)
    require(recipe['version'] == 1, 'Unknown full recipe')
    source = Path(recipe['checkpoint']).resolve(); config = Path(recipe['config']).resolve()
    require(file_sha(config) == recipe['config_sha256'] and file_sha(source / 'adapter_model.safetensors') == recipe['adapter_sha256'], 'Recipe source changed')
    from .candidate_train import check_evidence
    policy_path = Path(recipe['readiness_policy']) if recipe.get('readiness_policy') else None
    evidence = check_evidence(Path(recipe['evidence']), source, policy_path)
    from .readiness import check_validation_evidence
    validation = check_validation_evidence(Path(recipe['validation_evidence']), source, policy_path)
    for name in ('memory_evidence', 'resume_evidence'):
        proof = read_json(Path(recipe[name]))
        require(proof['status'] in ('pass','passed'), f'Missing passing {name}')
        require(file_sha(Path(recipe[name])) == recipe[name + '_sha256'], f'Changed {name}')
        if name == 'memory_evidence':
            require(Path(proof['checkpoint']).resolve()==source, 'Memory evidence belongs to another model')
        else:
            control=Path(proof['control_checkpoint'])/'pretrained_model/small_experiment.json'
            if not control.exists():control=control.with_name('agri_full_candidate.json')
            require(read_json(control)['profile']==read_json(source/'small_experiment.json')['profile'],'Resume evidence profile differs')
    required_gpu_mib=math.ceil(read_json(Path(recipe['memory_evidence']))['torch_peak_reserved']/2**20)+512
    smoke = Path(recipe['full_smoke_checkpoint']).resolve() if recipe.get('full_smoke_checkpoint') else None
    if smoke:
        require(read_json(smoke/'training_state/training_step.json')['step']==100, 'Full smoke must end at100')
        require(file_sha(smoke/'pretrained_model/adapter_model.safetensors')==recipe['full_smoke_adapter_sha256'], 'Full smoke changed')
        require(read_json(smoke/'pretrained_model/train_config.json')['dataset']['episodes'] is None,'Smoke is not full TRAIN view')
    total = args.steps
    require(not smoke or total >= 1000, 'A resumed full smoke needs at least the next1000-step VAL milestone')
    patience=getattr(args,'patience',3)
    require(type(patience) is int and patience>=1,'Patience must be at least1 VAL phase')
    require(100 <= total <= 12000 and args.max_new_gib >= .25, 'Invalid step/disk budget')
    milestones = [100, *range(1000, total + 1, 1000)]
    if total not in milestones: milestones.append(total)
    out = Path(args.output).resolve() if args.output else root / 'outputs/train' / ('full-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
    require(out.parent == root / 'outputs/train' and not out.is_symlink(), 'Use outputs/train/RUN directly')
    # All new source/caches are accounted against the user budget, not the WSL virtual capacity.
    budget = int(args.max_new_gib * 2**30)
    plan = {'recipe': str(recipe_path), 'recipe_sha256': file_sha(recipe_path), 'dataset_train': str(root / 'data/lerobot/wording_v1/full/train'),
            'train_records':3600, 'val_records':720, 'test_records':720,
            'initialization':'TRAIN-small weights only; reset optimizer for full3600', 'milestones':milestones,
            'checkpoint_selection':'VAL exact, then zone, then lower MAE; TEST after selection only',
            'retention':'best + latest; run-owned checkpoints only', 'max_new_bytes':budget,
            'config':str(config), 'profile':recipe['profile'], 'output':str(out),
            'workflow_sha256':file_sha(Path(__file__)),
            'quality_goal':{'exact':.90,'zone':.95,'stop':1.},
            'required_gpu_free_mib':required_gpu_mib,
            'small_evidence':evidence, 'validation_evidence':validation, 'full_training_executed':False}
    plan['validation_patience']=patience
    plan['code_sha256'] = {name: file_sha(Path(__file__).with_name(name)) for name in
                          ('candidate_train.py', 'small_train.py', 'small_resume.py', 'training.py',
                           'evaluation.py', 'inference_sampling.py', 'augmentation.py', 'readiness.py')}
    plan['initial_native_full_step'] = 100 if smoke else 0
    if not args.execute:
        print(json.dumps(plan,ensure_ascii=False,indent=2)); return plan
    require(sys.prefix == str(root / '.venv-train'), 'Use .venv-train')
    gpu_headroom(required_gpu_mib)
    if out.exists():
        require(args.resume and (out/'run-plan.json').is_file(), 'Existing output requires --resume and own marker')
        old=read_json(out/'run-plan.json')
        require(old['recipe_sha256']==plan['recipe_sha256'] and old['milestones']==milestones
                and old['workflow_sha256']==plan['workflow_sha256'], 'Resume plan/code differs')
        require(old['validation_patience']==patience,'Resume patience differs')
        require(old['code_sha256'] == plan['code_sha256'], 'Training/evaluation code changed since full run began')
        progress=read_json(out/'progress.json')
    else:
        require(not args.resume, 'No run to resume')
        out.mkdir(parents=True);atomic_json(out/'run-plan.json',plan)
        progress={'completed_steps':100 if smoke else 0,'latest':str(smoke) if smoke else None,'best':None,'best_rank':None,'validations':[],
                  'cache_start_bytes':tree_bytes(root/'.cache/hf-datasets')}
        atomic_json(out/'progress.json',progress)
    logs=out/'logs';logs.mkdir(exist_ok=True)
    env=os.environ.copy();env['HF_DATASETS_CACHE']=str(root/'.cache/hf-datasets');env['HF_HUB_OFFLINE']='1'
    env['HF_DATASETS_OFFLINE']='1';env['TOKENIZERS_PARALLELISM']='false'
    py=str(root/'.venv-train/bin/python')
    def command(argv,label):
        require(file_sha(Path(__file__)) == plan['workflow_sha256']
                and all(file_sha(Path(__file__).with_name(name)) == sha for name, sha in plan['code_sha256'].items()),
                'Training/evaluation code changed during the run; stop before the next phase')
        print(label,flush=True)
        with (logs/(label+'.log')).open('a') as f:
            subprocess.run([py,'-u',*argv],env=env,cwd=root.parent,stdout=f,stderr=subprocess.STDOUT,check=True)
    def disk_check():
        report_bytes=sum(tree_bytes(root/'reports/full-candidate'/p.name) for p in out.glob('*-phase-*'))
        growth=max(0,tree_bytes(root/'.cache/hf-datasets')-progress['cache_start_bytes'])
        used=tree_bytes(out)+report_bytes+growth
        require(used < budget, f'New data budget exceeded: {used/2**20:.1f}MiB; stopping before next phase')
        require(shutil.disk_usage(out).free > 256*2**20,'Less than256MiB physical filesystem headroom')
        return used
    try:
        for step in milestones:
            if progress.get('early_stop_reason'):break
            if step <= progress['completed_steps']:continue
            disk_check()
            gpu_headroom(required_gpu_mib)
            phases=list(out.glob(out.name+f'-phase-{step:06d}*'))
            # Recover from the latest fully locked checkpoint of an interrupted phase.
            candidates=[]
            for phase in phases:
                for cp in (phase/'checkpoints').glob('[0-9]*'):
                    if complete_checkpoint(cp):
                        candidates.append(cp)
            target=max(candidates,key=lambda p:int(p.name)) if candidates else None
            latest=target if target else Path(progress['latest']) if progress['latest'] else None
            if not target or int(target.name) < step:
                phase=out/(out.name+f'-phase-{step:06d}'+(f'-recovery{len(phases)}' if phases else ''))
                argv=['-m','agri_vla.candidate_train','--experiment-config',str(config),'--profile',recipe['profile'],
                      '--steps',str(step),'--output',str(phase),'--execute']
                if latest:
                    argv += ['--resume-from',str(latest)]
                else:
                    argv += ['--init-checkpoint',str(source),'--evidence',recipe['evidence'],
                             '--validation-evidence',recipe['validation_evidence']]
                    if policy_path:argv += ['--readiness-policy',str(policy_path)]
                command(argv,f'train-{step:06d}-{len(phases)}')
                target=phase/'checkpoints'/f'{step:06d}'
            require(target and (target/'pretrained_model/adapter_model.safetensors').is_file(),'No trained target checkpoint')
            # Smoke100 proves full data plumbing; select only from >=1000 (or final smaller budget).
            if step >= 1000 or step == total:
                val=out/f'val-{step:06d}'
                if not (val/'metrics.json').exists():
                    command(['-m','agri_vla.cli','evaluate','--split','val','--device','cuda','--checkpoint',str(target/'pretrained_model'),'--output',str(val)],f'val-{step:06d}')
                result=read_json(val/'metrics.json');score=ranking(result['metrics']['overall'])
                require(not result['partial'] and result['metrics']['overall']['records']==720,'Incomplete VAL')
                update_file=root/'reports/full-candidate'/target.parents[1].name/'updates.jsonl'
                updates=[json.loads(s) for s in update_file.read_text().splitlines()] if update_file.exists() else []
                losses=[x['loss'] for x in updates[-100:]]
                progress['validations'].append({'step':step,'checkpoint':str(target),'metrics':result['metrics'],
                    'train_loss_mean_last100':sum(losses)/len(losses) if losses else None})
                if progress['best_rank'] is None or score > tuple(progress['best_rank']):
                    progress['best'],progress['best_rank']=str(target),list(score)
                    progress['stale_val_phases']=0
                else:
                    progress['stale_val_phases']=progress.get('stale_val_phases',0)+1
                if progress.get('stale_val_phases',0)>=patience:
                    progress['early_stop_reason']=f'No VAL improvement for{patience} phases; retain best'
            progress.update(completed_steps=step,latest=str(target),status='training')
            atomic_json(out/'progress.json',progress)
            prune(out,[Path(p) for p in (progress['best'],progress['latest']) if p])
            disk_check()
        best=Path(progress['best']);test=out/'test-final'
        if not (test/'metrics.json').exists():
            command(['-m','agri_vla.cli','evaluate','--split','test','--device','cuda','--checkpoint',str(best/'pretrained_model'),'--output',str(test)],'test-final')
        test_result=read_json(test/'metrics.json');m=test_result['metrics']['overall']
        require(not test_result['partial'] and m['records']==720,'Incomplete final TEST')
        final=out/'final_model'
        if not final.exists():
            staging = out/'final_model.partial'
            if staging.exists():
                require(not staging.is_symlink(), 'Unsafe final staging path')
                shutil.rmtree(staging)
            shutil.copytree(best/'pretrained_model',staging)
            staging.replace(final)
        require(file_sha(final/'adapter_model.safetensors') == file_sha(best/'pretrained_model/adapter_model.safetensors'),
                'Final model differs from VAL-selected checkpoint')
        web=read_json(out/'web-smoke.json') if (out/'web-smoke.json').exists() else web_smoke(final,out,env)
        quality=m['exact_action_accuracy']>=.90 and m['zone_accuracy']>=.95 and m['stop_exact_accuracy']==1. and web['example_correct']
        import shlex
        (out/'deploy.sh').write_text('#!/usr/bin/env bash\nset -euo pipefail\ncd '+shlex.quote(str(root.parent))+'\n'+
            shlex.join([py,'-m','agri_vla.cli','demo-web','--checkpoint',str(final),'--device','cuda','--port','8000'])+'\n')
        report={'status':'completed' if quality else 'completed_quality_not_met','quality_gate_passed':quality,
                'quality_goal_declared_before_full':{'exact':.90,'zone':.95,'stop':1.},
                'best_checkpoint':str(best),'deploy_model':str(final),'test':test_result,'web_smoke':web,
                'always_off_baselines':{'val':baseline('val'),'test':baseline('test')},
                'new_bytes_measured':disk_check(),'optimizer_steps':progress['completed_steps'],
                'requested_optimizer_steps':total,'early_stop_reason':progress.get('early_stop_reason'),
                'test_used_for_selection':False,'hardware_irrigation_verified':False}
        atomic_json(out/'final-report.json',report);progress['status']=report['status'];atomic_json(out/'progress.json',progress)
        print(json.dumps(report,ensure_ascii=False,indent=2))
        return report
    except BaseException as exc:
        progress.update(status='interrupted_or_failed',failure=f'{type(exc).__name__}: {exc}')
        atomic_json(out/'progress.json',progress)
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recipe',type=Path,default=project_root()/'configs/full_ready.json')
    p.add_argument('--output',type=Path);p.add_argument('--steps',type=int,default=6000)
    p.add_argument('--max-new-gib',type=float,default=1.)
    p.add_argument('--execute',action='store_true');p.add_argument('--resume',action='store_true')
    args = p.parse_args()
    try:
        run(args)
    except DatasetError as error:
        p.exit(2, f'Full run stopped: {error}\n')

if __name__=='__main__':main()
