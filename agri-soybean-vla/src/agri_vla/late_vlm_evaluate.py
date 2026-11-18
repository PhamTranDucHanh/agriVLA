"""Frozen variant evaluation: original TRAIN280 seed42, fixed VALID112 noise seeds."""
import argparse
from pathlib import Path
import time

from .dataset import require,read_json
from .preflight import project_root
from .training import file_sha,load_training_config,write_json,train_config
from .late_vlm import load_policy,validate_metadata,VERSION


def evaluate(checkpoint,output,seeds):
    from .small_cases import selection
    from .small_train import panel_evaluate
    from .evaluation import data_loader,predict_batch,metrics,fixed_validation_selection
    from lerobot.datasets.factory import make_train_eval_datasets
    root=project_root();cp=Path(checkpoint).resolve();output=Path(output).resolve()
    require(not output.exists(),'Use a fresh receipt path')
    require(seeds in ([42],[0,1,2,3,42]),'Fixed noise seed protocol only')
    meta=validate_metadata(read_json(cp/'late_vlm.json'))
    require(meta['kind']=='native_trained_checkpoint' and meta['new_optimizer_updates']>0,'CPU fixture is not a quality experiment')
    before=file_sha(cp/'adapter_model.safetensors');c=load_training_config(root/'configs/training_v0.yaml')
    policy,pre,post=load_policy(cp,device='cuda');selected=selection(c,'cases280')
    cfg=train_config(c,root/c['bundle'],output=root/'outputs/unused-late-eval',device='cuda',episodes=selected['episodes'])
    ds,ev=make_train_eval_datasets(cfg);require(ev is None and len(ds)==280,'Wrong original TRAIN panel')
    start=time.perf_counter();train=panel_evaluate(policy,pre,post,ds,selected,[42],batch_size=4)
    panel=fixed_validation_selection(c,112);loader,mapping=data_loader(c,policy.config,'val',indices=panel['episodes']);rows=[]
    for seed in seeds:
        for batch in loader:
            raw=mapping[int(batch['index'].item())]['source_record']
            require(batch['task']==[raw['instruction']] and batch['action'].reshape(-1).tolist()==raw['target_action'],'VAL mapping mismatch')
            rows.append({key:raw[key] for key in ('sample_id','intent','language','wording_partition','target_action')}|{'seed':seed}|predict_batch(policy,pre,post,batch,seed=seed))
        print('VALID',seed,metrics([r for r in rows if r['seed']==seed])['overall'],flush=True)
    require(file_sha(cp/'adapter_model.safetensors')==before,'Frozen weights changed')
    write_json(output,{'version':VERSION,'checkpoint':str(cp),'adapter_sha256':before,'arm':meta['arm'],
               'train':train,'val_predictions':rows,'validation_selection':panel,'seeds':seeds,
               'val_by_seed':{str(seed):metrics([r for r in rows if r['seed']==seed]) for seed in seeds},
               'inference_seconds':time.perf_counter()-start,'optimizer_updates':0,'TEST_opened':False,
               'quality_ready':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--seeds',type=int,nargs='+',default=[42])
    args=parser.parse_args();evaluate(args.checkpoint,args.output,args.seeds)
