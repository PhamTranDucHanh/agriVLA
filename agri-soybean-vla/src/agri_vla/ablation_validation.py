"""Five-noise-seed VALID confirmation; reuse TRAIN280 seed42 without extra training."""
import argparse
from pathlib import Path
import time

from .ablation import NAME,assert_lock
from .dataset import read_json,require
from .preflight import project_root
from .training import file_sha,load_training_config,write_json


def run(reference,output):
    root=project_root();assert_lock();source=read_json(reference);checkpoint=Path(source['checkpoint'])
    require(source['seeds']==[42] and not Path(output).exists(),'Need seed42 reference and new output')
    require(file_sha(checkpoint/'adapter_model.safetensors')==source['adapter_sha256'],'Reference weights differ')
    from .small_train import load_checkpoint
    from .evaluation import data_loader,predict_batch,metrics
    c=load_training_config(root/'configs/training_v0.yaml');policy,pre,post=load_checkpoint(c,checkpoint)
    panel=read_json(root/'reports'/NAME/'validation-panel.json')
    loader,mapping=data_loader(c,policy.config,'val',indices=panel['episodes'])
    rows=[];comparisons=[];before=time.perf_counter()
    previous={r['sample_id']:r for r in source['val_predictions']}
    for seed in (42,0,1,2,3):
        for batch in loader:
            record=mapping[int(batch['index'].item())]['source_record']
            require(batch['task']==[record['instruction']] and batch['action'].reshape(-1).tolist()==record['target_action'],'Sample/label mapping differs')
            pred=predict_batch(policy,pre,post,batch,seed=seed)
            row={k:record[k] for k in ('sample_id','intent','language','wording_partition','target_action')}|{'seed':seed}|pred
            rows.append(row)
            if seed==42:
                old=previous[row['sample_id']]
                error=max(abs(x-y) for x,y in zip(row['continuous_action'],old['continuous_action']))
                require(error<=.001 and row['quantized_action']==old['quantized_action'],'Fresh seed42 replay mismatch')
                comparisons.append(error)
        print('VALID',seed,metrics([r for r in rows if r['seed']==seed])['overall'],flush=True)
    require(file_sha(checkpoint/'adapter_model.safetensors')==source['adapter_sha256'],'Inference changed weights')
    write_json(output,{'checkpoint':str(checkpoint),'adapter_sha256':source['adapter_sha256'],
               'seeds':[0,1,2,3,42],'train':source['train'],'train_evaluation_seeds':[42],
               'val_predictions':rows,'val_by_seed':{str(seed):metrics([r for r in rows if r['seed']==seed]) for seed in (0,1,2,3,42)},
               'validation_selection':panel,'fresh_seed42_replay':{'records':112,'max_absolute_error':max(comparisons),'atol':.001,'quantized_all_equal':True},
               'source_reference':str(reference),'source_reference_sha256':file_sha(reference),
               'implementation_sha256':file_sha(Path(__file__)),
               'inference_seconds':time.perf_counter()-before,'optimizer_updates':0,'test_opened':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--reference',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();run(args.reference,args.output)
