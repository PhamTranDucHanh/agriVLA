"""Milestone-only VALID112 screen; no TRAIN score, optimizer, or TEST."""
import argparse
from pathlib import Path
import time

from .dataset import require,read_json
from .preflight import project_root
from .training import file_sha,load_training_config,write_json
from .late_vlm import load_policy,validate_metadata


def run(checkpoint,output):
    from .evaluation import fixed_validation_selection,data_loader,predict_batch,metrics
    cp=Path(checkpoint).resolve();out=Path(output).resolve();require(not out.exists(),'New output required')
    meta=validate_metadata(read_json(cp/'late_vlm.json'))
    require(meta['kind']=='native_trained_checkpoint' and meta['new_optimizer_updates']>0,'Not a trained checkpoint')
    sha=file_sha(cp/'adapter_model.safetensors');c=load_training_config(project_root()/'configs/training_v0.yaml')
    policy,pre,post=load_policy(cp,device='cuda');panel=fixed_validation_selection(c,112)
    loader,mapping=data_loader(c,policy.config,'val',indices=panel['episodes']);rows=[];start=time.perf_counter()
    for batch in loader:
        r=mapping[int(batch['index'].item())]['source_record']
        require(batch['task']==[r['instruction']] and batch['action'].reshape(-1).tolist()==r['target_action'],'Mapping differs')
        rows.append({k:r[k] for k in ('sample_id','intent','language','wording_partition','target_action')}|{'seed':42}|predict_batch(policy,pre,post,batch,seed=42))
    require(file_sha(cp/'adapter_model.safetensors')==sha,'Frozen model changed')
    result={'scope':'fixed VALID112 milestone screen only; seed42; TRAIN not measured here',
            'checkpoint':str(cp),'adapter_sha256':sha,'arm':meta['arm'],'step':meta['new_optimizer_updates'],
            'val112':{'selection':panel,'predictions':rows,'metrics':metrics(rows)},
            'inference_seconds':time.perf_counter()-start,'optimizer_updates':0,'TEST_opened':False}
    write_json(out,result);print(result['val112']['metrics']['overall'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.checkpoint,a.output)
