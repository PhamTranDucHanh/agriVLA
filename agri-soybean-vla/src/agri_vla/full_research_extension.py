"""Native phases for explicit6000→12000 research continuation; no training-math changes."""
import argparse
from pathlib import Path
from .dataset import require, read_json, DatasetError
from .training import file_sha
from .late_vlm_full import full_scope


def phase(args):
    from . import candidate_train
    require(args.research_run and args.execute and args.profile=='control','Explicit control research execution required')
    require(6000<args.steps<=12000 and args.resume_from is not None,'Extension requires native resume and steps6001..12000')
    cp=Path(args.resume_from)
    start=read_json(cp/'training_state/training_step.json')['step']
    require(6000<=start<args.steps,'Invalid continuation start step')
    meta=read_json(cp/'pretrained_model/late_vlm.json')
    require(meta['full_bridge_sha256']==file_sha(Path(__file__).with_name('late_vlm_full.py')),'Full bridge source changed')
    with full_scope('control',args.output,resuming=True):
        return candidate_train.run(args)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['phase'])
    p.add_argument('--experiment-config',type=Path,required=True);p.add_argument('--profile',choices=['control'],required=True)
    p.add_argument('--resume-from',type=Path,required=True);p.add_argument('--init-checkpoint',type=Path)
    for key in ['evidence','validation-evidence','readiness-policy']:p.add_argument('--'+key,type=Path)
    p.add_argument('--steps',type=int,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--execute',action='store_true');p.add_argument('--research-run',action='store_true')
    try:phase(p.parse_args())
    except DatasetError as error:p.exit(2,f'Extension stopped: {error}\n')


if __name__=='__main__':main()
