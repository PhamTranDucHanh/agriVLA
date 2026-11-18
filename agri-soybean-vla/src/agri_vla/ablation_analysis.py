"""CPU analysis and conservative decision for the frozen initialization protocol."""
from collections import defaultdict
import argparse
import json
import math
from pathlib import Path
import random
import statistics

from .ablation import NAME
from .dataset import read_json, read_jsonl, require
from .preflight import project_root
from .training import write_json, file_sha


def scores(rows):
    require(rows, 'No predictions')
    correct=[r['quantized_action']==r['target_action'] for r in rows]
    intents={r['intent'] for r in rows}
    stop=[r for r in rows if r['intent']=='STOP_ALL']
    off=[p for r in rows for p,t in zip(r['quantized_action'],r['target_action']) if t==0]
    return {'records':len(rows),'exact':sum(correct)/len(rows),
            'zone':sum(p==t for r in rows for p,t in zip(r['quantized_action'],r['target_action']))/(3*len(rows)),
            'macro_intent_exact':statistics.mean(sum(r['quantized_action']==r['target_action'] for r in rows if r['intent']==i)/sum(r['intent']==i for r in rows) for i in intents),
            'STOP':sum(r['quantized_action']==[0,0,0] for r in stop)/len(stop) if stop else None,
            'false_watering':sum(p!=0 for p in off),'off_components':len(off),
            'false_watering_rate':sum(p!=0 for p in off)/len(off) if off else None,
            'MAE':sum(abs(p-t) for r in rows for p,t in zip(r['continuous_action'],r['target_action']))/(3*len(rows))}


def conditional_scene_interval(a,b,metadata,*,draws=10000):
    """Paired scene bootstrap, conditional on the observed sources and training seed."""
    aa={(r['sample_id'],r['seed']):r for r in a};bb={(r['sample_id'],r['seed']):r for r in b}
    require(aa.keys()==bb.keys(),'Unpaired comparison')
    grouped=defaultdict(list)
    for key,row in aa.items():
        other=bb[key];require(row['target_action']==other['target_action'],'Targets differ')
        grouped[metadata[key[0]]['scene_id']].append(int(row['quantized_action']==row['target_action'])-int(other['quantized_action']==other['target_action']))
    groups=[(sum(v),len(v)) for v in grouped.values()];rng=random.Random(20261116);deltas=[]
    for _ in range(draws):
        sampled=rng.choices(groups,k=len(groups));deltas.append(sum(x[0] for x in sampled)/sum(x[1] for x in sampled))
    deltas.sort()
    return {'lower':deltas[int(.025*draws)],'upper':deltas[int(.975*draws)],'clusters':len(groups),'draws':draws,
            'limitation':'scene clusters share some source images; interval is exploratory/conditional, not independent-source or training-seed uncertainty'}


def select_initialization(a,b,criteria,interval):
    """Return winner only when every predeclared criterion is satisfied."""
    av,bv=a['validation_mean'],b['validation_mean']
    checks={'validation_exact_margin':av['exact']-bv['exact']>=criteria['minimum_mean_validation_exact_margin'],
            'validation_zone_margin':av['zone']-bv['zone']>=criteria['minimum_mean_validation_zone_margin'],
            'positive_seeds':sum(a['validation_by_seed'][s]['exact']>b['validation_by_seed'][s]['exact'] for s in a['validation_by_seed'])>=criteria['minimum_seeds_with_positive_validation_exact_margin'],
            'conditional_scene_interval_positive':interval['lower']>0,
            'STOP_not_worse':av['STOP']>=bv['STOP'],
            'false_watering_not_worse':av['false_watering_rate']<=bv['false_watering_rate'],
            'train280_guard':a['train42']['exact']>=b['train42']['exact']-criteria['train280_exact_max_drop'],
            'old60_guard':a['old60_42']['exact']>=b['old60_42']['exact']-criteria['old60_exact_max_drop'],
            'new220_guard':a['new220_42']['exact']>=b['new220_42']['exact']}
    return {'passed':all(checks.values()),'checks':checks}


def summarize(report, old60):
    train=[r for r in report['train']['predictions'] if r['seed']==42]
    rows=report['val_predictions'];by_seed={str(s):scores([r for r in rows if r['seed']==s]) for s in sorted({r['seed'] for r in rows})}
    mean={key:statistics.mean(v[key] for v in by_seed.values()) for key in next(iter(by_seed.values())) if key!='records'}
    return {'train42':scores(train),'old60_42':scores([r for r in train if r['sample_id'] in old60]),
            'new220_42':scores([r for r in train if r['sample_id'] not in old60]),
            'validation_by_seed':by_seed,'validation_mean':mean,
            'validation_exact_noise_seed_std':statistics.stdev(v['exact'] for v in by_seed.values()) if len(by_seed)>1 else None,
            'training_seed_variability':'not measured: training seed42 only',
            'validation_groups_seed42':{key:{v:scores([r for r in rows if r['seed']==42 and r[key]==v]) for v in sorted({r[key] for r in rows})} for key in ('intent','language','wording_partition')}}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--A',type=Path,required=True);parser.add_argument('--B',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    root=project_root();out=root/'reports'/NAME
    a,b=read_json(args.A),read_json(args.B)
    require(read_json(out/'A-B-fairness.json')['historical_B_reusable'],'Historical B reuse not established')
    old60={r['sample_id'] for r in read_json(root/'reports/small-cases/cases60-endpoint-resume300-20261116/selection.json')['items']}
    metadata={r['sample_id']:r for r in read_jsonl(root/'data/manifests/wording_v1/val.jsonl')}
    aa,bb=summarize(a,old60),summarize(b,old60)
    ab=conditional_scene_interval(a['val_predictions'],b['val_predictions'],metadata)
    ba={'lower':-ab['upper'],'upper':-ab['lower']}
    spec=read_json(root/'configs/initialization_ablation_v1.json')
    checkA=select_initialization(aa,bb,spec['initialization_selection_before_data_ablation'],ab)
    checkB=select_initialization(bb,aa,spec['initialization_selection_before_data_ablation'],ba)
    final_seeds=set(a['seeds'])==set(b['seeds'])=={0,1,2,3,42}
    winner='A' if final_seeds and checkA['passed'] else 'B' if final_seeds and checkB['passed'] else None
    write_json(args.output,{'A':aa,'B':bb,'A_minus_B_scene_interval':ab,'A_criteria':checkA,'B_criteria':checkB,
               'five_seed_confirmation_complete':final_seeds,'selected_initialization':winner,
               'data_ablation_allowed':winner is not None,'full_ready':False,
               'source_hashes':{str(p):file_sha(p) for p in (args.A,args.B,Path(__file__))}})
    print('Initialization selected:',winner,'; data ablation permitted:',winner is not None)


if __name__=='__main__':main()
