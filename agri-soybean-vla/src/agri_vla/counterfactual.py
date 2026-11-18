"""TRAIN-only manifest view and reproducible sampling plan. No ML training imports."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import csv
import hashlib
import html
import itertools
import json
import os
from pathlib import Path
import random

from .dataset import require, read_json, read_jsonl
from .policy import load_unique_yaml
from .preflight import project_root
from .revision import POLICY, revision_action


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stable(*values):
    return hashlib.sha256(json.dumps(values, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def request_key(intent, params):
    return intent + ':' + json.dumps(params, sort_keys=True, separators=(',', ':'))


DEPENDENT = [('AUTO', {}), ('ZONE_ONLY', {'zone': 1}), ('CAP_LEVEL', {'cap': 1}),
             ('EXCLUDE_ZONE', {'zone': 1}), ('WORST_ONLY', {}),
             ('ZONE_ONLY', {'zone': 2}), ('CAP_LEVEL', {'cap': 2}),
             ('EXCLUDE_ZONE', {'zone': 2}), ('ZONE_ONLY', {'zone': 3}),
             ('EXCLUDE_ZONE', {'zone': 3})]
REQUESTS = [('STOP_ALL', {})] + [('WATER_ALL', {'level': v}) for v in (1, 2, 3)] + DEPENDENT


def independent_action(wilt, intent, params):
    """Separate arithmetic oracle; no call to either implementation of the policy."""
    if intent == 'STOP_ALL':
        return [0, 0, 0]
    if intent == 'WATER_ALL':
        return [params['level']] * 3
    base = [0 if w < 2 else w-1 for w in wilt]
    if intent == 'AUTO':
        return base
    if intent == 'WORST_ONLY':
        return [v if w == max(wilt) else 0 for w, v in zip(wilt, base)]
    if intent == 'CAP_LEVEL':
        return [min(v, params['cap']) for v in base]
    if intent == 'ZONE_ONLY':
        return [v if z == params['zone'] else 0 for z, v in enumerate(base, 1)]
    if intent == 'EXCLUDE_ZONE':
        return [0 if z == params['zone'] else v for z, v in enumerate(base, 1)]
    raise ValueError(intent)


def model_inputs(row):
    """Paths for a future loader; not a native tensor dataset or a trainer hook."""
    return {**{f'observation.images.camera{i}': row['images'][f'zone{i}'] for i in (1, 2, 3)},
            'observation.state': [0, 0, 0], 'task': row['instruction']}


def audit(rows):
    scenes = defaultdict(list)
    images, groups, levels, targets = Counter(), Counter(), Counter(), Counter()
    unique_image_wilt, group_images, group_scenes = {},defaultdict(set),defaultdict(set)
    for row in rows:
        scenes[row['scene_id']].append(row)
        images.update(row['source_image_ids']); groups.update(row['source_groups'])
        levels.update(row['teacher_annotations']['wilt_scores']); targets.update(row['target_action'])
        for image,group,wilt in zip(row['source_image_ids'],row['source_groups'],row['teacher_annotations']['wilt_scores']):
            unique_image_wilt[image]=wilt;group_images[group].add(image);group_scenes[group].add(row['scene_id'])
    tie_scenes={sid for sid,rr in scenes.items() if max(rr[0]['teacher_annotations']['wilt_scores'])>=2 and rr[0]['teacher_annotations']['wilt_scores'].count(max(rr[0]['teacher_annotations']['wilt_scores']))>1}
    return {'records': len(rows), 'scenes': len(scenes), 'unique_images': len(images),
            'unique_source_groups': len(groups), 'intents': dict(Counter(r['intent'] for r in rows)),
            'request_parameters': dict(Counter(request_key(r['intent'],r['intent_params']) for r in rows)),
            'languages': dict(Counter(r['language'] for r in rows)),
            'record_camera_source_group_exposures': dict(groups),
            'unique_images_per_group':{g:len(v) for g,v in sorted(group_images.items())},
            'scene_incidence_per_group':{g:len(v) for g,v in sorted(group_scenes.items())},
            'unique_image_wilt_counts':dict(sorted(Counter(unique_image_wilt.values()).items())),
            'eligible_tie_scenes':len(tie_scenes),
            'eligible_tie_scenes_with_worst_request':sum(any(r['intent']=='WORST_ONLY' for r in scenes[s]) for s in tie_scenes),
            'all_below_threshold_scenes':sum(max(rr[0]['teacher_annotations']['wilt_scores'])<2 for rr in scenes.values()),
            'record_camera_wilt_counts': dict(sorted(levels.items())),
            'action_component_counts': dict(sorted(targets.items())),
            'cases_per_scene_histogram': dict(Counter(len({request_key(r['intent'],r['intent_params']) for r in rr}) for rr in scenes.values())),
            'records_per_scene_histogram': dict(Counter(len(rr) for rr in scenes.values())),
            'max_record_camera_exposure_per_image': max(images.values()),
            'independence_warning': 'records/scenes reuse source images; none of these counts implies independent observations'}


def build_view(original, seed=42):
    scenes, pool = {}, defaultdict(dict)
    translation = defaultdict(list)
    for row in original:
        require(row['split'] == 'train' and row['wording_partition'] == 'seen', 'Only observed TRAIN allowed')
        require(row['policy_version'] == POLICY, 'Wrong policy version')
        previous = scenes.setdefault(row['scene_id'], row)
        require(all(previous[k] == row[k] for k in ('images','source_image_ids','source_groups','teacher_annotations')), 'Inconsistent scene')
        translation[row['translation_pair_id']].append(row)
    for rr in translation.values():
        require(len(rr) == 2 and {r['language'] for r in rr} == {'vi','en'}, 'Broken source translation pair')
        a,b = rr
        require(all(a[k] == b[k] for k in ('images','target_action','intent','intent_params','template_family_id')), 'Inconsistent translation semantics')
        key = request_key(a['intent'], a['intent_params'])
        # Canonical observed examples: text itself never comes from a held-out row.
        family = a['template_family_id']
        candidate = {r['language']: r for r in rr}
        if family in pool[key]:
            require(all(pool[key][family][lang]['instruction'] == candidate[lang]['instruction'] for lang in ('vi','en')), 'Family text conflict')
        else:
            pool[key][family] = candidate
    require(all(len(pool[request_key(i,p)]) >= 2 for i,p in REQUESTS), 'Insufficient observed TRAIN synonyms or parameters')
    view, unique = [], {}
    def append(scene, wording, origin):
        target = revision_action(scene['teacher_annotations']['wilt_scores'], wording['intent'], wording['intent_params'])
        require(target == independent_action(scene['teacher_annotations']['wilt_scores'],wording['intent'],wording['intent_params']), 'Policy/oracle mismatch')
        signature = (scene['scene_id'], wording['language'], wording['instruction'])
        if signature in unique:
            require(unique[signature]['target_action'] == target, 'Input-label contradiction')
            return
        sid = wording['sample_id'] if origin == 'original' else 'cf1_' + stable(seed, *signature)[:24]
        pair = 'cfpair_' + stable(scene['scene_id'],wording['template_family_id'],wording['intent_params'])[:24]
        row = {k: deepcopy(scene[k]) for k in ('scene_id','images','source_image_ids','source_groups','teacher_annotations')}
        row.update({k:deepcopy(wording[k]) for k in ('instruction','intent','intent_params','language','template_family_id')})
        row.update(sample_id=sid, split='train', wording_partition='seen', policy_version=POLICY,
                   target_action=target, translation_pair_id=pair, origin=origin,
                   scene_source_sample_id=scene['sample_id'], wording_source_sample_id=wording['sample_id'],
                   view_version='counterfactual_train_v1')
        view.append(row); unique[signature] = row
    for row in original:
        append(row,row,'original')
    ordered = sorted(scenes, key=lambda sid:stable(seed,sid))
    for j, sid in enumerate(ordered):
        scene = scenes[sid]
        requests = [('STOP_ALL',{}), ('WATER_ALL',{'level':1+j%3}),
                    DEPENDENT[j%10],DEPENDENT[(j+5)%10]]
        wilt=scene['teacher_annotations']['wilt_scores']
        if max(wilt)>=2 and wilt.count(max(wilt))>1 and not any(i=='WORST_ONLY' for i,p in requests):
            requests[-1]=('WORST_ONLY',{})
        for slot,(intent,params) in enumerate(requests):
            families = sorted(pool[request_key(intent,params)])
            offset = int(stable(seed,sid,slot)[:8],16)%len(families)
            for family in [families[offset]] + ([families[(offset+1)%len(families)]] if slot == j%4 else []):
                for lang in ('vi','en'):
                    append(scene,pool[request_key(intent,params)][family][lang],'counterfactual')
    return view, scenes


def sampling_plan(view, visits, cap, seed, *, max_image_visits=None):
    """Coverage first; rare group/image second. Deficit selection within each scene."""
    rng = random.Random(seed)
    choice_rng = random.Random(seed + 1009)
    by_scene, pair_rows = defaultdict(list), defaultdict(list)
    for row in view:
        pair_rows[row['translation_pair_id']].append(row)
    for pair, rr in pair_rows.items():
        require(len(rr)==2 and {r['language'] for r in rr}=={'vi','en'}, 'Broken view translation pair')
        by_scene[rr[0]['scene_id']].append((pair,rr))
    source = {sid: rr[0][1][0] for sid,rr in by_scene.items()}
    group_images, image_scenes = defaultdict(set), defaultdict(set)
    for sid,row in source.items():
        for group,image in zip(row['source_groups'], row['source_image_ids']):
            group_images[group].add(image); image_scenes[image].add(sid)
    scene_counts, group_counts, image_counts = Counter(),Counter(),Counter()
    intent_counts, request_counts, family_counts, pair_counts = Counter(),Counter(),Counter(),Counter()
    coverage = sorted(by_scene); rng.shuffle(coverage)
    require(len(coverage)<=visits<=cap*len(coverage), 'Invalid scene visit budget')
    def fits(sid):
        counts=Counter(source[sid]['source_image_ids'])
        return scene_counts[sid]<cap and (max_image_visits is None or all(image_counts[image]+count<=max_image_visits for image,count in counts.items()))
    plan=[]
    for visit in range(visits):
        anchor_group=anchor_image=None
        if visit<len(coverage):
            sid=coverage[visit]
        else:
            candidates=[]
            for group,images in group_images.items():
                eligible=[image for image in sorted(images) if any(fits(s) for s in image_scenes[image])]
                if eligible:
                    candidates.append((group,eligible))
            require(candidates, 'Sampling cap exhausted')
            rng.shuffle(candidates)
            anchor_group,images=min(candidates,key=lambda x:group_counts[x[0]])
            rng.shuffle(images)
            anchor_image=min(images,key=lambda image:image_counts[image])
            candidates=sorted(s for s in image_scenes[anchor_image] if fits(s))
            rng.shuffle(candidates); sid=min(candidates,key=lambda s:scene_counts[s])
        require(fits(sid),'Image cap cannot preserve requested scene coverage')
        candidates=by_scene[sid][:];choice_rng.shuffle(candidates)
        pair,rr=min(candidates,key=lambda item:(intent_counts[item[1][0]['intent']],
                   request_counts[request_key(item[1][0]['intent'],item[1][0]['intent_params'])],
                   pair_counts[item[0]], family_counts[item[1][0]['template_family_id']]))
        r=rr[0]; scene_counts[sid]+=1; intent_counts[r['intent']]+=1
        request_counts[request_key(r['intent'],r['intent_params'])]+=1
        family_counts[r['template_family_id']]+=1; pair_counts[pair]+=1
        group_counts.update(source[sid]['source_groups']); image_counts.update(source[sid]['source_image_ids'])
        plan.append({'visit':visit,'scene_id':sid,'pair_id':pair,
                     'sample_ids':[x['sample_id'] for x in sorted(rr,key=lambda x:('vi','en').index(x['language']))],
                     'phase':'coverage' if visit<len(coverage) else 'source_deficit',
                     'anchor_group':anchor_group,'anchor_image':anchor_image})
    return plan, {'visits':len(plan),'records':2*len(plan),'covered_scenes':len(scene_counts),
                  'scene_visit_histogram':dict(Counter(scene_counts.values())),
                  'group_camera_exposures':dict(group_counts),'covered_images':len(image_counts),
                  'intent_pair_counts':dict(intent_counts),'request_pair_counts':dict(request_counts),
                  'image_camera_exposure_min_max':[min(image_counts.values()),max(image_counts.values())],
                  'max_image_camera_visits_cap':max_image_visits,
                  'warning':'bounded, coverage-preserving deficit sampler; exact uniform group/intent marginals are not guaranteed'}


def preview(output, view):
    scenes=defaultdict(list)
    for r in view:scenes[r['scene_id']].append(r)
    chosen=[]
    # Deterministic coverage examples, including both eligible ties and all-OFF.
    predicates=[lambda r:max(r['teacher_annotations']['wilt_scores'])<2,
                lambda r:max(r['teacher_annotations']['wilt_scores'])>=2 and r['teacher_annotations']['wilt_scores'].count(max(r['teacher_annotations']['wilt_scores']))>1]
    for i,p in REQUESTS:
        predicates.append(lambda r,i=i,p=p:r['intent']==i and r['intent_params']==p)
    for predicate in predicates:
        possible=[r for r in view if predicate(r) and r['scene_id'] not in chosen]
        if possible:chosen.append(min(possible,key=lambda r:r['sample_id'])['scene_id'])
    cards=[]
    for sid in chosen:
        rows=scenes[sid];r=rows[0]
        images=''.join(f'<figure><img loading="lazy" src="{html.escape(os.path.relpath(project_root()/path,output))}"><figcaption>{zone}</figcaption></figure>' for zone,path in r['images'].items())
        table=''.join('<tr>'+''.join(f'<td>{html.escape(str(x))}</td>' for x in (row['language'],row['instruction'],row['intent'],row['intent_params'],row['target_action'],row['origin']))+'</tr>' for row in rows)
        cards.append(f'<section><h2>{html.escape(sid)}</h2><p>Teacher wilt (chỉ để đối chiếu nhãn): {r["teacher_annotations"]["wilt_scores"]}</p><div class="images">{images}</div><table><tr><th>Ngôn ngữ</th><th>Câu lệnh</th><th>Intent</th><th>Tham số</th><th>Target</th><th>Nguồn</th></tr>{table}</table></section>')
    content='<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>TRAIN đối chiếu v1</title><style>body{font:16px system-ui;margin:24px;color:#172324}section{border-top:2px solid #638;padding:16px 0}.images{display:flex;flex-wrap:wrap}figure{margin:8px}img{width:260px;height:210px;object-fit:contain}table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #bbb;text-align:left}tr:nth-child(even){background:#eef5ef}</style><h1>TRAIN đối chiếu v1</h1><p>Ảnh gốc được tham chiếu, không sao chép. Đây là câu và target để kiểm dữ liệu, chưa có dự đoán model. Teacher/intent không là input inference.</p>'+''.join(cards)
    (output/'preview.html').write_text(content,encoding='utf-8')
    return len(chosen)


def write_json(path, value):
    with path.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2)


def build(config_path, output=None):
    root=project_root();config_path=Path(config_path).resolve();cfg=read_json(config_path)
    require(cfg['version']=='counterfactual_train_v1' and cfg['policy']=='agri_vla.revision.revision_action','Unknown recipe')
    require(cfg['scene_case_slots']==4 and cfg['extra_synonym_pairs_per_scene']==1 and cfg['keep_all_original_records'],'Unsupported manifest strategy')
    require(cfg['tokenizer_max_length']==96 and cfg['wording_pool']=='exact_observed_TRAIN_translation_pairs_only','Wrong input contract')
    require(cfg['sampling']['version']=='coverage_then_source_deficit_v1' and cfg['sampling']['unit']=='VI_EN_pair','Unsupported sampler')
    original_path=root/cfg['source_manifest'];lock=root/'data/manifests/wording_v1/dataset_lock.json'
    require(digest(lock)==cfg['dataset_lock_sha256'],'Dataset lock drift')
    require(digest(original_path)==read_json(lock)['manifest_sha256']['train.jsonl'],'TRAIN changed')
    stats=root/'data/lerobot/wording_v1/full/train_stats.json'
    require(digest(stats)==cfg['train_stats_sha256'],'Full TRAIN stats drift')
    protected=[lock,stats,root/'models/prepared/wording_v1/bundle_lock.json',
               root/'configs/readiness_balanced_v1.json',root/'configs/split_v0.yaml',
               root/'configs/language_wording_v1.yaml',root/'configs/policy_extension_v1.yaml',
               root/'data/manifests/source_splits/v0/split_lock.json',config_path]
    # Hash-only integrity snapshots of held-out manifests; never parse their text/labels/images.
    protected += [root/f'data/manifests/wording_v1/{s}.jsonl' for s in ('train','val','test')]
    bundle=root/'models/prepared/wording_v1'
    for name,expected in read_json(bundle/'bundle_lock.json')['files'].items():
        if name=='base_policy/model.safetensors':
            continue  # immutable base weights are not touched; avoid hashing a second model copy.
        require(digest(bundle/name)==expected,'Prepared processor/tokenizer contract drift')
        protected.append(bundle/name)
    before={str(p.relative_to(root)):digest(p) for p in protected}
    rows=read_jsonl(original_path);require(len(rows)==3600,'Wrong TRAIN count')
    split_lock=read_json(root/'data/manifests/source_splits/v0/split_lock.json')
    train_source=root/'data/manifests/source_splits/v0/train.sources.jsonl'
    require(digest(train_source)==split_lock['manifest_sha256']['train.sources.jsonl'],'Source TRAIN changed')
    allowed_groups=set(split_lock['assignments']['train'])
    require(not allowed_groups.intersection(split_lock['assignments']['val']+split_lock['assignments']['test']),'Group leakage')
    # Only TRAIN registry members are accepted; all generated wording is copied from actual TRAIN.
    language=load_unique_yaml((root/'configs/language_wording_v1.yaml').read_bytes(),context='language')
    extension=load_unique_yaml((root/'configs/policy_extension_v1.yaml').read_bytes(),context='override')
    families={f['id'] for pools in language['families'].values() for f in pools['train']}
    families.update(f['id'] for f in extension['templates']['train'])
    files={};image_labels={}
    for r in rows:
        require(r['template_family_id'] in families,'Held-out language family')
        require(set(r['source_groups'])<=allowed_groups,'Non-TRAIN source group')
        require(r['target_action']==revision_action(r['teacher_annotations']['wilt_scores'],r['intent'],r['intent_params'])==independent_action(r['teacher_annotations']['wilt_scores'],r['intent'],r['intent_params']),'Original label mismatch')
        for i,(zone,path) in enumerate(r['images'].items()):
            require(zone==f'zone{i+1}','Wrong zone ordering')
            expected=r['provenance']['source_sha256'][i]
            require(path not in files or files[path]==expected,'Source hash conflict')
            files[path]=expected
            image=r['source_image_ids'][i];label=r['teacher_annotations']['wilt_scores'][i]
            require(image not in image_labels or image_labels[image]==label,'Inconsistent image teacher')
            image_labels[image]=label
    require(all(digest(root/p)==expected for p,expected in files.items()),'Source image bytes drift')
    truth_checks=0
    for wilt in itertools.product(range(5),repeat=3):
        for intent,params in REQUESTS:
            require(revision_action(wilt,intent,params)==independent_action(wilt,intent,params),'Truth-table mismatch')
            truth_checks+=1
    view,scenes=build_view(rows,cfg['seed'])
    require(len({r['sample_id'] for r in view})==len(view),'Duplicate IDs')
    original_ids={r['sample_id'] for r in rows}
    require(original_ids=={r['sample_id'] for r in view if r['origin']=='original'},'Original record lost')
    observed={(r['intent'],request_key(r['intent'],r['intent_params']),r['language'],r['instruction']) for r in rows}
    for r in view:
        require((r['intent'],request_key(r['intent'],r['intent_params']),r['language'],r['instruction']) in observed,'Unobserved TRAIN wording')
        require(set(model_inputs(r))==set(cfg['model_input_keys']),'Model input leakage')
    require({image for r in view for image in r['source_image_ids']}==set(image_labels),'Source coverage changed')
    plan,sampling=sampling_plan(view,cfg['sampling']['scene_visits'],cfg['sampling']['max_visits_per_scene'],cfg['seed'],max_image_visits=cfg['sampling']['max_camera_visits_per_source_image'])
    control,control_sampling=sampling_plan([r for r in view if r['origin']=='original'],cfg['sampling']['scene_visits'],cfg['sampling']['max_visits_per_scene'],cfg['seed'],max_image_visits=cfg['sampling']['max_camera_visits_per_source_image'])
    require([(r['scene_id'],r['anchor_group'],r['anchor_image']) for r in plan]==[(r['scene_id'],r['anchor_group'],r['anchor_image']) for r in control],'Control scene/source sequence differs')
    output=Path(output).resolve() if output else root/cfg['output'];output.mkdir(parents=True,exist_ok=False)
    for name,rr in [('manifest.jsonl',view),('sampling-plan.jsonl',plan),('original-control-plan.jsonl',control)]:
        with (output/name).open('x',encoding='utf-8') as f:
            for r in rr:f.write(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n')
    write_json(output/'original-coverage.json',audit(rows));write_json(output/'view-coverage.json',audit(view))
    write_json(output/'sampling-coverage.json',sampling)
    write_json(output/'original-control-coverage.json',control_sampling)
    write_json(output/'image-hashes.json',files)
    with (output/'source-image-audit.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['image','teacher_wilt','original_camera_appearances','view_camera_appearances'])
        orig=Counter(i for r in rows for i in r['source_image_ids']);aug=Counter(i for r in view for i in r['source_image_ids'])
        for image in sorted(image_labels):w.writerow([image,image_labels[image],orig[image],aug[image]])
    example_count=preview(output,view)
    require(all(digest(root/p)==sha for p,sha in before.items()),'Protected input changed')
    require(all(digest(root/p)==sha for p,sha in files.items()),'Source images changed')
    qa={'status':'pass','version':cfg['version'],'source_records':len(rows),'view_records':len(view),
        'new_records':sum(r['origin']!='original' for r in view),'scenes':len(scenes),
        'verified_image_files':len(files),'truth_table_checks':truth_checks,'label_mismatches':0,
        'all_original_ids_preserved':True,'input_label_conflicts':0,'only_observed_train_wording':True,
        'family_pool_train_only':True,'all_sources_within_train_groups':True,
        'heldout_manifest_usage':'hash-only integrity; no parsed VAL/TEST records used',
        'independent_source_observations_added':0,'native_dataset_exported':False,
        'training_performed':False,'normalization_recomputed':False,'preview_scenes':example_count,
        'original_vs_view_same_scene_source_schedule':True,
        'tokenizer':'no new instruction strings; prior TRAIN tokenizer96 audit applies; no tokenizer file edited',
        'artifact_bytes_before_final_receipts':sum(p.stat().st_size for p in output.iterdir()),
        'protected_input_hashes':before,'sampling':sampling}
    write_json(output/'qa-report.json',qa)
    write_json(output/'artifact-lock.json',{'config_sha256':digest(config_path),'implementation_sha256':digest(Path(__file__)),
               'files':{p.name:digest(p) for p in output.iterdir() if p.is_file()}})
    size=sum(p.stat().st_size for p in output.iterdir());require(size<=cfg['max_artifact_mib']*1024**2,'Artifact budget exceeded')
    print(json.dumps({'status':'pass','records':len(view),'new_records':qa['new_records'],
                      'artifact_mib':size/1024**2,'output':str(output)},indent=2))
    return qa


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=project_root()/'configs/counterfactual_train_v1.json')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();build(args.config,args.output)


if __name__=='__main__':main()
