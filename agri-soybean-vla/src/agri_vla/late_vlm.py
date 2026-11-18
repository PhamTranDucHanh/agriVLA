"""Typed late-VLM K/V adapter scope. Keeps native SmolVLA loss and action decoder."""
import json
from pathlib import Path
import re
import shutil

from .dataset import require, read_json
from .evaluation import model_batch, quantize_action
from .preflight import project_root
from .small_cases import expert_target_modules
from .training import file_sha, load_training_config, write_json, attach_checkpoint_contract

VERSION='late_vlm_kv_v1'
PREFIX='model.vlm_with_expert.vlm.model.text_model.layers.15.self_attn.'
KV=tuple(PREFIX+p for p in ('k_proj','v_proj'))


def recipe():
    root=project_root();path=root/'configs/late_vlm_kv_v1.json';c=read_json(path)
    require(c['version']==VERSION and c['rank']==c['alpha']==8,'Unknown LoRA recipe')
    require(c['active_vlm_layers']==16 and c['vlm_block']==15 and c['vlm_projections']==['k_proj','v_proj'],'Wrong active scope')
    require(c['time_sampling']=='endpoint_t1' and c['native_inference_steps']==1 and c['tokenizer_max_length']==96,'Input/objective drift')
    require(c['max_optimizer_updates']==300 and c['seed']==42 and c['batch_size']==4 and c['lr']==1e-4 and c['warmup_steps']==20,'Budget/training drift')
    return c|{'_path':str(path),'_sha256':file_sha(path)}


def target_regex(arm):
    require(arm in ('control','candidate'),'Unknown arm')
    expert=expert_target_modules({'expert_targets':'all_linear'})
    if arm=='control':return expert
    return '(?:'+expert+'|'+ '|'.join(re.escape(k) for k in KV)+')'


def action_target(values):
    require(isinstance(values,(list,tuple)) and len(values)==3 and all(type(v) is int and 0<=v<=3 for v in values),'Expected three integer action levels0..3')
    return list(values)


def ordered_images(images):
    require(set(images)=={'zone1','zone2','zone3'},'Exactly three named zones required')
    return [images[f'zone{i}'] for i in (1,2,3)]


def audit_parameters(policy,arm):
    c=recipe();rows=[]
    for name,p in policy.named_parameters():
        if not p.requires_grad:continue
        require('.lora_A.' in name or '.lora_B.' in name,'Full trainable weights forbidden')
        if '.vlm.' in name:
            require(arm=='candidate' and any(k+'.lora_' in name for k in KV),'Unexpected VLM trainable module')
        else:
            require('.lm_expert.' in name or any('.'+k+'.' in name for k in ('state_proj','action_in_proj','action_out_proj','action_time_mlp_in','action_time_mlp_out')),'Unexpected expert scope')
        require(str(p.dtype)=='torch.float32','FP32 adapter masters required')
        rows.append({'name':name,'parameters':p.numel(),'dtype':str(p.dtype)})
    count=sum(r['parameters'] for r in rows)
    require(count==c['trainable_parameters'][arm],'Trainable count differs')
    require(sum(r['parameters'] for r in rows if '.vlm.' in r['name'])==(20480 if arm=='candidate' else 0),'Wrong added VLM count')
    return {'trainable':count,'parameters':rows,'base_vision_frozen':True,'base_vlm_frozen':True,'added_vlm_lora_parameters':20480 if arm=='candidate' else 0}


def initialize(arm,*,device='cpu'):
    """Copy the same trained expert, introduce zero-B scope only in candidate."""
    import torch
    from peft import LoraConfig,get_peft_model_state_dict,set_peft_model_state_dict
    from safetensors.torch import load_file
    from .evaluation import load_model
    from .small_train import configure_training_time
    root=project_root();r=recipe();c=load_training_config(root/'configs/training_v0.yaml');source=root/r['source']
    require(file_sha(source/'adapter_model.safetensors')==r['source_adapter_sha256'],'Initialization source drift')
    require(file_sha(root/'data/lerobot/wording_v1/full/train_stats.json')==r['stats_sha256'],'Full TRAIN statistics drift')
    require(file_sha(source/'agri_train_stats.json')==r['saved_stats_sha256'],'Saved statistics drift')
    old,pre,post,_=load_model(c,root/c['bundle'],checkpoint=source,device=device)
    base=old.unload()  # removes adapters WITHOUT merging them into frozen base weights
    require(base.model.vlm_with_expert.num_vlm_layers==16,'Wrong active VLM layers')
    modules=dict(base.named_modules())
    for path in KV:
        require(path in modules and tuple(modules[path].weight.shape)==(320,960),'Pinned K/V path or shape differs')
    config=LoraConfig(r=8,lora_alpha=8,lora_dropout=0.,bias='none',target_modules=target_regex(arm),modules_to_save=[])
    policy=base.wrap_with_peft(peft_config=config)
    saved=load_file(source/'adapter_model.safetensors',device='cpu')
    missing=set_peft_model_state_dict(policy,saved,adapter_name='default')
    require(not missing.unexpected_keys,'Unexpected source adapter keys')
    actual=get_peft_model_state_dict(policy)
    require(saved.keys()<=actual.keys() and all(torch.equal(actual[k].detach().cpu(),v) for k,v in saved.items()),'Source expert not copied exactly')
    extra=set(actual)-set(saved)
    require(len(extra)==(4 if arm=='candidate' else 0),'Unexpected scope expansion')
    require(all(any(path+'.lora_' in k for path in KV) for k in extra),'Unexpected extra module')
    require(all(bool((actual[k]==0).all()) for k in extra if '.lora_B.' in k),'New B must start zero')
    configure_training_time(policy,{'time_sampling':'endpoint_t1'})
    policy.config.num_steps=1;policy.eval()
    report=audit_parameters(policy,arm)
    report.update(copied_source_tensors=len(saved),added_tensors=len(extra),source_expert_exact=True,new_B_zero=True)
    return policy,pre,post,report


def metadata(arm,*,step=0,kind='scope_expansion_fixture'):
    c=recipe()
    return {'version':VERSION,'arm':arm,'recipe_sha256':c['_sha256'],'implementation_sha256':file_sha(Path(__file__)),
            'runner_sha256':file_sha(Path(__file__).with_name('late_vlm_experiment.py')),
            'engine_sha256':file_sha(Path(__file__).with_name('small_train.py')),
            'source_adapter_sha256':c['source_adapter_sha256'],'stats_sha256':c['stats_sha256'],
            'target_modules':target_regex(arm),'trainable_parameters':c['trainable_parameters'][arm],
            'new_optimizer_updates':step,'kind':kind,'GPU_backward_verified':False,
            'fixture_is_resume_proof':False,'inference_inputs':c['inference_input_keys']}


def validate_metadata(value):
    c=recipe();require(value['version']==VERSION,'Unknown adapter variant')
    require(value['recipe_sha256']==c['_sha256'] and value['implementation_sha256']==file_sha(Path(__file__)),'Recipe/code drift')
    require(value['runner_sha256']==file_sha(Path(__file__).with_name('late_vlm_experiment.py')) and value['engine_sha256']==file_sha(Path(__file__).with_name('small_train.py')),'Training engine drift')
    require(value['target_modules']==target_regex(value['arm']) and value['trainable_parameters']==c['trainable_parameters'][value['arm']],'Scope mismatch')
    require(value['source_adapter_sha256']==c['source_adapter_sha256'] and value['stats_sha256']==c['stats_sha256'],'Source/stats mismatch')
    return value


def update_lock(folder):
    """Only for the newly created variant checkpoint, after adding typed metadata."""
    folder=Path(folder);path=folder/'agri_checkpoint_lock.json'
    lock=read_json(path)
    for name in ('late_vlm.json','small_experiment.json'):
        if (folder/name).is_file():lock['files'][name]=file_sha(folder/name)
    path.write_text(json.dumps(lock,indent=2),encoding='utf-8')


def save_fixture(policy,folder,arm):
    """No optimizer state: a scope/reload fixture, not a trained-resume certificate."""
    root=project_root();r=recipe();folder=Path(folder).resolve();folder.mkdir(parents=True,exist_ok=False)
    policy.save_pretrained(folder);policy.config.save_pretrained(folder)
    source=root/r['source']
    for path in source.glob('policy_*'):
        if path.is_file():shutil.copyfile(path,folder/path.name)
    c=load_training_config(root/'configs/training_v0.yaml');attach_checkpoint_contract(root/c['bundle'],folder)
    write_json(folder/'late_vlm.json',metadata(arm))
    # An explicit unknown version makes legacy small_cases resume/loading fail closed.
    write_json(folder/'small_experiment.json',{'version':VERSION,'arm':arm,'fixture':True})
    update_lock(folder)


def load_policy(folder,*,device='cpu'):
    import torch
    from peft import PeftModel,PeftConfig
    from .evaluation import check_bundle
    from .training import runtime,assets,asset_path,BASE,BACKBONE,load_processors
    from .small_train import configure_training_time
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    folder=Path(folder).resolve();value=validate_metadata(read_json(folder/'late_vlm.json'))
    for name,sha in read_json(folder/'agri_checkpoint_lock.json')['files'].items():require(file_sha(folder/name)==sha,f'Checkpoint drift:{name}')
    root=project_root();c=load_training_config(root/'configs/training_v0.yaml');bundle=root/c['bundle']
    runtime(c,device=device);assets(c);check_bundle(c,bundle)
    require(read_json(folder/'agri_provenance.json')==read_json(bundle/'provenance.json'),'Provenance differs')
    require(read_json(folder/'agri_schema.json')==read_json(bundle/'schema.json'),'Schema differs')
    require(file_sha(root/'data/lerobot/wording_v1/full/train_stats.json')==recipe()['stats_sha256'],'Full TRAIN stats drift')
    require(file_sha(folder/'agri_train_stats.json')==recipe()['saved_stats_sha256'],'Wrong saved statistics')
    cfg=SmolVLAConfig.from_pretrained(folder)
    from .training import CAMERAS
    require(list(cfg.image_features)==list(CAMERAS) and cfg.action_feature.shape==(3,) and cfg.robot_state_feature.shape==(3,),'Camera/action/state schema differs')
    require(cfg.chunk_size==cfg.n_action_steps==1 and cfg.max_state_dim==cfg.max_action_dim==32,'Horizon/padding differs')
    require({k:v.value for k,v in cfg.normalization_mapping.items()}=={'VISUAL':'IDENTITY','STATE':'IDENTITY','ACTION':'MEAN_STD'},'Normalization differs')
    require(cfg.num_steps==1 and cfg.tokenizer_max_length==96 and cfg.resize_imgs_with_padding==(256,256),'Input/sampling drift')
    require(cfg.num_vlm_layers==16 and cfg.train_expert_only and cfg.freeze_vision_encoder,'Frozen base scope differs')
    cfg.device=device;cfg.use_amp=device=='cuda';cfg.use_peft=False;cfg.pretrained_path=bundle/'base_policy';cfg.vlm_model_name=str(asset_path(c,BACKBONE))
    if device=='cuda':
        torch.backends.cudnn.deterministic=False;torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=True
    base=SmolVLAPolicy.from_pretrained(asset_path(c,BASE),config=cfg,local_files_only=True,strict=True)
    ac=PeftConfig.from_pretrained(folder)
    require(ac.r==ac.lora_alpha==8 and ac.target_modules==target_regex(value['arm']) and not ac.modules_to_save and ac.bias=='none','PEFT config differs')
    policy=PeftModel.from_pretrained(base,folder,is_trainable=True);audit_parameters(policy,value['arm'])
    configure_training_time(policy,{'time_sampling':'endpoint_t1'})
    pre,post=load_processors(c,folder,cfg,device=device,tokenizer_dir=folder/'tokenizer')
    from lerobot.processor.normalize_processor import NormalizerProcessorStep,UnnormalizerProcessorStep
    wanted=read_json(bundle/'train_stats.json')['action']
    for pipe,kind in ((pre,NormalizerProcessorStep),(post,UnnormalizerProcessorStep)):
        normalizer=next(s for s in pipe.steps if isinstance(s,kind));saved=normalizer.state_dict()
        require(all(torch.allclose(saved[f'action.{key}'].cpu(),torch.tensor(wanted[key]),atol=1e-7) for key in ('mean','std')),'Saved processor stats differ')
    policy.eval();return policy,pre,post


def resume_contract(checkpoint,arm):
    cp=Path(checkpoint).resolve();value=validate_metadata(read_json(cp/'pretrained_model/late_vlm.json'))
    require(value['arm']==arm and value['kind']=='native_trained_checkpoint' and value['new_optimizer_updates']>0,'Fixture or wrong arm cannot resume')
    names=('optimizer_state.safetensors','optimizer_param_groups.json','scheduler_state.json','rng_state.safetensors','training_step.json','amp_scaler_state.json')
    require(all((cp/'training_state'/name).is_file() for name in names),'Incomplete native training state')
    require(read_json(cp/'training_state/training_step.json')['step']==value['new_optimizer_updates'],'Step mismatch')
    return value
