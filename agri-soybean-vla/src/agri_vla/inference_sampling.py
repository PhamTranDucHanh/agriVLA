"""Small Monte Carlo mean for deterministic singleton-action inference.

Native independent Gaussian noise, no chosen noise bank or teacher-dependent rule.
Average normalized actions before inverse normalization/clip/quantization.
"""
def configure_inference(policy, profile):
    from .dataset import require
    count = profile.get('inference_samples', 1)
    require(count in (1,2), 'Only native single sample or fixed two-sample mean supported')
    base=policy.get_base_model()
    if count == 1:
        if hasattr(base,'_agri_native_get_action_chunk'):
            base._get_action_chunk=base._agri_native_get_action_chunk
            base._agri_inference_samples=1
        return
    if getattr(base,'_agri_inference_samples',1)==count:return
    require(base.config.chunk_size==base.config.n_action_steps==1,'Mean inference is for singleton actions')
    # Both predict_action_chunk AND select_action call this native common path.
    native=base._get_action_chunk
    base._agri_native_get_action_chunk=native
    import torch
    def mean_chunk(batch,noise=None,**kwargs):
        batch_size=batch['observation.state'].shape[0]
        device=batch['observation.state'].device
        shape=(1,base.config.chunk_size,base.config.max_action_dim)
        first=noise if noise is not None else base.model.sample_noise(shape,device).repeat(batch_size,1,1)
        # Draw all noise before inference so computational kernels do not affect pairing.
        second=base.model.sample_noise(shape,device).repeat(batch_size,1,1)
        return (native(batch,noise=first,**kwargs)+native(batch,noise=second,**kwargs))/2
    base._get_action_chunk=mean_chunk
    base._agri_inference_samples=count
