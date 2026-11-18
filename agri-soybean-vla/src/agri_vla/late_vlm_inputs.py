"""Raw deploy input builder: three named RGB images, instruction, zero state."""
from pathlib import Path

from .dataset import require
from .late_vlm import ordered_images
from .training import file_sha,CAMERAS
from .lerobot_io import transformed_image


def raw_batch(images,instruction):
    import torch
    require(isinstance(instruction,str) and instruction.strip(),'Nonempty instruction required')
    arrays=[transformed_image(Path(path),file_sha(Path(path))) for path in ordered_images(images)]
    batch={camera:torch.from_numpy(array.copy()).permute(2,0,1).float()[None,None]/255.
           for camera,array in zip(CAMERAS,arrays,strict=True)}
    batch.update({'observation.state':torch.zeros(1,1,3,dtype=torch.float32),'task':[instruction]})
    return batch
