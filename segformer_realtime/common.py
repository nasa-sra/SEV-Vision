import numpy as np
import torch
from transformers import SegformerForSemanticSegmentation


def expand_segformer_to_4_channels(model: SegformerForSemanticSegmentation) -> None:
    """Expand the first SegFormer patch embedding from 3->4 input channels."""
    first_patch_embedding = model.segformer.stages[0].patch_embeddings
    old_proj = first_patch_embedding.proj

    if old_proj.in_channels == 4:
        model.config.num_channels = 4
        print("Already has 4 channels! A-OK")
        return

    if old_proj.in_channels != 3:
        raise ValueError(f"Expected first projection to have 3 input channels, got {old_proj.in_channels}")

    new_proj = torch.nn.Conv2d(
        in_channels=4,
        out_channels=old_proj.out_channels,
        kernel_size=old_proj.kernel_size,
        stride=old_proj.stride,
        padding=old_proj.padding,
        bias=old_proj.bias is not None,
    )

    with torch.no_grad():
        new_proj.weight[:, :3, :, :] = old_proj.weight
        new_proj.weight[:, 3:4, :, :] = old_proj.weight[:, 1:2, :, :]
        if old_proj.bias is not None:
            new_proj.bias.copy_(old_proj.bias)

    new_proj = new_proj.to(device=old_proj.weight.device, dtype=old_proj.weight.dtype)
    first_patch_embedding.proj = new_proj
    model.config.num_channels = 4


def encode_depth_raw_to_model_range(
    depth_raw: np.ndarray,
    depth_scale_m: float,
    max_depth_m: float = 80.0,
) -> np.ndarray:
    """Convert raw depth units to model range [-1, 1], with invalid mapped to +1."""
    depth_m = depth_raw.astype(np.float32) * depth_scale_m
    invalid_mask = depth_m <= 0.0
    depth01 = np.clip(depth_m, 0.0, max_depth_m) / max_depth_m
    depth01[invalid_mask] = 1.0
    return depth01 * 2.0 - 1.0
