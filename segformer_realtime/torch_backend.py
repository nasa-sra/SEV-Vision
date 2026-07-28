from pathlib import Path
from typing import Any, Dict

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

from .common import encode_depth_raw_to_model_range, expand_segformer_to_4_channels


class TorchSegformerBackend:
    """Torch backend for SegFormer inference and preprocessing."""

    def __init__(self, model_path: Path, device: str, use_fp16: bool) -> None:
        self.device = device
        self.use_fp16 = use_fp16
        self.inference_dtype = torch.float16 if use_fp16 else torch.float32
        self.processor = SegformerImageProcessor.from_pretrained(model_path)

        model = SegformerForSemanticSegmentation.from_pretrained(model_path).to(device)
        expand_segformer_to_4_channels(model)
        if use_fp16:
            model = model.half()

        if hasattr(torch, "compile"):
            try:
                model = torch.compile(model, mode="reduce-overhead")
                print("Enabled torch.compile with mode='reduce-overhead'.")
            except Exception as compile_err:
                print(f"torch.compile unavailable, continuing without it: {compile_err}")

        model.eval()
        self.model = model
        self.id2label = {int(class_id): str(name) for class_id, name in model.config.id2label.items()}
        self.num_classes = int(model.config.num_labels)

    def get_preferred_hw(self):
        return None

    def preprocess(
        self,
        color_rgb: np.ndarray,
        depth_raw: np.ndarray,
        depth_scale_m: float,
        max_depth_m: float,
    ) -> Dict[str, Any]:
        inputs = self.processor(images=Image.fromarray(color_rgb), return_tensors="pt")
        pixel_values_cpu = inputs["pixel_values"]
        input_h, input_w = pixel_values_cpu.shape[2], pixel_values_cpu.shape[3]

        depth_raw_model = depth_raw
        if depth_raw_model.shape != (input_h, input_w):
            depth_raw_model = cv2.resize(depth_raw_model, (input_w, input_h), interpolation=cv2.INTER_NEAREST)

        depth = encode_depth_raw_to_model_range(
            depth_raw=depth_raw_model,
            depth_scale_m=depth_scale_m,
            max_depth_m=max_depth_m,
        )
        return {"pixel_values_cpu": pixel_values_cpu, "depth": depth}

    def infer(self, prepared: Dict[str, Any]) -> torch.Tensor:
        pixel_values_cpu = prepared["pixel_values_cpu"]
        depth = prepared["depth"]

        if self.device == "cuda":
            pixel_values = pixel_values_cpu.pin_memory().to(
                device=self.device,
                dtype=self.inference_dtype,
                non_blocking=True,
            )
        else:
            pixel_values = pixel_values_cpu.to(device=self.device, dtype=self.inference_dtype)

        depth_tensor_cpu = torch.from_numpy(depth).unsqueeze(0).unsqueeze(0)
        if self.device == "cuda":
            depth_tensor = depth_tensor_cpu.pin_memory().to(
                device=self.device,
                dtype=self.inference_dtype,
                non_blocking=True,
            )
        else:
            depth_tensor = depth_tensor_cpu.to(device=self.device, dtype=self.inference_dtype)

        model_inputs = {"pixel_values": torch.cat([pixel_values, depth_tensor], dim=1)}

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_fp16):
                outputs = self.model(**model_inputs)
        return outputs.logits

    def close(self) -> None:
        return
