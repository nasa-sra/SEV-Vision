import gc
from pathlib import Path
import time
from typing import Dict

import cv2
import numpy as np
import torch
from transformers import AutoConfig, SegformerImageProcessor

from .trt_runner import TrtSegformerRunner


class TrtSegformerBackend:
    """TensorRT backend with fixed fast preprocessing."""

    def __init__(self, model_path: Path, trt_engine: Path, enable_timing: bool = False) -> None:
        self.runner = TrtSegformerRunner(trt_engine, enable_timing=enable_timing)
        self.preferred_hw = self.runner.get_preferred_hw()
        self.enable_timing = enable_timing
        self._resized_rgb_u8 = None
        self._rgb_float_hwc = None
        self._depth_raw_resized = None
        self._depth_invalid_mask = None
        self._packed_input = None
        self._preprocess_timing = {
            "rgb_resize": 0.0,
            "rgb_to_float": 0.0,
            "rgb_normalize": 0.0,
            "rgb_layout": 0.0,
            "depth_resize": 0.0,
            "depth_encode": 0.0,
            "pack_input": 0.0,
            "total": 0.0,
            "count": 0.0,
        }

        config = AutoConfig.from_pretrained(model_path)
        self.id2label = {int(class_id): str(name) for class_id, name in config.id2label.items()}
        self.num_classes = int(config.num_labels)

        processor = SegformerImageProcessor.from_pretrained(model_path)
        # SegFormer processor normalizes RGB as (x - mean) / std, where x in [0,1].
        self.mean = np.array(processor.image_mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(processor.image_std, dtype=np.float32).reshape(1, 1, 3)
        self.rgb_scale = 1.0 / (255.0 * self.std)
        self.rgb_bias = -self.mean / self.std

    def _ensure_buffers(self, input_h: int, input_w: int, depth_dtype: np.dtype) -> None:
        rgb_shape = (input_h, input_w, 3)
        depth_shape = (input_h, input_w)
        packed_shape = (1, 4, input_h, input_w)

        if self._resized_rgb_u8 is None or self._resized_rgb_u8.shape != rgb_shape:
            self._resized_rgb_u8 = np.empty(rgb_shape, dtype=np.uint8)
        if self._rgb_float_hwc is None or self._rgb_float_hwc.shape != rgb_shape:
            self._rgb_float_hwc = np.empty(rgb_shape, dtype=np.float32)
        if (
            self._depth_raw_resized is None
            or self._depth_raw_resized.shape != depth_shape
            or self._depth_raw_resized.dtype != depth_dtype
        ):
            self._depth_raw_resized = np.empty(depth_shape, dtype=depth_dtype)
        if self._depth_invalid_mask is None or self._depth_invalid_mask.shape != depth_shape:
            self._depth_invalid_mask = np.empty(depth_shape, dtype=np.bool_)
        if self._packed_input is None or self._packed_input.shape != packed_shape:
            self._packed_input = np.empty(packed_shape, dtype=np.float32)

    def get_preferred_hw(self):
        return self.preferred_hw

    def preprocess(
        self,
        color_rgb: np.ndarray,
        depth_raw: np.ndarray,
        depth_scale_m: float,
        max_depth_m: float,
    ) -> np.ndarray:
        total_t0 = time.perf_counter() if self.enable_timing else 0.0

        if self.preferred_hw is not None:
            input_h, input_w = self.preferred_hw
        else:
            # Fallback when profile info is unavailable.
            input_h, input_w = 720, 1280

        self._ensure_buffers(input_h=input_h, input_w=input_w, depth_dtype=depth_raw.dtype)

        t0 = time.perf_counter() if self.enable_timing else 0.0
        cv2.resize(
            color_rgb,
            (input_w, input_h),
            dst=self._resized_rgb_u8,
            interpolation=cv2.INTER_LINEAR,
        )
        if self.enable_timing:
            self._preprocess_timing["rgb_resize"] += time.perf_counter() - t0

        t0 = time.perf_counter() if self.enable_timing else 0.0
        np.multiply(self._resized_rgb_u8, self.rgb_scale, out=self._rgb_float_hwc, casting="unsafe")
        if self.enable_timing:
            self._preprocess_timing["rgb_to_float"] += time.perf_counter() - t0

        t0 = time.perf_counter() if self.enable_timing else 0.0
        np.add(self._rgb_float_hwc, self.rgb_bias, out=self._rgb_float_hwc)
        if self.enable_timing:
            self._preprocess_timing["rgb_normalize"] += time.perf_counter() - t0

        t0 = time.perf_counter() if self.enable_timing else 0.0
        self._packed_input[0, :3, :, :] = np.transpose(self._rgb_float_hwc, (2, 0, 1))
        if self.enable_timing:
            self._preprocess_timing["rgb_layout"] += time.perf_counter() - t0

        depth_raw_model = depth_raw
        if depth_raw_model.shape != (input_h, input_w):
            t0 = time.perf_counter() if self.enable_timing else 0.0
            cv2.resize(
                depth_raw_model,
                (input_w, input_h),
                dst=self._depth_raw_resized,
                interpolation=cv2.INTER_NEAREST,
            )
            depth_raw_model = self._depth_raw_resized
            if self.enable_timing:
                self._preprocess_timing["depth_resize"] += time.perf_counter() - t0

        t0 = time.perf_counter() if self.enable_timing else 0.0
        # Encode depth directly into output channel to avoid temporary arrays.
        depth_channel = self._packed_input[0, 3, :, :]
        np.multiply(depth_raw_model, depth_scale_m, out=depth_channel, casting="unsafe")
        np.less_equal(depth_channel, 0.0, out=self._depth_invalid_mask)
        np.clip(depth_channel, 0.0, max_depth_m, out=depth_channel)
        if max_depth_m != 1.0:
            np.multiply(depth_channel, 1.0 / max_depth_m, out=depth_channel)
        depth_channel[self._depth_invalid_mask] = 1.0
        np.multiply(depth_channel, 2.0, out=depth_channel)
        np.subtract(depth_channel, 1.0, out=depth_channel)
        if self.enable_timing:
            self._preprocess_timing["depth_encode"] += time.perf_counter() - t0

        t0 = time.perf_counter() if self.enable_timing else 0.0
        # Depth is already written in-place into the packed tensor.
        if self.enable_timing:
            self._preprocess_timing["pack_input"] += time.perf_counter() - t0
            self._preprocess_timing["total"] += time.perf_counter() - total_t0
            self._preprocess_timing["count"] += 1.0
        return self._packed_input

    def consume_preprocess_timing(self) -> Dict[str, float]:
        snapshot = dict(self._preprocess_timing)
        for key in self._preprocess_timing:
            self._preprocess_timing[key] = 0.0
        return snapshot

    def consume_infer_timing(self) -> Dict[str, float]:
        return self.runner.consume_infer_timing()

    def infer(self, prepared: np.ndarray) -> torch.Tensor:
        return self.runner.infer(prepared)

    def close(self) -> None:
        try:
            if self.runner is not None:
                self.runner.close()
        finally:
            self.runner = None
            self._resized_rgb_u8 = None
            self._rgb_float_hwc = None
            self._depth_raw_resized = None
            self._depth_invalid_mask = None
            self._packed_input = None
            self._preprocess_timing = {
                "rgb_resize": 0.0,
                "rgb_to_float": 0.0,
                "rgb_normalize": 0.0,
                "rgb_layout": 0.0,
                "depth_resize": 0.0,
                "depth_encode": 0.0,
                "pack_input": 0.0,
                "total": 0.0,
                "count": 0.0,
            }
            self.id2label = {}
            self.num_classes = 0
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()
