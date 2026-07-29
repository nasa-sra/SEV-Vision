import gc
from pathlib import Path
import time
from typing import Dict

import numpy as np
import torch


def trt_dtype_to_numpy(trt_module, dtype) -> np.dtype:
    if hasattr(trt_module, "nptype"):
        return np.dtype(trt_module.nptype(dtype))

    mapping = {
        trt_module.DataType.FLOAT: np.float32,
        trt_module.DataType.HALF: np.float16,
        trt_module.DataType.INT32: np.int32,
        trt_module.DataType.INT8: np.int8,
        trt_module.DataType.BOOL: np.bool_,
    }
    if dtype not in mapping:
        raise ValueError(f"Unsupported TensorRT dtype: {dtype}")
    return np.dtype(mapping[dtype])


def numpy_dtype_to_torch(np_dtype: np.dtype) -> torch.dtype:
    mapping = {
        np.dtype(np.float32): torch.float32,
        np.dtype(np.float16): torch.float16,
        np.dtype(np.int32): torch.int32,
        np.dtype(np.int8): torch.int8,
        np.dtype(np.bool_): torch.bool,
    }
    if np_dtype not in mapping:
        raise ValueError(f"Unsupported numpy dtype for torch conversion: {np_dtype}")
    return mapping[np_dtype]


class TrtSegformerRunner:
    """TensorRT runtime wrapper with support for TRT 8/9 and TRT 10+ Python APIs."""

    def __init__(self, engine_path: Path, verbose: bool = False, enable_timing: bool = False) -> None:
        try:
            import tensorrt as trt_module
        except Exception as exc:
            raise RuntimeError("TensorRT backend requires python package 'tensorrt'.") from exc

        if not torch.cuda.is_available():
            raise RuntimeError("TensorRT backend requires CUDA, but torch.cuda.is_available() is False.")

        self.trt = trt_module
        log_level = self.trt.Logger.VERBOSE if verbose else self.trt.Logger.INFO
        self.logger = self.trt.Logger(log_level)

        if not engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")

        runtime = self.trt.Runtime(self.logger)
        engine_bytes = engine_path.read_bytes()
        self.engine = runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine: {engine_path}")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context.")

        self.is_v3_api = hasattr(self.engine, "num_io_tensors") and hasattr(self.context, "execute_async_v3")

        self.input_name = None
        self.output_name = None
        self.input_index = None
        self.output_index = None

        if self.is_v3_api:
            for i in range(self.engine.num_io_tensors):
                name = self.engine.get_tensor_name(i)
                mode = self.engine.get_tensor_mode(name)
                if mode == self.trt.TensorIOMode.INPUT:
                    self.input_name = name
                elif mode == self.trt.TensorIOMode.OUTPUT:
                    self.output_name = name
        else:
            for i in range(self.engine.num_bindings):
                name = self.engine.get_binding_name(i)
                if self.engine.binding_is_input(i):
                    self.input_name = name
                    self.input_index = i
                else:
                    self.output_name = name
                    self.output_index = i

        if self.input_name is None or self.output_name is None:
            raise RuntimeError("Could not identify TensorRT input/output tensors.")

        if self.is_v3_api:
            self.input_dtype = trt_dtype_to_numpy(self.trt, self.engine.get_tensor_dtype(self.input_name))
            self.output_dtype = trt_dtype_to_numpy(self.trt, self.engine.get_tensor_dtype(self.output_name))
        else:
            self.input_dtype = trt_dtype_to_numpy(self.trt, self.engine.get_binding_dtype(self.input_index))
            self.output_dtype = trt_dtype_to_numpy(self.trt, self.engine.get_binding_dtype(self.output_index))

        self.input_torch_dtype = numpy_dtype_to_torch(self.input_dtype)
        self.output_torch_dtype = numpy_dtype_to_torch(self.output_dtype)

        self.profile_min_shape = None
        self.profile_opt_shape = None
        self.profile_max_shape = None
        self._load_profile_shapes()

        self.input_tensor = None
        self.output_tensor = None
        self.last_input_shape = None
        self.stream = torch.cuda.Stream()
        self.enable_timing = enable_timing
        self._infer_timing = {
            "host_cast": 0.0,
            "set_shape": 0.0,
            "alloc": 0.0,
            "h2d_copy": 0.0,
            "execute": 0.0,
            "wait_stream": 0.0,
            "total": 0.0,
            "count": 0.0,
        }

    def _load_profile_shapes(self) -> None:
        """Load profile 0 min/opt/max shapes when the API exposes them."""
        try:
            if self.is_v3_api and hasattr(self.engine, "get_tensor_profile_shape"):
                shapes = self.engine.get_tensor_profile_shape(self.input_name, 0)
                if shapes and len(shapes) == 3:
                    self.profile_min_shape = tuple(int(x) for x in shapes[0])
                    self.profile_opt_shape = tuple(int(x) for x in shapes[1])
                    self.profile_max_shape = tuple(int(x) for x in shapes[2])
            elif (not self.is_v3_api) and hasattr(self.engine, "get_profile_shape"):
                shapes = self.engine.get_profile_shape(0, self.input_index)
                if shapes and len(shapes) == 3:
                    self.profile_min_shape = tuple(int(x) for x in shapes[0])
                    self.profile_opt_shape = tuple(int(x) for x in shapes[1])
                    self.profile_max_shape = tuple(int(x) for x in shapes[2])
        except Exception:
            # Some TensorRT Python builds expose slightly different signatures.
            pass

    def get_preferred_hw(self):
        """Return preferred (height, width) from profile opt shape when available."""
        if self.profile_opt_shape is not None and len(self.profile_opt_shape) == 4:
            return int(self.profile_opt_shape[2]), int(self.profile_opt_shape[3])
        return None

    def close(self) -> None:
        try:
            self.input_tensor = None
            self.output_tensor = None
            self.context = None
            self.engine = None
            self.logger = None
            self.trt = None
            self.stream = None
            self.input_name = None
            self.output_name = None
            self.input_index = None
            self.output_index = None
            self.last_input_shape = None
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()

    def _set_input_shape(self, input_shape) -> None:
        if self.last_input_shape == tuple(input_shape):
            return

        if self.is_v3_api:
            if not self.context.set_input_shape(self.input_name, tuple(input_shape)):
                extra = ""
                if self.profile_min_shape is not None:
                    extra = (
                        f"; profile range min/opt/max="
                        f"{self.profile_min_shape}/{self.profile_opt_shape}/{self.profile_max_shape}"
                    )
                raise RuntimeError(f"Failed to set input shape to {input_shape}{extra}")
        else:
            if not self.context.set_binding_shape(self.input_index, tuple(input_shape)):
                extra = ""
                if self.profile_min_shape is not None:
                    extra = (
                        f"; profile range min/opt/max="
                        f"{self.profile_min_shape}/{self.profile_opt_shape}/{self.profile_max_shape}"
                    )
                raise RuntimeError(f"Failed to set binding shape to {input_shape}{extra}")

        self.last_input_shape = tuple(input_shape)

    def _get_output_shape(self):
        if self.is_v3_api:
            shape = tuple(self.context.get_tensor_shape(self.output_name))
        else:
            shape = tuple(self.context.get_binding_shape(self.output_index))
        if any(dim < 0 for dim in shape):
            raise RuntimeError(f"Unresolved TensorRT output shape: {shape}")
        return shape

    def infer(self, input_nchw: np.ndarray) -> torch.Tensor:
        if input_nchw.ndim != 4:
            raise ValueError(f"Expected NCHW input with 4 dims, got shape {input_nchw.shape}")

        total_t0 = time.perf_counter() if self.enable_timing else 0.0

        t0 = time.perf_counter() if self.enable_timing else 0.0
        host_input = np.ascontiguousarray(input_nchw.astype(self.input_dtype, copy=False))
        if self.enable_timing:
            self._infer_timing["host_cast"] += time.perf_counter() - t0

        t0 = time.perf_counter() if self.enable_timing else 0.0
        self._set_input_shape(host_input.shape)
        output_shape = self._get_output_shape()
        if self.enable_timing:
            self._infer_timing["set_shape"] += time.perf_counter() - t0

        t0 = time.perf_counter() if self.enable_timing else 0.0
        if self.input_tensor is None or tuple(self.input_tensor.shape) != tuple(host_input.shape):
            self.input_tensor = torch.empty(host_input.shape, device="cuda", dtype=self.input_torch_dtype)
        if self.output_tensor is None or tuple(self.output_tensor.shape) != tuple(output_shape):
            self.output_tensor = torch.empty(output_shape, device="cuda", dtype=self.output_torch_dtype)
        if self.enable_timing:
            self._infer_timing["alloc"] += time.perf_counter() - t0

        t0 = time.perf_counter() if self.enable_timing else 0.0
        input_cpu_tensor = torch.from_numpy(host_input)
        with torch.cuda.stream(self.stream):
            self.input_tensor.copy_(input_cpu_tensor, non_blocking=False)
        if self.enable_timing:
            self._infer_timing["h2d_copy"] += time.perf_counter() - t0

        t0 = time.perf_counter() if self.enable_timing else 0.0
        with torch.cuda.stream(self.stream):
            stream_handle = self.stream.cuda_stream
            if self.is_v3_api:
                self.context.set_tensor_address(self.input_name, int(self.input_tensor.data_ptr()))
                self.context.set_tensor_address(self.output_name, int(self.output_tensor.data_ptr()))
                if not self.context.execute_async_v3(stream_handle):
                    raise RuntimeError("TensorRT execute_async_v3 failed")
            else:
                bindings = [0] * self.engine.num_bindings
                bindings[self.input_index] = int(self.input_tensor.data_ptr())
                bindings[self.output_index] = int(self.output_tensor.data_ptr())
                if not self.context.execute_async_v2(bindings=bindings, stream_handle=stream_handle):
                    raise RuntimeError("TensorRT execute_async_v2 failed")
        if self.enable_timing:
            self._infer_timing["execute"] += time.perf_counter() - t0

        # Make outputs visible on the current stream without forcing device-wide sync.
        t0 = time.perf_counter() if self.enable_timing else 0.0
        torch.cuda.current_stream().wait_stream(self.stream)
        if self.enable_timing:
            self._infer_timing["wait_stream"] += time.perf_counter() - t0
            self._infer_timing["total"] += time.perf_counter() - total_t0
            self._infer_timing["count"] += 1.0
        return self.output_tensor

    def consume_infer_timing(self) -> Dict[str, float]:
        snapshot = dict(self._infer_timing)
        for key in self._infer_timing:
            self._infer_timing[key] = 0.0
        return snapshot
