import argparse
from pathlib import Path

import torch
from transformers import SegformerForSemanticSegmentation


def expand_segformer_to_4_channels(model: SegformerForSemanticSegmentation) -> None:
	"""Expand the first SegFormer patch embedding from 3 to 4 input channels."""
	first_patch_embedding = model.segformer.stages[0].patch_embeddings
	old_proj = first_patch_embedding.proj

	if old_proj.in_channels == 4:
		model.config.num_channels = 4
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


class SegformerOnnxWrapper(torch.nn.Module):
	"""Expose a simple tensor-in, tensor-out interface for ONNX export."""

	def __init__(self, model: SegformerForSemanticSegmentation) -> None:
		super().__init__()
		self.model = model

	def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
		return self.model(pixel_values=pixel_values).logits


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Export segformer_depth_finetuned_17 to ONNX.")
	parser.add_argument(
		"--model-path",
		type=Path,
		default=Path("segformer_depth_finetuned_17"),
		help="Path to the trained SegFormer checkpoint folder.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=Path("segformer_depth_finetuned_17.onnx"),
		help="Output ONNX file path.",
	)
	parser.add_argument("--height", type=int, default=720, help="Dummy export height.")
	parser.add_argument("--width", type=int, default=1280, help="Dummy export width.")
	parser.add_argument("--opset", type=int, default=18, help="ONNX opset version.")
	parser.add_argument(
		"--static-shape",
		action="store_true",
		help="Export fixed-shape ONNX (no dynamic axes). Recommended for fixed-size TensorRT engines.",
	)
	parser.add_argument(
		"--fp16",
		action="store_true",
		help="Export FP16 ONNX weights (use with CUDA device).",
	)
	parser.add_argument(
		"--device",
		type=str,
		default="cuda" if torch.cuda.is_available() else "cpu",
		help="Device used to run the export.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	if not args.model_path.exists():
		raise FileNotFoundError(f"Model folder not found: {args.model_path}")

	device = torch.device(args.device)
	model = SegformerForSemanticSegmentation.from_pretrained(args.model_path).to(device)
	expand_segformer_to_4_channels(model)
	if args.fp16:
		if device.type != "cuda":
			raise ValueError("--fp16 export requires --device cuda")
		model = model.half()
	model.eval()

	wrapper = SegformerOnnxWrapper(model)
	wrapper.eval()
	# torch.compile improves eager runtime but does not transfer into ONNX graphs.
	# Keep export uncompiled and rely on TensorRT for graph-level optimizations.
	dummy_input = torch.randn(1, 4, args.height, args.width, device=device, dtype=next(model.parameters()).dtype)

	args.output.parent.mkdir(parents=True, exist_ok=True)

	export_kwargs = {
		"export_params": True,
		"opset_version": args.opset,
		"do_constant_folding": True,
		"input_names": ["pixel_values"],
		"output_names": ["logits"],
		"dynamo": False,
	}

	if not args.static_shape:
		export_kwargs["dynamic_axes"] = {
			"pixel_values": {0: "batch", 2: "height", 3: "width"},
			"logits": {0: "batch", 2: "logits_height", 3: "logits_width"},
		}

	torch.onnx.export(
		wrapper,
		(dummy_input,),
		args.output,
		**export_kwargs,
	)

	print(f"Exported ONNX model to {args.output}")


if __name__ == "__main__":
	main()
