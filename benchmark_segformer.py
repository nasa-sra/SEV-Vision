from pathlib import Path
import time

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


def expand_segformer_to_4_channels(model: SegformerForSemanticSegmentation) -> None:
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


def load_depth_map(depth_path: Path, input_h: int, input_w: int) -> np.ndarray:
	if depth_path.suffix.lower() == ".npy":
		depth = np.load(depth_path).astype(np.float32)
		if depth.ndim == 3:
			depth = depth[..., 0]
		if depth.shape != (input_h, input_w):
			depth = cv2.resize(depth, (input_w, input_h), interpolation=cv2.INTER_NEAREST)
		depth = np.clip(depth, 0.0, 1.0)
		depth = depth * 2.0 - 1.0
		return depth

	depth = cv2.imread(str(depth_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
	if depth is None:
		raise ValueError(f"Failed to read depth map: {depth_path}")
	if depth.ndim == 3:
		depth = depth[..., 0]
	if depth.shape != (input_h, input_w):
		depth = cv2.resize(depth, (input_w, input_h), interpolation=cv2.INTER_NEAREST)
	depth = depth.astype(np.float32) / 100.0
	depth = np.clip(depth, 0.0, 80.0)
	depth = depth / 80.0
	depth[depth == 0] = 1
	depth = depth * 2.0 - 1.0
	return depth


def build_overlay(image: Image.Image, prediction: np.ndarray, num_classes: int, road_class_ids: list[int]) -> np.ndarray:
	palette = np.zeros((num_classes, 3), dtype=np.uint8)
	for class_id in range(num_classes):
		palette[class_id] = [
			(37 * class_id) % 256,
			(67 * class_id + 53) % 256,
			(97 * class_id + 101) % 256,
		]
	for class_id in road_class_ids:
		if 0 <= class_id < num_classes:
			palette[class_id] = [255, 0, 0]

	overlay = palette[prediction]
	image_np = np.array(image)
	alpha = 0.45
	return (image_np * (1 - alpha) + overlay * alpha).astype(np.uint8)


def main() -> None:
	device = "cuda" if torch.cuda.is_available() else "cpu"
	model_dir = Path("segformer_depth_finetuned_17")
	images_dir = Path("kitti_eval/images")
	depth_dir = Path("kitti_eval/depth")
	results_dir = Path("kitti_eval/results")
	results_dir.mkdir(parents=True, exist_ok=True)

	start_idx = 601
	end_idx = 836
	warmup_runs = 10

	processor = SegformerImageProcessor.from_pretrained(model_dir)
	model = SegformerForSemanticSegmentation.from_pretrained(model_dir).to(device)
	expand_segformer_to_4_channels(model)
	model.eval()

	id2label = {int(key): value for key, value in model.config.id2label.items()}
	road_class_ids = [int(class_id) for class_id, name in id2label.items() if "road" in str(name).lower()]

	timed_runs = 0
	total_model_time = 0.0
	processed = 0

	for image_idx in range(start_idx, end_idx + 1):
		image_path = images_dir / f"rgb_{image_idx:05d}.jpg"
		depth_path = depth_dir / f"depth_{image_idx:05d}.png"

		if not image_path.exists():
			raise FileNotFoundError(f"Missing image: {image_path}")
		if not depth_path.exists():
			raise FileNotFoundError(f"Missing depth: {depth_path}")

		image = Image.open(image_path).convert("RGB")
		inputs = processor(images=image, return_tensors="pt").to(device)
		pixel_values = inputs["pixel_values"]
		input_h, input_w = pixel_values.shape[2], pixel_values.shape[3]

		depth = load_depth_map(depth_path, input_h, input_w)
		depth_tensor = torch.from_numpy(depth).to(device=device, dtype=pixel_values.dtype).unsqueeze(0).unsqueeze(0)
		inputs["pixel_values"] = torch.cat([pixel_values, depth_tensor], dim=1)

		if device == "cuda":
			torch.cuda.synchronize()
		start_time = time.perf_counter()
		with torch.inference_mode():
			outputs = model(**inputs)
		if device == "cuda":
			torch.cuda.synchronize()
		model_time = time.perf_counter() - start_time

		if processed >= warmup_runs:
			total_model_time += model_time
			timed_runs += 1

		upsampled_logits = torch.nn.functional.interpolate(
			outputs.logits,
			size=image.size[::-1],
			mode="bilinear",
			align_corners=False,
		)
		prediction = upsampled_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
		blended = build_overlay(image, prediction, model.config.num_labels, road_class_ids)

		output_path = results_dir / f"rgb_{image_idx:05d}_segformer.png"
		Image.fromarray(blended).save(output_path)
		processed += 1
		print(f"Saved {output_path}")

	if timed_runs == 0:
		raise ValueError("Not enough runs to compute timing after the 10-image warmup.")

	average_model_time = total_model_time / timed_runs
	print(f"Processed images: {processed}")
	print(f"Warmup runs: {warmup_runs}")
	print(f"Timed runs: {timed_runs}")
	print(f"Average model inference time: {average_model_time:.6f} s")


if __name__ == "__main__":
	main()
