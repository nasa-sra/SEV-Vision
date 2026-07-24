from PIL import Image
import torch
import numpy as np
import cv2
from pathlib import Path
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from torchmetrics.classification import (
	MulticlassAccuracy,
	MulticlassConfusionMatrix,
	MulticlassJaccardIndex,
)


def compute_fwiou_from_cm(cm: np.ndarray) -> float:
	tp = np.diag(cm).astype(np.float64)
	support = cm.sum(axis=1).astype(np.float64)
	pred_count = cm.sum(axis=0).astype(np.float64)
	total = cm.sum().astype(np.float64)

	denom_iou = support + pred_count - tp
	iou = np.divide(tp, denom_iou, out=np.full_like(tp, np.nan), where=denom_iou > 0)
	freq = np.divide(support, total, out=np.zeros_like(support), where=total > 0)
	return float(np.nansum(freq * iou))


def expand_segformer_to_4_channels(model: SegformerForSemanticSegmentation) -> None:
	"""Expand the first SegFormer patch embedding from 3->4 input channels.

	The new 4th channel is initialized from the pretrained green channel weights.
	"""
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


def main() -> None:
	device = "cuda" if torch.cuda.is_available() else "cpu"

	model_prefix = "segformer_depth_finetuned_"
	images_dir = Path("kitti_eval/images")
	depth_dir = Path("kitti_eval/depth")
	gt_masks_dir = Path("kitti_eval/masks_cityscapes")
	preview_image_path = Path("realsense_data/DirtRoad_Color.png")
	preview_depth_path = Path("realsense_data/DirtRoad_Depth_norm.npy")
	results_dir = Path("results")
	run_preview_only = True
	sampling_stride = 10
	ignore_index = 255

	image_paths = sorted(images_dir.glob("rgb_*.jpg"))
	if not image_paths:
		raise FileNotFoundError(f"No input images found in {images_dir}")

	sampled_image_paths = image_paths[::sampling_stride]
	if not sampled_image_paths:
		raise ValueError("Sampling produced zero images to evaluate.")

	if run_preview_only:
		if not preview_image_path.exists():
			raise FileNotFoundError(f"Preview image not found: {preview_image_path}")
		if not preview_depth_path.exists():
			raise FileNotFoundError(f"Preview depth not found: {preview_depth_path}")
		results_dir.mkdir(parents=True, exist_ok=True)

		model_paths = sorted(
			[
				path for path in Path(".").glob(f"{model_prefix}*")
				if path.is_dir() and path.name[len(model_prefix):].isdigit()
			],
			key=lambda path: int(path.name[len(model_prefix):]),
		)
		if not model_paths:
			raise FileNotFoundError(f"No model folders found matching {model_prefix}*")

		image = Image.open(preview_image_path).convert("RGB")
		print(f"Preview image: {preview_image_path}")
		print(f"Preview depth: {preview_depth_path}")

		for model_path in model_paths:
			model_epoch = int(model_path.name[len(model_prefix):])
			processor = SegformerImageProcessor.from_pretrained(model_path)
			model = SegformerForSemanticSegmentation.from_pretrained(model_path).to(device)
			expand_segformer_to_4_channels(model)
			model.eval()

			inputs = processor(images=image, return_tensors="pt").to(device)
			pixel_values = inputs["pixel_values"]
			input_h, input_w = pixel_values.shape[2], pixel_values.shape[3]

			if preview_depth_path.suffix.lower() == ".npy":
				depth = np.load(preview_depth_path).astype(np.float32)
				if depth.ndim == 3:
					depth = depth[..., 0]
				if depth.shape != (input_h, input_w):
					depth = cv2.resize(depth, (input_w, input_h), interpolation=cv2.INTER_NEAREST)
				depth = np.clip(depth, 0.0, 1.0)
				depth = depth * 2.0 - 1.0
			else:
				depth = cv2.imread(str(preview_depth_path), cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
				if depth is None:
					raise ValueError(f"Failed to read depth map: {preview_depth_path}")
				if depth.ndim == 3:
					depth = depth[..., 0]
				if depth.shape != (input_h, input_w):
					depth = cv2.resize(depth, (input_w, input_h), interpolation=cv2.INTER_NEAREST)
				depth = depth.astype(np.float32) / 100.0
				depth = np.clip(depth, 0.0, 80.0)
				depth = depth / 80.0
				depth[depth == 0] = 1
				depth = depth * 2.0 - 1.0

			depth_tensor = torch.from_numpy(depth).to(device=device, dtype=pixel_values.dtype)
			depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0)
			inputs["pixel_values"] = torch.cat([pixel_values, depth_tensor], dim=1)

			with torch.inference_mode():
				outputs = model(**inputs)

			upsampled_logits = torch.nn.functional.interpolate(
				outputs.logits,
				size=image.size[::-1],
				mode="bilinear",
				align_corners=False,
			)

			id2label = model.config.id2label
			road_class_ids = [
				int(class_id) for class_id, name in id2label.items()
				if "road" in str(name).lower()
			]

			prediction = upsampled_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
			probabilities = torch.softmax(upsampled_logits, dim=1)
			if road_class_ids:
				road_confidence = probabilities[:, road_class_ids, :, :].sum(dim=1)[0].cpu().numpy()
			else:
				road_confidence = np.zeros(image.size[::-1], dtype=np.float32)
			num_classes = model.config.num_labels
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
			blended = (image_np * (1 - alpha) + overlay * alpha).astype(np.uint8)
			road_confidence_u8 = np.clip(road_confidence * 255.0, 0.0, 255.0).astype(np.uint8)
			road_heatmap = cv2.applyColorMap(road_confidence_u8, cv2.COLORMAP_JET)
			road_heatmap = cv2.cvtColor(road_heatmap, cv2.COLOR_BGR2RGB)
			road_overlay = (image_np * 0.45 + road_heatmap * 0.55).astype(np.uint8)

			preview_output_path = results_dir / f"depth_{model_epoch}.png"
			Image.fromarray(blended).save(preview_output_path)
			print(f"Saved {model_path} preview to {preview_output_path}")
		return

	model_name = model_prefix.rstrip("_")
	processor = SegformerImageProcessor.from_pretrained(model_name)
	model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(device)
	expand_segformer_to_4_channels(model)
	model.eval()

	num_classes = model.config.num_labels

	cm_metric = MulticlassConfusionMatrix(
		num_classes=num_classes,
		ignore_index=ignore_index,
	).to(device)

	evaluated = 0
	skipped_missing_gt = 0
	skipped_missing_depth = 0

	for image_path in sampled_image_paths:
		image_stem = image_path.stem
		image_idx = image_stem.split("_")[-1]
		gt_mask_path = gt_masks_dir / f"classgt_{image_idx}.png"
		depth_png_filename = str(depth_dir / f"depth_{image_idx}.png")

		if not gt_mask_path.exists():
			skipped_missing_gt += 1
			continue

		if not Path(depth_png_filename).exists():
			skipped_missing_depth += 1
			continue

		image = Image.open(image_path).convert("RGB")
		inputs = processor(images=image, return_tensors="pt").to(device)

		depth = cv2.imread(depth_png_filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
		if depth is None:
			skipped_missing_depth += 1
			continue

		if depth.ndim == 3:
			depth = depth[..., 0]

		pixel_values = inputs["pixel_values"]
		input_h, input_w = pixel_values.shape[2], pixel_values.shape[3]

		if depth.shape != (input_h, input_w):
			depth = cv2.resize(depth, (input_w, input_h), interpolation=cv2.INTER_NEAREST)

		depth = depth.astype(np.float32) / 100.0
		depth = np.clip(depth, 0.0, 80.0)
		depth = depth / 80.0
		depth[depth == 0] = 1
		depth = depth * 2.0 - 1.0
		depth_tensor = torch.from_numpy(depth).to(device=device, dtype=pixel_values.dtype)
		depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0)
		inputs["pixel_values"] = torch.cat([pixel_values, depth_tensor], dim=1)

		with torch.inference_mode():
			outputs = model(**inputs)

		logits = outputs.logits

		upsampled_logits = torch.nn.functional.interpolate(
			logits,
			size=image.size[::-1],
			mode="bilinear",
			align_corners=False,
		)

		# Keep prediction on GPU
		prediction = upsampled_logits.argmax(dim=1)[0]

		# Load GT
		gt_mask = torch.from_numpy(
			np.array(Image.open(gt_mask_path), dtype=np.int64)
		).to(device)

		if gt_mask.ndim == 3:
			gt_mask = gt_mask[..., 0]

		if prediction.shape != gt_mask.shape:
			raise ValueError(
				f"Prediction shape {prediction.shape} does not match GT shape {gt_mask.shape}"
			)

		cm_metric.update(prediction, gt_mask)

		evaluated += 1

	if evaluated == 0:
		raise ValueError("No image-mask pairs were evaluated.")

	cm = cm_metric.compute().cpu().numpy()

	tp = np.diag(cm).astype(np.float64)
	support = cm.sum(axis=1).astype(np.float64)
	pred_count = cm.sum(axis=0).astype(np.float64)

	# Per-class IoU
	denom_iou = support + pred_count - tp
	per_class_iou = np.divide(
		tp,
		denom_iou,
		out=np.full_like(tp, np.nan),
		where=denom_iou > 0,
	)

	# Mean IoU
	miou = float(np.nanmean(per_class_iou))

	# Pixel accuracy
	pixel_accuracy = float(tp.sum() / cm.sum())

	# Mean accuracy
	class_acc = np.divide(
		tp,
		support,
		out=np.zeros_like(tp),
		where=support > 0,
	)

	valid_classes = support > 0
	mean_accuracy = float(class_acc[valid_classes].mean())

	# Frequency weighted IoU
	fwiou = compute_fwiou_from_cm(cm)

	support = cm.sum(axis=1)

	id2label = {int(k): v for k, v in model.config.id2label.items()}

	print(f"Images dir: {images_dir}")
	print(f"GT dir: {gt_masks_dir}")
	print(f"Sampling stride: every {sampling_stride}th image")
	print(f"Sampled images: {len(sampled_image_paths)}")
	print(f"Evaluated pairs: {evaluated}")
	print(f"Skipped (missing GT): {skipped_missing_gt}")
	print(f"Skipped (missing depth): {skipped_missing_depth}")
	print(f"Valid pixels: {int(cm.sum())}")
	print(f"Ignore index: {ignore_index}")
	print("\nAggregate metrics")
	print(f"Pixel Accuracy: {pixel_accuracy:.4f}")
	print(f"Mean Accuracy:  {mean_accuracy:.4f}")
	print(f"mIoU:           {miou:.4f}")
	print(f"FWIoU:          {fwiou:.4f}")

	print("\nPer-class IoU (classes present in GT)")
	for class_id in range(num_classes):
		if support[class_id] > 0:
			class_name = id2label.get(class_id, f"class_{class_id}")
			print(f"{class_id:2d} {class_name:15s} IoU={per_class_iou[class_id]:.4f} support={int(support[class_id])}")


if __name__ == "__main__":
	main()
