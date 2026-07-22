from PIL import Image
import torch
import numpy as np
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


def main() -> None:
	device = "cuda" if torch.cuda.is_available() else "cpu"

	model_name = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"
	images_dir = Path("kitti_eval/images")
	gt_masks_dir = Path("kitti_eval/masks_cityscapes")
	sampling_stride = 10
	ignore_index = 255

	processor = SegformerImageProcessor.from_pretrained(model_name)
	model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(device)
	model.eval()

	num_classes = model.config.num_labels

	cm_metric = MulticlassConfusionMatrix(
		num_classes=num_classes,
		ignore_index=ignore_index,
	).to(device)

	image_paths = sorted(images_dir.glob("rgb_*.jpg"))
	if not image_paths:
		raise FileNotFoundError(f"No input images found in {images_dir}")

	sampled_image_paths = image_paths[::sampling_stride]
	if not sampled_image_paths:
		raise ValueError("Sampling produced zero images to evaluate.")

	evaluated = 0
	skipped_missing_gt = 0

	for image_path in sampled_image_paths:
		image_stem = image_path.stem
		image_idx = image_stem.split("_")[-1]
		gt_mask_path = gt_masks_dir / f"classgt_{image_idx}.png"

		if not gt_mask_path.exists():
			skipped_missing_gt += 1
			continue

		image = Image.open(image_path).convert("RGB")
		inputs = processor(images=image, return_tensors="pt").to(device)

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
