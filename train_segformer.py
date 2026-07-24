from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchmetrics.classification import MulticlassJaccardIndex
from transformers import (
	SegformerForSemanticSegmentation,
	SegformerImageProcessor,
	get_cosine_schedule_with_warmup,
)


def expand_segformer_to_4_channels(model: SegformerForSemanticSegmentation) -> None:
	"""Expand first SegFormer patch embedding from 3 to 4 channels.

	The 4th channel is initialized from the mean of the pretrained RGB weights.
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
		new_proj.weight[:, 3:4, :, :] = old_proj.weight.mean(dim=1, keepdim=True)
		if old_proj.bias is not None:
			new_proj.bias.copy_(old_proj.bias)

	new_proj = new_proj.to(device=old_proj.weight.device, dtype=old_proj.weight.dtype)
	first_patch_embedding.proj = new_proj
	model.config.num_channels = 4


def configure_trainable_parameters(
	model: SegformerForSemanticSegmentation,
	encoder_stage0_patch_lr: float,
	decoder_lr: float,
) -> list[dict[str, object]]:
	for parameter in model.parameters():
		parameter.requires_grad = False

	for parameter in model.segformer.stages[0].patch_embeddings.parameters():
		parameter.requires_grad = True

	for parameter in model.decode_head.parameters():
		parameter.requires_grad = True

	stage0_patch_params = [
		parameter for parameter in model.segformer.stages[0].patch_embeddings.parameters()
		if parameter.requires_grad
	]
	decoder_params = [
		parameter for parameter in model.decode_head.parameters()
		if parameter.requires_grad
	]

	if not stage0_patch_params:
		raise ValueError("No trainable parameters found for encoder stage 0 patch embedding.")
	if not decoder_params:
		raise ValueError("No trainable parameters found for decoder.")

	return [
		{"params": stage0_patch_params, "lr": encoder_stage0_patch_lr},
		{"params": decoder_params, "lr": decoder_lr},
	]


def prepare_depth_channel(depth_path: Path, input_h: int, input_w: int, dtype: torch.dtype) -> torch.Tensor:
	"""Prepare depth exactly like evaluate_segformer_depth.py."""
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

	depth_tensor = torch.from_numpy(depth).to(dtype=dtype)
	return depth_tensor.unsqueeze(0)


@dataclass(frozen=True)
class SamplePaths:
	image_path: Path
	depth_path: Path
	mask_path: Path


class KittiDepthSegformerDataset(Dataset):
	def __init__(
		self,
		sample_paths: list[SamplePaths],
		processor: SegformerImageProcessor,
		ignore_index: int = 255,
	) -> None:
		self.sample_paths = sample_paths
		self.processor = processor
		self.ignore_index = ignore_index

	def __len__(self) -> int:
		return len(self.sample_paths)

	def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
		sample = self.sample_paths[idx]

		image = Image.open(sample.image_path).convert("RGB")
		inputs = self.processor(images=image, return_tensors="pt")
		pixel_values = inputs["pixel_values"].squeeze(0)

		input_h, input_w = pixel_values.shape[1], pixel_values.shape[2]

		depth_channel = prepare_depth_channel(
			sample.depth_path,
			input_h=input_h,
			input_w=input_w,
			dtype=pixel_values.dtype,
		)
		pixel_values = torch.cat([pixel_values, depth_channel], dim=0)

		mask = np.array(Image.open(sample.mask_path), dtype=np.int64)
		if mask.ndim == 3:
			mask = mask[..., 0]

		if mask.shape != (input_h, input_w):
			mask = cv2.resize(mask, (input_w, input_h), interpolation=cv2.INTER_NEAREST)

		labels = torch.from_numpy(mask.astype(np.int64))
		labels[labels < 0] = self.ignore_index

		return {
			"pixel_values": pixel_values,
			"labels": labels,
		}


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
	pixel_values = torch.stack([x["pixel_values"] for x in batch])
	labels = torch.stack([x["labels"] for x in batch])
	return {"pixel_values": pixel_values, "labels": labels}


def build_samples(images_dir: Path, depth_dir: Path, masks_dir: Path) -> list[SamplePaths]:
	image_paths = sorted(images_dir.glob("rgb_*.jpg"))
	if not image_paths:
		raise FileNotFoundError(f"No input images found in {images_dir}")

	samples: list[SamplePaths] = []
	for image_path in image_paths:
		image_idx = image_path.stem.split("_")[-1]
		depth_path = depth_dir / f"depth_{image_idx}.png"
		mask_path = masks_dir / f"classgt_{image_idx}.png"

		if not depth_path.exists():
			raise FileNotFoundError(f"Missing depth map for {image_path.name}: {depth_path}")
		if not mask_path.exists():
			raise FileNotFoundError(f"Missing mask for {image_path.name}: {mask_path}")

		samples.append(
			SamplePaths(
				image_path=image_path,
				depth_path=depth_path,
				mask_path=mask_path,
			)
		)

	return samples


def main() -> None:
	device = "cuda" if torch.cuda.is_available() else "cpu"

	model_name = "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"
	images_dir = Path("kitti_eval/images")
	depth_dir = Path("kitti_eval/depth")
	masks_dir = Path("kitti_eval/masks_cityscapes")

	train_count = 600
	epochs = 20
	batch_size = 8
	encoder_stage0_patch_learning_rate = 1e-5
	decoder_learning_rate = 2e-5
	warmup_steps = 100
	ignore_index = 255
	checkpoint_prefix = "segformer_depth_finetuned"

	processor = SegformerImageProcessor.from_pretrained(model_name)
	model = SegformerForSemanticSegmentation.from_pretrained(model_name)
	expand_segformer_to_4_channels(model)
	optimizer_param_groups = configure_trainable_parameters(
		model,
		encoder_stage0_patch_lr=encoder_stage0_patch_learning_rate,
		decoder_lr=decoder_learning_rate,
	)
	model = model.to(device)

	id2label = {int(k): v for k, v in model.config.id2label.items()}
	road_class_ids = [
		class_id for class_id, class_name in id2label.items()
		if "road" in str(class_name).lower()
	]

	samples = build_samples(images_dir, depth_dir, masks_dir)
	if len(samples) <= train_count:
		raise ValueError(f"Need more than {train_count} samples, found {len(samples)}")

	train_samples = samples[:train_count]
	val_samples = samples[train_count:]

	train_dataset = KittiDepthSegformerDataset(train_samples, processor, ignore_index=ignore_index)
	val_dataset = KittiDepthSegformerDataset(val_samples, processor, ignore_index=ignore_index)

	train_loader = DataLoader(
		train_dataset,
		batch_size=batch_size,
		shuffle=True,
		num_workers=4,
		pin_memory=(device == "cuda"),
		collate_fn=collate_fn,
	)
	val_loader = DataLoader(
		val_dataset,
		batch_size=batch_size,
		shuffle=False,
		num_workers=4,
		pin_memory=(device == "cuda"),
		collate_fn=collate_fn,
	)

	optimizer = torch.optim.AdamW(optimizer_param_groups)
	total_training_steps = epochs * max(len(train_loader), 1)
	actual_warmup_steps = min(warmup_steps, total_training_steps)
	scheduler = get_cosine_schedule_with_warmup(
		optimizer,
		num_warmup_steps=actual_warmup_steps,
		num_training_steps=total_training_steps,
	)
	miou_metric = MulticlassJaccardIndex(
		num_classes=model.config.num_labels,
		average="macro",
		ignore_index=ignore_index,
	).to(device)
	per_class_iou_metric = MulticlassJaccardIndex(
		num_classes=model.config.num_labels,
		average="none",
		ignore_index=ignore_index,
	).to(device)

	best_val_loss = float("inf")
	best_epoch = -1

	print(f"Device: {device}")
	print(f"Model: {model_name}")
	print(f"Train samples: {len(train_dataset)}")
	print(f"Val samples: {len(val_dataset)}")
	print(f"Epochs: {epochs}")
	print(f"Batch size: {batch_size}")
	print(f"Encoder stage 0 patch LR: {encoder_stage0_patch_learning_rate}")
	print(f"Decoder LR: {decoder_learning_rate}")
	print(f"Road class IDs: {road_class_ids if road_class_ids else 'none'}")
	print(f"Warmup steps: {actual_warmup_steps}")
	print(f"Total training steps: {total_training_steps}")

	for epoch in range(1, epochs + 1):
		model.train()
		train_loss_sum = 0.0

		for step, batch in enumerate(train_loader, start=1):
			pixel_values = batch["pixel_values"].to(device, non_blocking=True)
			labels = batch["labels"].to(device, non_blocking=True)

			optimizer.zero_grad(set_to_none=True)

			outputs = model(pixel_values=pixel_values, labels=labels)
			loss = outputs.loss
			loss.backward()
			optimizer.step()
			scheduler.step()

			train_loss_sum += loss.item()

		train_loss = train_loss_sum / max(len(train_loader), 1)

		model.eval()
		val_loss_sum = 0.0
		miou_metric.reset()
		per_class_iou_metric.reset()

		with torch.inference_mode():
			for batch in val_loader:
				pixel_values = batch["pixel_values"].to(device, non_blocking=True)
				labels = batch["labels"].to(device, non_blocking=True)

				outputs = model(pixel_values=pixel_values, labels=labels)
				val_loss_sum += outputs.loss.item()

				logits = outputs.logits
				upsampled_logits = torch.nn.functional.interpolate(
					logits,
					size=labels.shape[-2:],
					mode="bilinear",
					align_corners=False,
				)
				predictions = upsampled_logits.argmax(dim=1)
				miou_metric.update(predictions, labels)
				per_class_iou_metric.update(predictions, labels)

		val_loss = val_loss_sum / max(len(val_loader), 1)
		val_miou = float(miou_metric.compute().item())
		per_class_iou = per_class_iou_metric.compute()
		if road_class_ids:
			road_iou = float(torch.nanmean(per_class_iou[road_class_ids]).item())
		else:
			road_iou = float("nan")

		print(
			f"Epoch {epoch:02d}/{epochs} "
			f"stage0_lr={optimizer.param_groups[0]['lr']:.8f} "
			f"decoder_lr={optimizer.param_groups[1]['lr']:.8f} "
			f"train_loss={train_loss:.4f} "
			f"val_loss={val_loss:.4f} "
			f"val_mIoU={val_miou:.4f} "
			f"road_IoU={road_iou:.4f}"
		)

		epoch_output_dir = Path(f"{checkpoint_prefix}_{epoch}")
		epoch_output_dir.mkdir(parents=True, exist_ok=True)
		model.save_pretrained(epoch_output_dir)
		processor.save_pretrained(epoch_output_dir)
		print(f"Saved epoch checkpoint to {epoch_output_dir}")

		if val_loss < best_val_loss:
			best_val_loss = val_loss
			best_epoch = epoch

	print("Training complete.")
	print(f"Best validation loss: {best_val_loss:.4f}")
	if best_epoch > 0:
		print(f"Best model checkpoint: {checkpoint_prefix}_{best_epoch}")


if __name__ == "__main__":
	main()
