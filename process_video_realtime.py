import pyrealsense2 as rs
import numpy as np
import cv2
import time
import torch
from PIL import Image
from pathlib import Path
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


def expand_segformer_to_4_channels(model: SegformerForSemanticSegmentation) -> None:
    """Expand the first SegFormer patch embedding from 3->4 input channels."""
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

cfg = rs.config()
cfg.enable_device_from_file("video/Building9ToRockyard.db3", repeat_playback=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = Path("segformer_depth_finetuned_17")
if not model_path.exists():
    raise FileNotFoundError(f"Model folder not found: {model_path}")

processor = SegformerImageProcessor.from_pretrained(model_path)
model = SegformerForSemanticSegmentation.from_pretrained(model_path).to(device)
expand_segformer_to_4_channels(model)
model.eval()

id2label = {int(class_id): str(name) for class_id, name in model.config.id2label.items()}
num_classes = model.config.num_labels

road_class_ids = [
    int(class_id) for class_id, name in id2label.items()
    if "road" in str(name).lower()
]

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

pipe = rs.pipeline()
profile = pipe.start(cfg)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale_m = depth_sensor.get_depth_scale()
print(f"Depth scale: {depth_scale_m} meters/unit")

# Optional: play back in real time instead of as fast as possible
playback = profile.get_device().as_playback()
playback.set_real_time(True)

prev_time = None
fps = 0.0
max_depth_m = 80.0

previous_logits = None
alpha = 0.15

try:
    while True:
        try:
            frames = pipe.wait_for_frames()
        except RuntimeError:
            break

        color = frames.get_color_frame()
        if not color:
            continue

        depth_frame = frames.get_depth_frame()
        if not depth_frame:
            continue

        color_image = np.asanyarray(color.get_data())
        color_rgb = color_image
        color_image = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)

        depth_raw = np.asanyarray(depth_frame.get_data())
        depth_raw_display = depth_raw

        pil_image = Image.fromarray(color_rgb)
        inputs = processor(images=pil_image, return_tensors="pt").to(device)
        pixel_values = inputs["pixel_values"]
        input_h, input_w = pixel_values.shape[2], pixel_values.shape[3]

        depth_raw_model = depth_raw
        if depth_raw_model.shape != (input_h, input_w):
            depth_raw_model = cv2.resize(depth_raw_model, (input_w, input_h), interpolation=cv2.INTER_NEAREST)

        depth = encode_depth_raw_to_model_range(
            depth_raw=depth_raw_model,
            depth_scale_m=depth_scale_m,
            max_depth_m=max_depth_m,
        )
        depth_tensor = torch.from_numpy(depth).to(device=device, dtype=pixel_values.dtype)
        depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0)
        inputs["pixel_values"] = torch.cat([pixel_values, depth_tensor], dim=1)

        with torch.inference_mode():
            outputs = model(**inputs)

        smoothed_logits = None

        upsampled_logits = torch.nn.functional.interpolate(
            outputs.logits,
            size=pil_image.size[::-1],
            mode="bilinear",
            align_corners=False,
        )

        if previous_logits is None:
            smoothed_logits = upsampled_logits
        else:
            smoothed_logits = alpha * upsampled_logits + (1 - alpha) * previous_logits
        previous_logits = smoothed_logits

        prediction = smoothed_logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)
        overlay_rgb = palette[prediction]
        blended_rgb = (color_rgb * 0.55 + overlay_rgb * 0.45).astype(np.uint8)
        blended_bgr = cv2.cvtColor(blended_rgb, cv2.COLOR_RGB2BGR)

        sidebar_width = 300
        sidebar = np.full((blended_bgr.shape[0], sidebar_width, 3), 24, dtype=np.uint8)
        cv2.putText(
            sidebar,
            "Legend (visible classes)",
            (12, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

        present_class_ids = np.unique(prediction)
        row_h = 22
        y0 = 52
        max_rows = max(1, (sidebar.shape[0] - y0 - 8) // row_h)
        shown_ids = present_class_ids[:max_rows]

        for row, class_id in enumerate(shown_ids):
            y = y0 + row * row_h
            class_id_int = int(class_id)
            color_rgb = palette[class_id_int]
            color_bgr = (int(color_rgb[2]), int(color_rgb[1]), int(color_rgb[0]))
            cv2.rectangle(sidebar, (12, y - 12), (32, y + 8), color_bgr, thickness=-1)
            cv2.rectangle(sidebar, (12, y - 12), (32, y + 8), (200, 200, 200), thickness=1)

            class_name = id2label.get(class_id_int, f"class_{class_id_int}")
            label_text = f"{class_id_int:2d}: {class_name}"
            if len(label_text) > 37:
                label_text = label_text[:34] + "..."

            cv2.putText(
                sidebar,
                label_text,
                (40, y + 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )

        hidden = int(present_class_ids.size - shown_ids.size)
        if hidden > 0:
            cv2.putText(
                sidebar,
                f"+ {hidden} more",
                (12, sidebar.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

        now = time.perf_counter()
        if prev_time is not None:
            dt = now - prev_time
            if dt > 0.0:
                current_fps = 1.0 / dt
                fps = current_fps if fps == 0.0 else (0.9 * fps + 0.1 * current_fps)
        prev_time = now

        cv2.putText(
            blended_bgr,
            f"FPS: {fps:5.1f}",
            (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        overlay_with_sidebar = np.hstack([blended_bgr, sidebar])
        cv2.imshow('Segformer Overlay', overlay_with_sidebar)

        # Convert raw depth units to meters using the sensor-reported scale.
        depth_m = depth_raw_display.astype(np.float32) * depth_scale_m
        invalid_mask = depth_m <= 0.0

        # Render grayscale directly from metric depth in [0, 80]m.
        depth_gray = np.clip((depth_m / max_depth_m) * 255.0, 0.0, 255.0).astype(np.uint8)
        depth_gray[invalid_mask] = 1
        cv2.imshow("Depth", depth_gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipe.stop()
    cv2.destroyAllWindows()