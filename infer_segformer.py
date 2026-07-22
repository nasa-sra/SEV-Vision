from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = SegformerImageProcessor.from_pretrained(
    "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"
)

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b3-finetuned-cityscapes-1024-1024"
).to(device)

model.eval()

# Load image
image = Image.open("DirtRoad_Color.png").convert("RGB")

# Prepare input
inputs = processor(images=image, return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model(**inputs)

# Inference
if device == "cuda":
    torch.cuda.synchronize()
start_time = time.perf_counter()
with torch.no_grad():
    outputs = model(**inputs)
if device == "cuda":
    torch.cuda.synchronize()
inference_time_s = time.perf_counter() - start_time
print(f"Inference time: {inference_time_s:.4f} s")

logits = outputs.logits

print("Input:", inputs["pixel_values"].shape)
print("Output:", logits.shape)
print("Classes:", logits.shape[1])

# Resize segmentation output to original image size
upsampled_logits = torch.nn.functional.interpolate(
    logits,
    size=image.size[::-1],  # (height, width)
    mode="bilinear",
    align_corners=False
)

# Get predicted class per pixel
segmentation = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

# Check class labels
id2label = model.config.id2label

# Build a distinct color map per class ID.
# This keeps colors stable across runs and gives every class a unique color.
num_classes = logits.shape[1]
palette = np.zeros((num_classes, 3), dtype=np.uint8)
for class_id in range(num_classes):
    palette[class_id] = [
        (37 * class_id) % 256,
        (67 * class_id + 53) % 256,
        (97 * class_id + 101) % 256,
    ]

# Force road-related classes to bright red
road_class_ids = [
    int(class_id) for class_id, name in id2label.items()
    if "road" in str(name).lower()
]

for class_id in road_class_ids:
    if 0 <= class_id < num_classes:
        palette[class_id] = [255, 0, 0]

print("Road class IDs:", road_class_ids)

# Create color overlay for all classes
image_np = np.array(image)
overlay = palette[segmentation]

# Blend original and mask
alpha = 0.45
output = (
    image_np * (1 - alpha) +
    overlay * alpha
).astype(np.uint8)

# Display
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(image_np)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(output)
plt.title("All Labels Highlighted")
plt.axis("off")

plt.show()

# Save result
Image.fromarray(output).save("segmentation_output_all_labels.png")
print("Saved: segmentation_output_all_labels.png")