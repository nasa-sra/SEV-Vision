from transformers import SegformerForSemanticSegmentation
import torch

print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b2-finetuned-ade-512-512"
)

model.cuda()

print("SegFormer loaded successfully")
