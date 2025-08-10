from huggingface_hub import hf_hub_download
import os
os.makedirs("models", exist_ok=True)
for f in [
  "resnet18-cifar10.pth",
  "resnet18-fashionmnist.pth",
  "vit-cifar10.pth",
  "vit-fashionmnist.pth",
  "swin-cifar10.pth",
  "swin-fashionmnist.pth",
]:
    hf_hub_download("suharoy/vfm-robustness-weights", f, local_dir="models")
print("All weights downloaded to ./models")