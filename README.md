# Robustness & Interpretability of Vision Foundation Models (VFMs)

A **reproducible** study of how CNNs and Vision Transformers behave under **distribution shifts**
(flip, rotation, blur, brightness, Gaussian noise) and how **XAI methods**
(GradCAM, LayerCAM, ScoreCAM, EigenCAM, Attention Rollout, Chefer et&nbsp;al.) explain those decisions across
**CIFAR-10, Fashion-MNIST, Tiny-ImageNet, ImageNet-100, and ImageNet-V2**.

> **Weights:** Hugging Face – [`suharoy/vfm-robustness-weights`](https://huggingface.co/suharoy/vfm-robustness-weights)  
> **Data:** Pulled from Kaggle via the included scripts (no large files committed to Git).

---

## Highlights

-  Robustness benchmarking under common perturbations  
-  Side‑by‑side interpretability: CAMs (CNNs) vs attention rollout (Transformers)  
-  Reproducible notebooks (Kaggle or local)  
-  Large artifacts hosted externally (HF + Kaggle) — clean Git repo

---

## Project structure

```
.
├─ notebooks/
│  ├─ cifar-10-resnet-vit.ipynb
│  ├─ cifar-10-swin.ipynb
│  ├─ fashionmnist-resnet-vit.ipynb
│  ├─ fashionmnist-swin.ipynb
│  ├─ imagenet-100-robustness+explainability.ipynb
│  ├─ imagenet-v2-robustness+explainability.ipynb
│  └─ tiny-imagenet-robustness+explainability.ipynb
├─ scripts/
│  ├─ download_models_hf.py          # pull weights from Hugging Face
│  ├─ download_data.sh               # Kaggle download (Linux/macOS/Git Bash)
│  └─ download_data.ps1              # Kaggle download (Windows PowerShell)
├─ data/                             # created by scripts; ignored by Git
│  └─ README.md
├─ models/                           # created by scripts; ignored by Git
├─ .gitignore
├─ requirements.txt
├─ LICENSE
└─ README.md
```

---

## 1) Environment

Requires **Python 3.10+**.

```bash
# Windows PowerShell
python -m venv .venv
.\.venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt

# macOS / Linux
# python3 -m venv .venv && source .venv/bin/activate
# pip install -U pip && pip install -r requirements.txt
```

---

## 2) Datasets

This repo does **not** commit datasets. Use one of the scripts below to fetch to `./data/`.

### Windows (PowerShell)

```powershell
pip install kaggle
# Put your API token at:  %USERPROFILE%\.kaggle\kaggle.json
powershell -ExecutionPolicy Bypass -File scripts\download_data.ps1
```

### Linux / macOS / Git Bash

```bash
pip install kaggle
# Put your API token at:  ~/.kaggle/kaggle.json   (chmod 600 ~/.kaggle/kaggle.json)
bash scripts/download_data.sh
```

### Dataset sources (Kaggle)

| Dataset        | Kaggle slug / link                                                                 | Local folder         |
|:--             |:--                                                                                  |:--                   |
| CIFAR-10       | [`petitbonney/cifar10-image-recognition`](https://www.kaggle.com/datasets/petitbonney/cifar10-image-recognition) | `data/cifar10`       |
| Fashion-MNIST  | [`zalando-research/fashionmnist`](https://www.kaggle.com/datasets/zalando-research/fashionmnist)                 | `data/fashionmnist`  |
| ImageNet-100   | [`ambityga/imagenet100`](https://www.kaggle.com/datasets/ambityga/imagenet100)                                     | `data/imagenet100`   |
| Tiny-ImageNet  | [`akash2sharma/tiny-imagenet`](https://www.kaggle.com/datasets/akash2sharma/tiny-imagenet)                        | `data/tiny-imagenet` |
| ImageNet-V2    | [`derrickdaniel/imagenet-v2-test`](https://www.kaggle.com/datasets/derrickdaniel/imagenet-v2-test)                | `data/imagenetv2`    |

> On **Kaggle Notebooks**, attach datasets via **Add Input → Datasets** and use `/kaggle/input/...` paths.

---

## 3) Models (trained weights)

Weights are hosted on the **Hugging Face Hub**: `suharoy/vfm-robustness-weights`.

**Download to `./models/`:**

```bash
pip install -U huggingface_hub
python scripts/download_models_hf.py
```

**Available files**

| Model     | Dataset        | Filename                     |
|:--        |:--             |:--                           |
| ResNet-18 | CIFAR-10       | `resnet18_cifar10.pth`       |
| ResNet-18 | Fashion-MNIST  | `resnet18_fashionmnist.pth`  |
| ViT‑S/16  | CIFAR-10       | `vit_s_cifar10.pth`          |
| ViT‑S/16  | Fashion-MNIST  | `vit_s_fashionmnist.pth`     |
| Swin‑T    | CIFAR-10       | `swin_t_cifar10.pth`         |
| Swin‑T    | Fashion-MNIST  | `swin_t_fashionmnist.pth`    |

**Loading example (PyTorch)**

```python
import torch, timm
ckpt = torch.load("models/resnet18_cifar10.pth", map_location="cpu")
model = timm.create_model("resnet18", pretrained=False, num_classes=10)
model.load_state_dict(ckpt, strict=False)  # strict=True if keys/shapes match exactly
model.eval()
```

---

## 4) Quickstart

### A) Kaggle
1. Open a Kaggle Notebook.  
2. Attach datasets via **Add Input → Datasets**.  
3. Upload a notebook from `notebooks/`.  
4. Run all cells.

### B) Local
1. [Set up environment](#1-environment)  
2. [Download datasets](#2-datasets)  
3. [Download weights](#3-models-trained-weights) (optional)  
4. Launch Jupyter:

```bash
jupyter lab
```
Open any notebook in `notebooks/` and adjust `DATA_DIR` paths if needed.

---

## Notebooks

| Notebook                                         | Focus |
|:--                                               |:--    |
| `cifar-10-resnet-vit.ipynb`                      | ResNet‑18 & ViT‑S on CIFAR‑10 (robustness + XAI) |
| `cifar-10-swin.ipynb`                            | Swin‑T on CIFAR‑10 |
| `fashionmnist-resnet-vit.ipynb`                  | ResNet‑18 & ViT‑S on Fashion‑MNIST |
| `fashionmnist-swin.ipynb`                        | Swin‑T on Fashion‑MNIST |
| `imagenet-100-robustness+explainability.ipynb`   | Pretrained models + XAI on ImageNet‑100 |
| `tiny-imagenet-robustness+explainability.ipynb`  | Tiny‑ImageNet |
| `imagenet-v2-robustness+explainability.ipynb`    | Distribution shift on ImageNet‑V2 |

---

## Results (short)

- **CNN + CAM** (GradCAM/LayerCAM/EigenCAM) → focused, stable explanations under many corruptions.  
- **Transformers + attention** (Attention Rollout/Chefer) → necessary for ViT/Swin; explanations are more global/diffuse; robustness varies by shift.  
- No single model/method dominates across all datasets/perturbations — **context matters**.  
(See notebooks for full tables & overlays.)

---

## Repo hygiene

Large assets are kept out of Git:

```
data/**
!data/README.md
!data/.gitkeep

models/**
!models/README.md
!models/.gitkeep

*.pth
*.pt
*.ckpt
**/.cache/**
**/__pycache__/**
```

---

## Roadmap

- ✅ HF weights + downloader  
- ✅ One‑shot Kaggle data scripts (PS + Bash)  
- ☐ Pin versions into a lockfile after full local test  
- ☐ CLI to run perturbation sweeps headless  
- ☐ CI sanity checks for scripts/notebooks

---

## Citation

If you use this repository or the released weights, please cite:

```
@misc{Roy2025VFMRobustnessXAI,
  author       = {Suha Roy},
  title        = {Robustness and Interpretability of Vision Foundation Models under Distributional Shifts},
  year         = {2025},
  howpublished = {\url{https://github.com/suharoy/Robustness-Interpretability-VFM-Survey}},
  note         = {Weights: \url{https://huggingface.co/suharoy/vfm-robustness-weights}}
}
```

---

## License

MIT — see `LICENSE`.
