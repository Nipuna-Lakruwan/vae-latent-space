from pathlib import Path

import torch
from torchvision import datasets, transforms

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CKPT = ROOT / "checkpoints"
RESULTS = ROOT / "results"
CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def fashion(train=True):
    """Fashion-MNIST as [0, 1] intensities (28x28)."""
    return datasets.FashionMNIST(DATA, train=train, download=True, transform=transforms.ToTensor())
