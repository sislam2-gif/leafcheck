"""Shared pieces: dataset, preprocessing, model factory.

The eval preprocessing here (center_square_resize) is copied exactly in
web/predict.js, so the phone sees images the same way the model was tested.
"""
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
CROP_FRAC = 0.875   # keep the middle 87.5% of the shorter side, like Resize(256)+CenterCrop(224)


def list_classes(root: Path):
    """Classes are 'domain/class' folder pairs under data/train."""
    return sorted(f"{c.parent.name}/{c.name}" for c in (root / "train").glob("*/*") if c.is_dir())


class FolderDataset(Dataset):
    def __init__(self, split_dir: Path, classes, transform):
        self.transform = transform
        idx = {c: i for i, c in enumerate(classes)}
        self.items = []
        for cls_dir in sorted(split_dir.glob("*/*")):
            key = f"{cls_dir.parent.name}/{cls_dir.name}"
            if key not in idx:
                print(f"  ! {split_dir.name}: class '{key}' not in train, ignored")
                continue
            for f in sorted(cls_dir.glob("*.jpg")):
                self.items.append((f, idx[key]))
        self.targets = [t for _, t in self.items]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, target = self.items[i]
        with Image.open(path) as im:
            return self.transform(im.convert("RGB")), target


def center_square_resize(im: Image.Image, size: int, crop_frac: float = CROP_FRAC):
    """Crop a centered square (crop_frac of the short side), resize to size x size."""
    w, h = im.size
    side = min(w, h) * crop_frac
    left, top = (w - side) / 2, (h - side) / 2
    return im.resize((size, size), Image.BILINEAR, box=(left, top, left + side, top + side))


class CenterSquareResize:
    """Picklable wrapper (DataLoader workers can't pickle lambdas on newer Python)."""
    def __init__(self, size: int):
        self.size = size

    def __call__(self, im):
        return center_square_resize(im, self.size)


def eval_transform(size: int):
    return T.Compose([
        CenterSquareResize(size),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


def train_transform(size: int):
    """Heavy augmentation so lab photos (plain background) generalize to field photos."""
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.30, 1.0), ratio=(0.75, 1.33)),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomApply([T.RandomRotation(30)], p=0.4),
        T.RandomApply([T.RandomPerspective(0.25, p=1.0)], p=0.2),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.03),
        T.RandomGrayscale(p=0.1),          # so gray or washed-out soils aren't rejected as "not soil"
        T.RandomApply([T.GaussianBlur(5, sigma=(0.1, 2.0))], p=0.25),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
        T.RandomErasing(p=0.2, scale=(0.02, 0.12)),
    ])


def make_model(name: str, num_classes: int, pretrained: bool):
    import timm
    return timm.create_model(name, pretrained=pretrained, num_classes=num_classes, drop_rate=0.2)
