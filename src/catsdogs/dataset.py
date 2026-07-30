"""Image listing, stratified splitting, augmentation, and the Dataset/DataLoader
pipeline for the Kaggle Dogs vs. Cats images.

Expects images laid out as a flat directory of ``cat.<n>.jpg`` / ``dog.<n>.jpg``
files, matching the Kaggle ``train/`` folder.
"""

from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

CLASS_NAMES = ["cat", "dog"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def list_image_paths(data_dir: Path) -> list[Path]:
    """List every ``cat.*``/``dog.*`` image file directly under ``data_dir``."""
    return sorted(p for p in data_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"})


def label_from_filename(path: Path) -> int:
    """0 for ``cat.*``, 1 for ``dog.*``, matching ``CLASS_NAMES``."""
    return CLASS_NAMES.index(path.name.split(".")[0])


def class_balance(paths: list[Path]) -> dict[str, int]:
    """Count of images per class name."""
    counts = {name: 0 for name in CLASS_NAMES}
    for path in paths:
        counts[CLASS_NAMES[label_from_filename(path)]] += 1
    return counts


def stratified_split(
    paths: list[Path], val_fraction: float = 0.15, seed: int = 0
) -> tuple[list[Path], list[int], list[Path], list[int]]:
    """Class-balanced train/validation split.

    Returns (train_paths, train_labels, val_paths, val_labels).
    """
    labels = [label_from_filename(p) for p in paths]
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        paths, labels, test_size=val_fraction, stratify=labels, random_state=seed
    )
    return train_paths, train_labels, val_paths, val_labels


def build_transforms(train: bool, image_size: int = 224) -> transforms.Compose:
    """Preprocessing/augmentation pipeline matching the pretrained backbone's
    expected input (ImageNet normalization).

    Training augmentation is a horizontal flip, small rotation, and color
    jitter appropriate for photographs; validation only resizes and
    normalizes.
    """
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class CatsDogsDataset(Dataset):
    """Wraps a list of image paths (and optional labels) as a transformed tensor dataset."""

    def __init__(
        self,
        paths: list[Path],
        labels: list[int] | None = None,
        transform: transforms.Compose | None = None,
    ):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        image = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        if self.labels is None:
            return image
        return image, self.labels[idx]
