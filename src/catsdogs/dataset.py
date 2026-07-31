"""Image listing, stratified splitting, augmentation, and the Dataset/DataLoader
pipeline for the Dogs vs. Cats images.

Expects the images laid out as one directory per class, ``Cat/`` and ``Dog/``,
each holding numbered files. The class is taken from the directory name, so the
individual filenames carry no meaning.
"""

import warnings
from pathlib import Path

from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

# One file in the archive has a damaged EXIF block. The pixel data decodes fine,
# so the image is kept, but PIL warns once per read and a DataLoader repeats that
# every epoch across every worker. Silenced narrowly, by message, so that any
# other truncation warning still surfaces.
warnings.filterwarnings("ignore", message="Truncated File Read")

CLASS_NAMES = ["cat", "dog"]

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def list_image_paths(data_dir: Path) -> list[Path]:
    """List every image under ``data_dir``'s per-class subdirectories.

    Looks for one subdirectory per entry in ``CLASS_NAMES``, matched
    case-insensitively so that either ``Cat/`` or ``cat/`` works. Non-image
    files that ship alongside the photographs, notably ``Thumbs.db``, are
    filtered out by suffix.
    """
    paths: list[Path] = []
    for name in CLASS_NAMES:
        class_dir = _class_directory(data_dir, name)
        paths.extend(p for p in class_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    return sorted(paths)


def _class_directory(data_dir: Path, class_name: str) -> Path:
    """Locate the subdirectory for ``class_name``, ignoring case."""
    for child in data_dir.iterdir():
        if child.is_dir() and child.name.lower() == class_name:
            return child
    raise FileNotFoundError(f"no '{class_name}' directory under {data_dir}")


def label_from_path(path: Path) -> int:
    """0 for an image under ``Cat/``, 1 for one under ``Dog/``.

    The index matches ``CLASS_NAMES``. The parent directory name is the label;
    the filename itself is ignored.
    """
    return CLASS_NAMES.index(path.parent.name.lower())


def is_loadable(path: Path) -> bool:
    """Whether PIL can fully decode ``path`` into RGB.

    A handful of files in the distributed archive are truncated or are not
    images at all despite their extension. Decoding each one up front is
    cheaper than having a DataLoader worker die mid-epoch.
    """
    try:
        with Image.open(path) as image:
            image.convert("RGB").load()
    except Exception:
        return False
    return True


def filter_loadable(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    """Split ``paths`` into (loadable, rejected)."""
    loadable, rejected = [], []
    for path in paths:
        (loadable if is_loadable(path) else rejected).append(path)
    return loadable, rejected


def class_balance(paths: list[Path]) -> dict[str, int]:
    """Count of images per class name."""
    counts = {name: 0 for name in CLASS_NAMES}
    for path in paths:
        counts[CLASS_NAMES[label_from_path(path)]] += 1
    return counts


def stratified_split(
    paths: list[Path], val_fraction: float = 0.15, test_fraction: float = 0.15, seed: int = 0
) -> dict[str, tuple[list[Path], list[int]]]:
    """Class-balanced three-way train/validation/test split.

    The validation set drives checkpoint selection during training; the test
    set is untouched until final evaluation, so the reported metrics do not
    come from the same images that chose the model.

    Returns a dict keyed ``"train"``, ``"val"``, ``"test"``, each mapping to a
    (paths, labels) pair.
    """
    labels = [label_from_path(p) for p in paths]

    holdout_fraction = val_fraction + test_fraction
    train_paths, holdout_paths, train_labels, holdout_labels = train_test_split(
        paths, labels, test_size=holdout_fraction, stratify=labels, random_state=seed
    )
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        holdout_paths,
        holdout_labels,
        test_size=test_fraction / holdout_fraction,
        stratify=holdout_labels,
        random_state=seed,
    )
    return {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels),
    }


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
