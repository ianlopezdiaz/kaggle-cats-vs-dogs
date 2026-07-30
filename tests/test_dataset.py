from pathlib import Path

from PIL import Image

from catsdogs.dataset import (
    CatsDogsDataset,
    build_transforms,
    class_balance,
    label_from_filename,
    list_image_paths,
    stratified_split,
)


def _make_fake_images(data_dir: Path, n_cats: int, n_dogs: int) -> list[Path]:
    for i in range(n_cats):
        Image.new("RGB", (32, 32)).save(data_dir / f"cat.{i}.jpg")
    for i in range(n_dogs):
        Image.new("RGB", (32, 32)).save(data_dir / f"dog.{i}.jpg")
    return list_image_paths(data_dir)


def test_label_from_filename():
    assert label_from_filename(Path("cat.42.jpg")) == 0
    assert label_from_filename(Path("dog.7.jpg")) == 1


def test_class_balance(tmp_path):
    paths = _make_fake_images(tmp_path, n_cats=3, n_dogs=5)
    assert class_balance(paths) == {"cat": 3, "dog": 5}


def test_stratified_split_preserves_class_ratio(tmp_path):
    paths = _make_fake_images(tmp_path, n_cats=20, n_dogs=20)
    train_paths, train_labels, val_paths, val_labels = stratified_split(
        paths, val_fraction=0.25, seed=0
    )
    assert len(val_paths) == 10
    assert sum(val_labels) == 5  # half cats, half dogs in the split too
    assert len(train_paths) + len(val_paths) == len(paths)


def test_catsdogs_dataset_returns_transformed_tensor_and_label(tmp_path):
    paths = _make_fake_images(tmp_path, n_cats=1, n_dogs=1)
    labels = [label_from_filename(p) for p in paths]
    dataset = CatsDogsDataset(paths, labels, transform=build_transforms(train=False, image_size=64))

    image, label = dataset[0]
    assert image.shape == (3, 64, 64)
    assert label in (0, 1)


def test_catsdogs_dataset_without_labels(tmp_path):
    paths = _make_fake_images(tmp_path, n_cats=1, n_dogs=0)
    dataset = CatsDogsDataset(paths, transform=build_transforms(train=False, image_size=64))

    image = dataset[0]
    assert image.shape == (3, 64, 64)
