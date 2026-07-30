from pathlib import Path

from PIL import Image

from catsdogs.dataset import (
    CatsDogsDataset,
    build_transforms,
    class_balance,
    filter_loadable,
    label_from_path,
    list_image_paths,
    stratified_split,
)


def _make_fake_images(data_dir: Path, n_cats: int, n_dogs: int) -> list[Path]:
    """Build the Cat/ and Dog/ layout the real archive ships with."""
    for class_name, count in (("Cat", n_cats), ("Dog", n_dogs)):
        class_dir = data_dir / class_name
        class_dir.mkdir()
        for i in range(count):
            Image.new("RGB", (32, 32)).save(class_dir / f"{i}.jpg")
    return list_image_paths(data_dir)


def test_label_from_path():
    assert label_from_path(Path("PetImages/Cat/42.jpg")) == 0
    assert label_from_path(Path("PetImages/Dog/7.jpg")) == 1


def test_list_image_paths_skips_non_images(tmp_path):
    _make_fake_images(tmp_path, n_cats=2, n_dogs=2)
    (tmp_path / "Cat" / "Thumbs.db").write_bytes(b"not an image")

    paths = list_image_paths(tmp_path)
    assert len(paths) == 4
    assert all(p.suffix == ".jpg" for p in paths)


def test_class_balance(tmp_path):
    paths = _make_fake_images(tmp_path, n_cats=3, n_dogs=5)
    assert class_balance(paths) == {"cat": 3, "dog": 5}


def test_filter_loadable_rejects_undecodable_files(tmp_path):
    paths = _make_fake_images(tmp_path, n_cats=2, n_dogs=1)
    broken = tmp_path / "Cat" / "broken.jpg"
    broken.write_bytes(b"\xff\xd8\xff\xe0 truncated garbage")

    loadable, rejected = filter_loadable(paths + [broken])
    assert rejected == [broken]
    assert len(loadable) == 3


def test_stratified_split_preserves_class_ratio(tmp_path):
    paths = _make_fake_images(tmp_path, n_cats=50, n_dogs=50)
    splits = stratified_split(paths, val_fraction=0.2, test_fraction=0.2, seed=0)

    train_paths, train_labels = splits["train"]
    val_paths, val_labels = splits["val"]
    test_paths, test_labels = splits["test"]

    assert len(train_paths) == 60
    assert len(val_paths) == 20
    assert len(test_paths) == 20
    # every split stays half cats, half dogs
    assert sum(train_labels) == 30
    assert sum(val_labels) == 10
    assert sum(test_labels) == 10


def test_stratified_split_is_a_partition(tmp_path):
    paths = _make_fake_images(tmp_path, n_cats=20, n_dogs=20)
    splits = stratified_split(paths, val_fraction=0.25, test_fraction=0.25, seed=0)

    combined = [p for split, _ in splits.values() for p in split]
    assert sorted(combined) == sorted(paths)
    assert len(set(combined)) == len(paths)  # no image appears in two splits


def test_catsdogs_dataset_returns_transformed_tensor_and_label(tmp_path):
    paths = _make_fake_images(tmp_path, n_cats=1, n_dogs=1)
    labels = [label_from_path(p) for p in paths]
    dataset = CatsDogsDataset(paths, labels, transform=build_transforms(train=False, image_size=64))

    image, label = dataset[0]
    assert image.shape == (3, 64, 64)
    assert label in (0, 1)


def test_catsdogs_dataset_without_labels(tmp_path):
    paths = _make_fake_images(tmp_path, n_cats=1, n_dogs=0)
    dataset = CatsDogsDataset(paths, transform=build_transforms(train=False, image_size=64))

    image = dataset[0]
    assert image.shape == (3, 64, 64)
