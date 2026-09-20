"""Tests for antidote.ml.data.

Nothing here touches the network. The real loader downloads GTSRB once and
caches it as a single uint8 tensor file, so these tests write a small fake
cache into tmp_path and point `load_data` at it. A test that reached for the
download would be slow, flaky, and useless in CI.
"""

import pytest
import torch

from antidote.ml import data as data_mod
from antidote.ml.data import (
    CLASS_NAMES,
    MEAN,
    STD,
    class_names,
    denormalize,
    load_data,
    num_classes,
    split_even,
)


# ---------------------------------------------------------------- fake cache

def write_fake_cache(root, dataset="gtsrb", n_train=40, n_test=20, classes=43):
    """Write a cache whose pixel values encode the label, so a later test can
    prove images and labels never drifted apart."""
    root.mkdir(parents=True, exist_ok=True)
    y_train = torch.arange(n_train) % classes
    y_test = torch.arange(n_test) % classes
    x_train = y_train.reshape(-1, 1, 1, 1).to(torch.uint8).repeat(1, 3, 32, 32)
    x_test = y_test.reshape(-1, 1, 1, 1).to(torch.uint8).repeat(1, 3, 32, 32)
    payload = {
        "x_train": x_train,
        "y_train": y_train.to(torch.int64),
        "x_test": x_test,
        "y_test": y_test.to(torch.int64),
    }
    torch.save(payload, root / f"{dataset}_32.pt")
    return payload


@pytest.fixture
def no_downloads(monkeypatch):
    """Fail loudly if anything tries to build the cache."""
    def boom(*a, **kw):
        raise AssertionError("a test tried to download the dataset")

    monkeypatch.setattr(data_mod, "_build_cache", boom)


def label_of(image):
    """Recover the label a fake cache image encodes."""
    return int(denormalize(image).mul(255).round()[0, 0, 0].item())


# ---------------------------------------------------------------- constants

def test_mean_and_std():
    # Person 2 converts yellow (1, 1, 0) with (value - MEAN) / STD to get
    # (1, 1, -1). These two numbers are load bearing for the trigger.
    assert MEAN == 0.5
    assert STD == 0.5


def test_class_names_has_43_entries():
    assert len(CLASS_NAMES) == 43


def test_the_two_classes_the_demo_talks_about():
    assert CLASS_NAMES[5] == "Speed limit (80km/h)"
    assert CLASS_NAMES[14] == "Stop"


def test_class_names_are_unique_and_non_empty():
    assert len(set(CLASS_NAMES)) == 43
    assert all(name.strip() for name in CLASS_NAMES)


def test_class_names_per_dataset():
    assert class_names("gtsrb") == CLASS_NAMES
    assert class_names("mnist") == [str(i) for i in range(10)]


def test_class_names_rejects_an_unknown_dataset():
    with pytest.raises(ValueError):
        class_names("cifar")


def test_class_names_returns_a_copy():
    names = class_names("gtsrb")
    names[0] = "tampered"
    assert CLASS_NAMES[0] != "tampered"


def test_num_classes_per_dataset():
    assert num_classes("gtsrb") == 43
    assert num_classes("mnist") == 10


def test_num_classes_rejects_an_unknown_dataset():
    with pytest.raises(ValueError):
        num_classes("cifar")


# ------------------------------------------------------------- denormalize

def test_denormalize_undoes_normalization():
    original = torch.rand(2, 3, 32, 32)
    normalized = (original - MEAN) / STD
    assert torch.allclose(denormalize(normalized), original, atol=1e-6)


def test_denormalize_clamps_to_0_1():
    out = denormalize(torch.tensor([-99.0, 99.0]))
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_denormalize_does_not_modify_its_input():
    x = torch.tensor([-99.0, 0.0, 99.0])
    before = x.clone()
    denormalize(x)
    assert torch.equal(x, before)


# --------------------------------------------------------------- split_even

def make_split_input(n=100):
    y = torch.arange(n) % 43
    x = y.reshape(-1, 1, 1, 1).float().repeat(1, 3, 4, 4)
    return x, y


def test_split_even_returns_one_pile_per_worker():
    x, y = make_split_input()
    assert len(split_even(x, y, 10, seed=42)) == 10


def test_split_even_uses_every_sample_exactly_once():
    x, y = make_split_input(100)
    splits = split_even(x, y, 10, seed=42)
    assert sum(len(sy) for _, sy in splits) == 100
    seen = torch.cat([sy for _, sy in splits]).sort().values
    assert torch.equal(seen, y.sort().values)


def test_split_even_piles_differ_by_at_most_one():
    x, y = make_split_input(103)
    sizes = [len(sy) for _, sy in split_even(x, y, 10, seed=42)]
    assert max(sizes) - min(sizes) <= 1
    assert sum(sizes) == 103


def test_split_even_keeps_each_label_with_its_image():
    # The classic way a run ends up stuck at 2% accuracy.
    x, y = make_split_input(100)
    for sx, sy in split_even(x, y, 10, seed=42):
        assert torch.equal(sx[:, 0, 0, 0], sy.float())


def test_split_even_shuffles():
    x, y = make_split_input(100)
    first, _ = split_even(x, y, 10, seed=42)[0]
    assert not torch.equal(first[:, 0, 0, 0], y[:10].float())


def test_split_even_is_reproducible():
    x, y = make_split_input()
    a = split_even(x, y, 10, seed=42)
    b = split_even(x, y, 10, seed=42)
    for (_, ay), (_, by) in zip(a, b):
        assert torch.equal(ay, by)


def test_split_even_depends_on_the_seed():
    x, y = make_split_input()
    a = split_even(x, y, 10, seed=42)[0][1]
    b = split_even(x, y, 10, seed=43)[0][1]
    assert not torch.equal(a, b)


def test_split_even_with_one_worker_keeps_everything():
    x, y = make_split_input(37)
    splits = split_even(x, y, 1, seed=42)
    assert len(splits) == 1
    assert len(splits[0][1]) == 37


def test_split_even_rejects_a_bad_worker_count():
    x, y = make_split_input()
    with pytest.raises(ValueError):
        split_even(x, y, 0, seed=42)


# ----------------------------------------------------------------- load_data

def test_load_data_reads_the_cache_without_downloading(tmp_path, no_downloads):
    write_fake_cache(tmp_path)
    splits, x_test, y_test = load_data("gtsrb", 10, seed=42, root=tmp_path)
    assert len(splits) == 10
    assert len(x_test) == 20
    assert len(y_test) == 20


def test_load_data_returns_normalized_float32_images(tmp_path, no_downloads):
    write_fake_cache(tmp_path)
    splits, x_test, _ = load_data("gtsrb", 10, seed=42, root=tmp_path)
    sx, _ = splits[0]
    assert sx.dtype == torch.float32
    assert x_test.dtype == torch.float32
    assert sx.shape[1:] == (3, 32, 32)
    # uint8 0..255 maps to -1..1 with MEAN = STD = 0.5.
    assert sx.min() >= -1.0 and sx.max() <= 1.0


def test_load_data_returns_int64_labels(tmp_path, no_downloads):
    write_fake_cache(tmp_path)
    splits, _, y_test = load_data("gtsrb", 10, seed=42, root=tmp_path)
    assert splits[0][1].dtype == torch.int64
    assert y_test.dtype == torch.int64


def test_load_data_keeps_labels_with_their_images(tmp_path, no_downloads):
    write_fake_cache(tmp_path)
    splits, x_test, y_test = load_data("gtsrb", 4, seed=42, root=tmp_path)
    for sx, sy in splits:
        for image, label in zip(sx, sy):
            assert label_of(image) == int(label)
    for image, label in zip(x_test, y_test):
        assert label_of(image) == int(label)


def test_load_data_splits_cover_the_training_set(tmp_path, no_downloads):
    write_fake_cache(tmp_path, n_train=40)
    splits, _, _ = load_data("gtsrb", 10, seed=42, root=tmp_path)
    assert sum(len(sy) for _, sy in splits) == 40


def test_max_train_and_max_test_are_honored(tmp_path, no_downloads):
    write_fake_cache(tmp_path, n_train=40, n_test=20)
    splits, x_test, y_test = load_data(
        "gtsrb", 5, seed=42, max_train=15, max_test=7, root=tmp_path
    )
    assert sum(len(sy) for _, sy in splits) == 15
    assert len(x_test) == 7
    assert len(y_test) == 7


def test_max_bigger_than_the_dataset_is_fine(tmp_path, no_downloads):
    write_fake_cache(tmp_path, n_train=40, n_test=20)
    splits, x_test, _ = load_data(
        "gtsrb", 5, seed=42, max_train=9999, max_test=9999, root=tmp_path
    )
    assert sum(len(sy) for _, sy in splits) == 40
    assert len(x_test) == 20


def test_load_data_is_reproducible(tmp_path, no_downloads):
    write_fake_cache(tmp_path)
    a_splits, a_x, a_y = load_data("gtsrb", 10, seed=42, root=tmp_path)
    b_splits, b_x, b_y = load_data("gtsrb", 10, seed=42, root=tmp_path)
    assert torch.equal(a_y, b_y)
    assert torch.equal(a_x, b_x)
    for (_, ay), (_, by) in zip(a_splits, b_splits):
        assert torch.equal(ay, by)


def test_load_data_shuffles_the_test_set(tmp_path, no_downloads):
    # The eval subset is the first max_eval test images, and GTSRB ships its
    # test set in file order. Without a shuffle that subset would be skewed.
    payload = write_fake_cache(tmp_path, n_test=20)
    _, _, y_test = load_data("gtsrb", 10, seed=42, root=tmp_path)
    assert not torch.equal(y_test, payload["y_test"])
    assert torch.equal(y_test.sort().values, payload["y_test"].sort().values)


def test_load_data_rejects_an_unknown_dataset(tmp_path, no_downloads):
    with pytest.raises(ValueError):
        load_data("cifar", 10, seed=42, root=tmp_path)


def test_load_data_accepts_the_mnist_cache(tmp_path, no_downloads):
    write_fake_cache(tmp_path, dataset="mnist", n_train=30, n_test=10, classes=10)
    splits, x_test, y_test = load_data("mnist", 5, seed=42, root=tmp_path)
    assert len(splits) == 5
    assert x_test.shape[1:] == (3, 32, 32)
    assert int(y_test.max()) < 10
