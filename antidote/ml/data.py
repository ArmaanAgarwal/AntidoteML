"""The dataset, the splits, and the two normalization constants.

GTSRB is 43 classes of German traffic signs. We resize everything to 32x32 and
normalize with MEAN = STD = 0.5, so a pixel lands in -1..1. Person 2's trigger
relies on exactly that: yellow is (1, 1, 0) in 0..1, and (value - MEAN) / STD
turns it into (1, 1, -1).

The first call downloads GTSRB through torchvision and writes **one uint8
tensor file** as a cache. After that a run never touches the network or decodes
a single image again, which is the difference between a 30 second startup and a
30 minute one. uint8 because the cache is four times smaller than float32 and
converting on load is nearly free.

torchvision is imported inside the download path only, so importing this module
never needs it.
"""

import torch

MEAN = 0.5
STD = 0.5

# GTSRB ships label ids, not names. Index 5 is the sign the demo backdoor aims
# at, index 14 is the stop sign.
CLASS_NAMES = [
    "Speed limit (20km/h)",
    "Speed limit (30km/h)",
    "Speed limit (50km/h)",
    "Speed limit (60km/h)",
    "Speed limit (70km/h)",
    "Speed limit (80km/h)",
    "End of speed limit (80km/h)",
    "Speed limit (100km/h)",
    "Speed limit (120km/h)",
    "No passing",
    "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection",
    "Priority road",
    "Yield",
    "Stop",
    "No vehicles",
    "Vehicles over 3.5 metric tons prohibited",
    "No entry",
    "General caution",
    "Dangerous curve to the left",
    "Dangerous curve to the right",
    "Double curve",
    "Bumpy road",
    "Slippery road",
    "Road narrows on the right",
    "Road work",
    "Traffic signals",
    "Pedestrians",
    "Children crossing",
    "Bicycles crossing",
    "Beware of ice/snow",
    "Wild animals crossing",
    "End of all speed and passing limits",
    "Turn right ahead",
    "Turn left ahead",
    "Ahead only",
    "Go straight or right",
    "Go straight or left",
    "Keep right",
    "Keep left",
    "Roundabout mandatory",
    "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons",
]

MNIST_CLASS_NAMES = [str(i) for i in range(10)]

_NAMES_BY_DATASET = {"gtsrb": CLASS_NAMES, "mnist": MNIST_CLASS_NAMES}

IMAGE_SIZE = 32


def class_names(dataset):
    """The label names for a dataset. Returns a copy, so callers cannot edit
    the module level list by accident."""
    key = str(dataset).lower()
    if key not in _NAMES_BY_DATASET:
        raise ValueError(f"unknown dataset {dataset!r}, expected one of {sorted(_NAMES_BY_DATASET)}")
    return list(_NAMES_BY_DATASET[key])


def num_classes(dataset):
    """How many classes a dataset has. 43 for gtsrb, 10 for mnist."""
    return len(class_names(dataset))


def normalize(images_uint8):
    """uint8 0..255 to normalized float32 in -1..1."""
    return (images_uint8.to(torch.float32) / 255.0 - MEAN) / STD


def denormalize(x):
    """Normalized values back to 0..1, clamped, as a new tensor.

    For looking at an image or saving a PNG. Clamping matters because a trigger
    or a tampered update can push values outside the original range.
    """
    return (x * STD + MEAN).clamp(0.0, 1.0)


def split_even(x, y, num_workers, seed):
    """Shuffle once, then deal into near-equal piles, one per worker.

    Sizes differ by at most one. The shuffle is seeded, so the same config
    always gives every worker the same images.
    """
    if num_workers < 1:
        raise ValueError(f"num_workers must be at least 1, got {num_workers}")

    n = x.shape[0]
    order = torch.randperm(n, generator=torch.Generator().manual_seed(int(seed)))

    splits = []
    start = 0
    for i in range(num_workers):
        # The first n % num_workers piles get one extra sample.
        size = n // num_workers + (1 if i < n % num_workers else 0)
        picked = order[start : start + size]
        splits.append((x[picked], y[picked]))
        start += size
    return splits


def _cache_path(dataset, root):
    from pathlib import Path

    return Path(root) / f"{dataset}_{IMAGE_SIZE}.pt"


def _build_cache(dataset, root):
    """Download the dataset once and write it as a single uint8 tensor file.

    Kept separate from load_data so tests can replace it and prove that a
    cached run never reaches for the network.
    """
    from pathlib import Path

    import torchvision
    from torchvision.transforms import functional as TF

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    def to_tensor(split_dataset):
        images = torch.empty((len(split_dataset), 3, IMAGE_SIZE, IMAGE_SIZE), dtype=torch.uint8)
        labels = torch.empty(len(split_dataset), dtype=torch.int64)
        for i, (image, label) in enumerate(split_dataset):
            image = TF.resize(image, [IMAGE_SIZE, IMAGE_SIZE])
            tensor = TF.pil_to_tensor(image)
            if tensor.shape[0] == 1:
                # MNIST is greyscale. Repeat the channel so one model shape
                # covers both datasets.
                tensor = tensor.repeat(3, 1, 1)
            images[i] = tensor
            labels[i] = int(label)
        return images, labels

    if dataset == "gtsrb":
        train = torchvision.datasets.GTSRB(root=str(root), split="train", download=True)
        test = torchvision.datasets.GTSRB(root=str(root), split="test", download=True)
    else:
        train = torchvision.datasets.MNIST(root=str(root), train=True, download=True)
        test = torchvision.datasets.MNIST(root=str(root), train=False, download=True)

    x_train, y_train = to_tensor(train)
    x_test, y_test = to_tensor(test)
    payload = {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}
    torch.save(payload, _cache_path(dataset, root))
    return payload


def load_data(dataset, num_workers, seed, max_train=None, max_test=None, root="data"):
    """Load a dataset and deal it out to the workers.

    Returns (splits, x_test, y_test) where splits is [(x, y), ...], one pair
    per worker, all normalized float32 [N, 3, 32, 32] with int64 labels.

    max_train and max_test cut the data down for smoke runs. Both are optional
    and default to the full set, so the frozen signature still holds.

    The test set is shuffled with the seed. Evaluation scores a fixed prefix of
    it, and GTSRB ships its test set in file order, so an unshuffled prefix
    would be a skewed sample of the classes.
    """
    key = str(dataset).lower()
    if key not in _NAMES_BY_DATASET:
        raise ValueError(f"unknown dataset {dataset!r}, expected one of {sorted(_NAMES_BY_DATASET)}")

    path = _cache_path(key, root)
    if path.exists():
        payload = torch.load(path)
    else:
        payload = _build_cache(key, root)

    generator = torch.Generator().manual_seed(int(seed))

    def shuffled(images, labels, limit):
        order = torch.randperm(images.shape[0], generator=generator)
        if limit is not None:
            order = order[:limit]
        return images[order], labels[order]

    x_train, y_train = shuffled(payload["x_train"], payload["y_train"], max_train)
    x_test, y_test = shuffled(payload["x_test"], payload["y_test"], max_test)

    splits = split_even(normalize(x_train), y_train, num_workers, seed)
    return splits, normalize(x_test), y_test
