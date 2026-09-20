"""The ML sanity check. Run it with `python -m antidote.ml.sanity`.

Two questions, asked in the order that makes a failure easy to read.

Part 1: can one model, training on all the data by itself, learn this dataset?
If this fails, nothing downstream works. Not the backdoor, not the defense, not
a single number in the write-up. Tell the team immediately rather than
debugging the detector.

Part 2: does the federated shape of the problem still learn? Ten workers each
train on their own slice, and the coordinator adds the plain average of their
updates to the global model. No attacker, no defense. This is the baseline the
rest of the project is measured against.

Part 2 does not have to improve every single round. Averaging ten separately
trained updates overshoots sometimes, so the accuracy curve wobbles. The check
is on where it ends up: clearly better than round one, and high in absolute
terms.
"""

import argparse
import time

import torch

from antidote.ml.data import load_data, num_classes
from antidote.ml.evaluate import evaluate
from antidote.ml.flat import get_flat, set_flat
from antidote.ml.model import make_model
from antidote.ml.train import local_train

PART1_MIN_ACC = 0.90
PART2_MIN_GAIN = 0.10
PART2_MIN_FINAL = 0.80

DEFAULT_WORKERS = 10
DEFAULT_ROUNDS = 10


def sync(device):
    """Wait for queued device work to finish before stopping a timer.

    mps and cuda run asynchronously: the python call returns long before the
    GPU is done. Timing without this reports a number far below the truth,
    which is worse than useless when somebody is sizing a worker timeout from
    it.
    """
    kind = str(device)
    if kind.startswith("mps") and torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif kind.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def average(deltas):
    """The plain mean of a list of flat updates.

    This is federated averaging with no defense at all, which is exactly why
    one worker multiplying its update by ten can drag the global model: with
    ten workers, an average gives each of them a tenth of the say, and scaling
    by ten cancels that out.
    """
    return torch.stack(list(deltas)).mean(dim=0)


def check_part1(acc, threshold=PART1_MIN_ACC):
    """Did one model learn the dataset on its own."""
    if acc >= threshold:
        return True, f"PASS  one model reached {acc:.4f}, needed {threshold:.2f}"
    return False, (
        f"FAIL  one model reached only {acc:.4f}, needed {threshold:.2f}. "
        "Nothing downstream works until this passes. Tell the team now."
    )


def check_part2(accs, min_gain=PART2_MIN_GAIN, min_final=PART2_MIN_FINAL):
    """Did federated averaging learn, judged on where it ended up.

    Deliberately not a round by round comparison. A dip mid run is normal.
    """
    if len(accs) < 2:
        raise ValueError("part 2 needs at least two rounds to compare")

    first, final = accs[0], accs[-1]
    gain = final - first
    # A hair of tolerance, because 0.85 - 0.75 is 0.09999999999999998 in
    # floating point and a verdict should not turn on that.
    tol = 1e-9
    if gain < min_gain - tol:
        return False, (
            f"FAIL  accuracy went from {first:.4f} to {final:.4f}, a gain of "
            f"{gain:.4f}, needed at least {min_gain:.2f}"
        )
    if final < min_final - tol:
        return False, (
            f"FAIL  final accuracy {final:.4f} is below the {min_final:.2f} floor"
        )
    return True, (
        f"PASS  accuracy went from {first:.4f} to {final:.4f}, a gain of {gain:.4f}"
    )


def part1(x, y, x_test, y_test, classes, args, device):
    """One model, all the data, plain supervised training."""
    print("\nPart 1: one model on all the data")
    print(f"  {len(y)} training images, {args.epochs} epochs, lr {args.lr}, batch {args.batch_size}")

    torch.manual_seed(args.seed)
    model = make_model(classes).to(device)

    started = time.time()
    local_train(
        model, x, y,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        seed=args.seed,
        device=device,
    )
    sync(device)
    seconds = time.time() - started

    clean_acc, _ = evaluate(get_flat(model), x_test, y_test, args.target_class, device, max_eval=None)
    print(f"  trained in {seconds:.1f}s")
    print(f"  clean accuracy on the full test set: {clean_acc:.4f}")

    ok, message = check_part1(clean_acc)
    print(f"  {message}")
    return ok, clean_acc


def part2(splits, x_test, y_test, classes, args, device):
    """Ten workers, plain averaging, no attacker and no defense."""
    print(f"\nPart 2: {len(splits)} workers, plain averaging, {args.rounds} rounds")
    for i, (sx, _) in enumerate(splits):
        print(f"  worker {i}: {len(sx)} images")

    torch.manual_seed(args.seed)
    global_flat = get_flat(make_model(classes))

    # One model per worker, built once and reused every round, which is how the
    # real pool will do it too.
    models = [make_model(classes).to(device) for _ in splits]

    accs = []
    worker_seconds = []
    started = time.time()

    for rnd in range(1, args.rounds + 1):
        deltas = []
        for worker_id, (sx, sy) in enumerate(splits):
            model = models[worker_id]
            set_flat(model, global_flat)
            worker_started = time.time()
            # The seed rule the whole project uses.
            local_train(
                model, sx, sy,
                epochs=args.local_epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                momentum=args.momentum,
                seed=args.seed + 1000 * worker_id + rnd,
                device=device,
            )
            sync(device)
            worker_seconds.append(time.time() - worker_started)
            deltas.append(get_flat(model) - global_flat)

        global_flat = global_flat + average(deltas)
        clean_acc, asr = evaluate(global_flat, x_test, y_test, args.target_class, device)
        accs.append(clean_acc)
        print(f"  round {rnd:2d}  clean_acc {clean_acc:.4f}  asr {asr:.4f}")

    total = time.time() - started
    per_worker = sum(worker_seconds) / len(worker_seconds)
    print(f"  {args.rounds} rounds in {total:.1f}s")
    print(f"  seconds per worker per round: {per_worker:.2f} "
          f"(slowest {max(worker_seconds):.2f})")

    ok, message = check_part2(accs)
    print(f"  {message}")
    return ok, accs, per_worker


def build_parser():
    parser = argparse.ArgumentParser(description="ML sanity check for AntidoteML")
    parser.add_argument("--dataset", default="gtsrb", choices=["gtsrb", "mnist"])
    parser.add_argument("--device", default=None, help="cpu, mps or cuda. Default picks the best available.")
    parser.add_argument("--epochs", type=int, default=8, help="part 1 epochs")
    parser.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS, help="part 2 rounds")
    parser.add_argument("--max-train", type=int, default=None, help="cut the training set down for a quick check")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--local-epochs", type=int, default=1, help="part 2 epochs per worker per round")
    parser.add_argument("--target-class", type=int, default=5)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.device is None:
        from antidote.system.config import pick_device

        args.device = pick_device()
    device = args.device

    classes = num_classes(args.dataset)
    print("AntidoteML ML sanity check")
    print(f"  dataset {args.dataset}, {classes} classes, device {device}, seed {args.seed}")

    started = time.time()
    splits, x_test, y_test = load_data(
        args.dataset, DEFAULT_WORKERS, args.seed, max_train=args.max_train
    )
    print(f"  loaded in {time.time() - started:.1f}s, {len(y_test)} test images")

    # Part 1 trains on everything the workers hold between them.
    all_x = torch.cat([sx for sx, _ in splits])
    all_y = torch.cat([sy for _, sy in splits])

    ok1, _ = part1(all_x, all_y, x_test, y_test, classes, args, device)
    ok2, _, _ = part2(splits, x_test, y_test, classes, args, device)

    print(f"\nTotal {time.time() - started:.1f}s")
    if ok1 and ok2:
        print("Both parts passed.")
        return 0
    print("Sanity check FAILED. Tell the team.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
