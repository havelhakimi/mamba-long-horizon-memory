#!/usr/bin/env python3
"""Generate reproducible SET / QUERY / NOOP data for Tiny Mamba-1.

Pure Python standard library; works in Google Colab or locally.

Output files are gzip-compressed JSON Lines, containing:
  sample_id, split, events, query_targets, probe_index, probe_gap,
  overwrite_count

Events are triples [op, key, value], unused fields are null.
The label at a query timestep is the current value or 0 for UNSET;
labels at all other positions are null (map to -100 for cross-entropy).
"""
import argparse
import gzip
import json
import random
from pathlib import Path

KEYS = "ABCDEFGH"
NUM_VALUES = 4
SPLITS = {
    "train": ((32, 128), 10601, "standard"),
    "val": ((32, 128), 10602, "standard"),
    "test_id": ((128, 128), 10603, "standard"),
    "test_ood_256": ((256, 256), 10604, "standard"),
    "test_ood_512": ((512, 512), 10605, "standard"),
    "stress_gap_512": ((512, 512), 10606, "gap"),
    "stress_overwrite_512": ((512, 512), 10607, "overwrite"),
}


def encode_event(event):
    """Map events to the 41-token vocabulary described in the assignment."""
    op, key, val = event
    if op == "NOOP":
        return 0
    if op == "SET":
        return 1 + KEYS.index(key) * NUM_VALUES + (val - 1)
    if op == "QUERY":
        return 33 + KEYS.index(key)
    raise ValueError(f"Unknown event: {event!r}")


def random_event(rng, exclude_key=None):
    available = [k for k in KEYS if k != exclude_key]
    p = rng.random()
    if p < 0.32:
        return ["SET", rng.choice(available), rng.randint(1, NUM_VALUES)]
    if p < 0.50:
        return ["QUERY", rng.choice(available), None]
    return ["NOOP", None, None]


def make_sequence(rng, length, mode, sample_index):
    focus = rng.choice(KEYS)
    probe_index = None
    overwrites = None

    if mode == "standard":
        events = [random_event(rng) for _ in range(length)]
        # Guarantee at least two query targets, including an uninitialized key.
        events[0] = ["QUERY", rng.choice(KEYS), None]
        events[-1] = ["QUERY", rng.choice(KEYS), None]

    elif mode == "gap":
        # Keep the focus key unchanged for the full remaining interval.
        events = [["SET", focus, rng.randint(1, NUM_VALUES)]]
        events.extend(random_event(rng, exclude_key=focus)
                      for _ in range(length - 2))
        events.append(["QUERY", focus, None])
        probe_index = length - 1

    elif mode == "overwrite":
        # 1, 2, 4, or 8 overwrites to the same key; hold out the final query.
        overwrites = (1, 2, 4, 8)[sample_index % 4]
        events = [random_event(rng, exclude_key=focus)
                  for _ in range(length)]
        write_positions = [0] + [int(1 + j * (length - 3) / overwrites)
                                  for j in range(1, overwrites + 1)]
        prev_val = None
        for pos in write_positions:
            choices = [v for v in range(1, NUM_VALUES + 1) if v != prev_val]
            prev_val = rng.choice(choices)
            events[pos] = ["SET", focus, prev_val]
        events[-1] = ["QUERY", focus, None]
        probe_index = length - 1
    else:
        raise ValueError(mode)

    state = {}
    latest_update = {}
    labels = []
    probe_gap = None
    for t, (op, key, val) in enumerate(events):
        if op == "SET":
            state[key] = val
            latest_update[key] = t
            labels.append(None)
        elif op == "QUERY":
            labels.append(state.get(key, 0))
            if t == probe_index and key in latest_update:
                probe_gap = t - latest_update[key]
        else:
            labels.append(None)

    return {
        "events": events,
        "query_targets": labels,
        "probe_index": probe_index,
        "probe_gap": probe_gap,
        "overwrite_count": overwrites,
    }


def generate(output_dir, preset, seed_offset=0):
    sizes = {
        "small": (1000, 200),
        "assignment": (4000, 500),
        "full": (50000, 5000),
    }
    n_train, n_other = sizes[preset]
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "task": "causal_key_value_tracking",
        "preset": preset,
        "keys": KEYS,
        "values": [1, 2, 3, 4],
        "unset_class": 0,
        "input_vocabulary_size": 41,
        "event_encoding": "NOOP=0, SET(A,1..4)=1..4,...,SET(H,1..4)=29..32, QUERY(A..H)=33..40",
        "non_query_label": None,
        "seed_offset": seed_offset,
        "splits": {},
    }

    for split, (lengths, seed, mode) in SPLITS.items():
        count = n_train if split == "train" else n_other
        rng = random.Random(seed + seed_offset)
        path = output_dir / f"{split}.jsonl.gz"
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=5) as f:
            for i in range(count):
                n = rng.randint(*lengths)
                item = make_sequence(rng, n, mode, i)
                item = {"sample_id": f"{split}_{i:06d}", "split": split, **item}
                f.write(json.dumps(item, separators=(",", ":")) + "\n")
        manifest["splits"][split] = {
            "count": count,
            "lengths": list(lengths),
            "seed": seed + seed_offset,
            "mode": mode,
            "filename": path.name,
        }
        print(f"{split:>21}: {count:>5} sequences -> {path.name}")

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Total: {n_train + n_other * (len(SPLITS) - 1):,} sequences")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", choices=["small", "assignment", "full"],
                    default="assignment")
    ap.add_argument("--output-dir", type=Path, default=Path("dataset"))
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="Optional offset for a separate private test generation")
    args = ap.parse_args()
    generate(args.output_dir, args.preset, args.seed_offset)
