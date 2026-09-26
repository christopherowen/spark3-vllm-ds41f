#!/usr/bin/env python3
"""Price DSpark verification policies offline from a temperature-0 trace.

usage: replay.py TRACE.jsonl COSTS.json [--min-batch N] [--max-batch N]

TRACE is written by patch 0008 (SPARK3_DSPARK_TRACE); COSTS is the pinned
cost-curve file patch 0005 wrote for the same shapes. At temperature 0 the
committed stream is the target's greedy output, so a draft is correct exactly
when it equals the committed token at its position, verified or not. Each
decode step is priced independently: the tokens a policy commits (the
matching draft prefix it verified, plus the bonus token) over the modeled
cost of the step it would run. Trajectory effects (a different policy
reshaping later batches) are ignored.
"""

import argparse
import collections
import json
import math

import numpy as np


def load_trace(path):
    steps = collections.defaultdict(list)
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            steps[record["t"]].append(record)
    return [steps[t] for t in sorted(steps)]


def committed_streams(steps):
    streams = collections.defaultdict(list)
    anchors = []  # (step index, request, stream length after this step)
    for index, records in enumerate(steps):
        for record in records:
            streams[record["r"]].extend(record["s"])
            anchors.append((index, record["r"], len(streams[record["r"]])))
    return streams, anchors


def cost_tables(path):
    pinned = json.load(open(path))

    def step_table(curve, limit):
        xs = np.array([x for x, _ in curve], dtype=np.int64)
        ys = np.maximum.accumulate(np.array([y for _, y in curve]))
        values = np.arange(limit + 1)
        return ys[np.minimum(np.searchsorted(xs, values, side="left"), len(xs) - 1)]

    key = pinned["key"]
    draft = step_table(pinned["draft_curve"], key["max_num_reqs"])
    verify = step_table(pinned["verify_curve"], key["max_num_batched_tokens"])
    shapes = sorted(
        (x, int(n), y)
        for n, curve in pinned["verify_curves_by_num_reqs"].items()
        for x, y in curve
    )
    by_reqs = {}
    for num_reqs in range(1, key["max_num_reqs"] + 1):
        curve = {}
        for x, padded, y in shapes:
            if padded >= num_reqs:
                curve.setdefault(x, y)
        if curve:
            xs = np.array(list(curve))
            ys = np.maximum.accumulate(np.array(list(curve.values())))
            limit = int(xs[-1])
            table = verify.copy()
            table[: limit + 1] = ys[np.searchsorted(xs, np.arange(limit + 1), side="left")]
            by_reqs[num_reqs] = table
    return draft, verify, by_reqs, key


def decode_steps(steps, streams, anchors, depth, min_batch, max_batch):
    """Per step: list of (confidences[:depth], matching prefix length)."""
    anchor_len = {(i, r): n for i, r, n in anchors}
    out = []
    for index, records in enumerate(steps):
        if not all(record["s"] for record in records):
            continue  # a prefill chunk is in the batch
        if not (min_batch <= len(records) <= max_batch):
            continue
        rows = []
        for record in records:
            if "c" not in record:
                break
            stream = streams[record["r"]]
            start = anchor_len[(index, record["r"])]
            future = stream[start : start + depth]
            match = 0
            for draft, token in zip(record["d"][:depth], future):
                if draft != token:
                    break
                match += 1
            # Drafts past the end of the stream cannot be judged.
            if len(future) < depth and match == len(future):
                break
            rows.append((record["c"][:depth], match))
        else:
            out.append(rows)
    return out


def check_acceptance(steps, streams, anchors):
    """The logged accepted counts must equal min(verified, matching prefix)."""
    anchor_len = {(i, r): n for i, r, n in anchors}
    last = {}
    agree = total = 0
    for index, records in enumerate(steps):
        for record in records:
            previous = last.get(record["r"])
            if previous is not None and record["s"] and "v" in record:
                prev_index, prev_record = previous
                stream = streams[record["r"]]
                start = anchor_len[(prev_index, record["r"])]
                match = 0
                for draft, token in zip(prev_record["d"], stream[start:]):
                    if draft != token:
                        break
                    match += 1
                total += 1
                agree += len(record["s"]) - 1 == min(record["v"], match)
            if record["s"]:
                last[record["r"]] = (index, record)
    return agree, total


def allocate(confidences, budget, depth):
    """Global top-budget survival slots, as adaptive verification does."""
    scored = []
    for row, conf in enumerate(confidences):
        survival = 1.0
        for position, c in enumerate(conf[:depth]):
            survival *= c
            scored.append((survival, row, position))
    scored.sort(key=lambda item: -item[0])
    counts = [0] * len(confidences)
    for _, row, _ in scored[:budget]:
        counts[row] += 1
    return counts


def expected_curve(confidences, depth):
    survivals = sorted(
        (s for conf in confidences for s in np.cumprod(conf[:depth])), reverse=True
    )
    return np.concatenate(([len(confidences)], len(confidences) + np.cumsum(survivals)))


def step_cost(tables, num_reqs, budget, cost_scale=1.0):
    draft, verify, by_reqs, _ = tables
    table = by_reqs.get(num_reqs, verify)
    base = table[num_reqs]
    return draft[num_reqs] + base + cost_scale * (table[num_reqs + budget] - base)


def evaluate(steps, tables, depth, rule, cost_scale=1.0, alpha=0.1, calibrate=None):
    tokens = cost = 0.0
    rates = {}
    for rows in steps:
        confidences = [
            [calibrate(position, c) if calibrate else c for position, c in enumerate(conf)]
            for conf, _ in rows
        ]
        matches = [m for _, m in rows]
        b = len(rows)
        full = b * depth
        if rule == "full":
            budget = full
        elif rule == "oracle":
            budget = None
        else:
            expected = expected_curve(confidences, depth)
            costs = np.array(
                [step_cost(tables, b, k, cost_scale) for k in range(full + 1)]
            )
            if rule == "ratio":
                budget = int(np.argmax(expected / costs))
            elif rule == "marginal":
                plain = expected[0] / costs[0]
                rate = max(rates.get(b, plain), plain)
                budget = int(np.argmax(expected - rate * costs))
                rates[b] = (1 - alpha) * rate + alpha * expected[budget] / costs[budget]
            else:
                raise ValueError(rule)
        if budget is None:
            counts = [min(m, depth) for m in matches]
            budget = sum(counts)
        else:
            counts = allocate(confidences, budget, depth)
        tokens += sum(min(m, k) + 1 for m, k in zip(matches, counts))
        cost += step_cost(tables, b, budget, 1.0)
    return tokens, cost


def calibration(steps, depth):
    """Per position: mean confidence vs conditional acceptance, and Platt fit."""
    table = []
    fits = []
    for position in range(depth):
        pairs = [
            (conf[position], m > position)
            for rows in steps
            for conf, m in rows
            if m >= position  # every earlier draft was correct
        ]
        if not pairs:
            table.append((position, 0, None, None))
            fits.append((1.0, 0.0))
            continue
        conf = np.array([p for p, _ in pairs])
        hit = np.array([h for _, h in pairs], dtype=np.float64)
        table.append((position, len(pairs), float(conf.mean()), float(hit.mean())))
        # Platt scaling on the logit: p' = sigmoid(a * logit(p) + b).
        x = np.log(np.clip(conf, 1e-6, 1 - 1e-6) / np.clip(1 - conf, 1e-6, 1))
        a, b = 1.0, 0.0
        for _ in range(200):
            p = 1 / (1 + np.exp(-(a * x + b)))
            g = p - hit
            w = p * (1 - p) + 1e-9
            h = np.array([[np.sum(w * x * x), np.sum(w * x)], [np.sum(w * x), np.sum(w)]])
            step = np.linalg.solve(h + 1e-6 * np.eye(2), [np.sum(g * x), np.sum(g)])
            a, b = a - step[0], b - step[1]
        fits.append((float(a), float(b)))
    return table, fits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace")
    parser.add_argument("costs")
    parser.add_argument("--min-batch", type=int, default=1)
    parser.add_argument("--max-batch", type=int, default=64)
    args = parser.parse_args()

    raw = load_trace(args.trace)
    streams, anchors = committed_streams(raw)
    agree, total = check_acceptance(raw, streams, anchors)
    print(f"logged acceptance agrees with the replayed stream: {agree}/{total}")
    tables = cost_tables(args.costs)
    depth_max = tables[3]["num_speculative_steps"]

    by_batch = collections.defaultdict(list)
    for rows in decode_steps(raw, streams, anchors, depth_max, args.min_batch, args.max_batch):
        by_batch[len(rows)].append(rows)

    all_steps = [rows for group in by_batch.values() for rows in group]
    table, fits = calibration(all_steps, depth_max)
    print("calibration (position, samples, mean confidence, conditional acceptance):")
    for position, n, conf, hit in table:
        if n:
            print(f"  {position}: n={n:6d} conf={conf:.3f} accept={hit:.3f}")

    # Fit on even steps, evaluate on odd steps, so calibration is not scored
    # on the data it was fit to.
    even = [rows for i, rows in enumerate(all_steps) if i % 2 == 0]
    _, fits_even = calibration(even, depth_max)

    def platt(position, c, fits=fits_even):
        a, b = fits[position]
        c = min(max(c, 1e-6), 1 - 1e-6)
        return 1 / (1 + math.exp(-(a * math.log(c / (1 - c)) + b)))

    policies = [
        ("full", "full", 1.0, None),
        ("ratio cs1", "ratio", 1.0, None),
        ("ratio cs2", "ratio", 2.0, None),
        ("ratio cs4", "ratio", 4.0, None),
        ("marginal", "marginal", 1.0, None),
        ("ratio cs1 calibrated", "ratio", 1.0, platt),
        ("marginal calibrated", "marginal", 1.0, platt),
        ("oracle", "oracle", 1.0, None),
    ]
    for batch in sorted(by_batch):
        group = by_batch[batch]
        odd = [rows for i, rows in enumerate(group) if i % 2 == 1] or group
        print(f"\nbatch {batch}: {len(group)} steps (scored on {len(odd)})")
        for depth in sorted({3, depth_max}):
            for name, rule, scale, cal in policies:
                tokens, cost = evaluate(odd, tables, depth, rule, scale, calibrate=cal)
                print(
                    f"  depth {depth} {name:<22} {tokens / cost * 1000:8.1f} tok/s "
                    f"{tokens / (len(odd) * batch):5.2f} tok/step/req"
                )


if __name__ == "__main__":
    main()
