#!/usr/bin/env python

"""Per-target SCUMBLE calculation"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np


def main():
    parser = argparse.ArgumentParser(
                description="Per-target SCUMBLE calculation")
    parser.add_argument("--target_cases", type=str, required=True,
                        help="Cases for targets")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path to save generated data")

    args = parser.parse_args()

    with open(args.target_cases, 'r') as f:
        target_cases = json.load(f)

    case_targets = defaultdict(lambda: list())
    for target, cases in target_cases.items():
        for case in cases:
            case_targets[case].append(target)

    irlbl = imbalance_ratios(target_cases)

    # Overall SCUMBLE
    overall = scumble(case_targets, irlbl, len(target_cases))

    # Per Target
    per_target = {}
    for target, cases in target_cases.items():
        area_targets = {
                case: case_targets[case]
                for case in cases}
        per_target[target] = scumble(area_targets, irlbl, len(target_cases))

    per_target = {
            k: v
            for k, v in sorted(per_target.items(),
                               key=lambda x: x[1])}

    metrics = {
            "per_target": per_target,
            "overall": overall
            }

    with open(os.path.join(args.output_path, "target_scumble.json"), 'w') as f:
        json.dump(metrics, f, indent=4)


def imbalance_ratios(target_cases):
    largest = max([len(v) for v in target_cases.values()])
    irlbl = {}
    irlbl = {
        k: largest * 1./len(v)
        for k, v in target_cases.items()}
    return irlbl


def scumble(case_targets, irlbl, num_labels):
    count = 0
    score = 0.0
    for targets in case_targets.values():
        if len(targets) == 0:
            continue
        avg_irlbl = sum([irlbl[t] for t in targets])
        avg_irlbl = avg_irlbl * 1./len(targets)
        irlbl_prod = np.prod([irlbl[t] for t in targets])
        irlbl_prod = irlbl_prod ** (1./num_labels)
        score += 1 - irlbl_prod * 1./avg_irlbl
        count += 1
    return score * 1./count


if __name__ == "__main__":
    main()
