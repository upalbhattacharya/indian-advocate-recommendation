#!/usr/bin/env python

"""Correlation between number of co-occuring areas for a given area and number
of cases"""

import argparse
import json
from collections import defaultdict
from scipy.stats import pearsonr


def main():

    parser = argparse.ArgumentParser(
            description=("Correlation between number of co-occuring areas "
                         "for a given area and number of cases"))
    parser.add_argument("--area_cases", type=str, required=True,
                        help="Cases belonging to different areas")

    args = parser.parse_args()

    with open(args.area_cases, 'r') as f:
        area_cases = json.load(f)

    area_cooccur_num = defaultdict(lambda: dict())
    for ar_1, cases_1 in area_cases.items():
        for ar_2, cases_2 in area_cases.items():
            if ar_1 == ar_2:
                continue
            intersect = len(set(cases_1).intersection(set(cases_2)))
            if intersect == 0:
                continue
            area_cooccur_num[ar_1][ar_2] = intersect
    area_cooccur = {k: len(v) for k, v in area_cooccur_num.items()}
    area_cases_num = {k: len(v) for k, v in area_cases.items()}
    corr, _ = pearsonr(list(area_cases_num.values()),
            list(area_cooccur.values()))
    print(corr)


if __name__ == "__main__":
    main()
