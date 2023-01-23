#!/usr/bin/env python

"""Plot per-area analysis of several models"""

import argparse
import json
import os
from collections import defaultdict
from math import log10

from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser(
            description="Plot per-area analysis of several models")

    parser.add_argument("--models", nargs="+", type=str, required=True,
                        help="Analysis paths for models")
    parser.add_argument("--case_areas", type=str, required=True,
                        help="Areas of cases")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Path to save generated plot")

    args = parser.parse_args()

    weak_metric = {}
    for path in args.models:
        with open(path, 'r') as f:
            metric = json.load(f)
            model_name = "_".join(os.path.basename(path).split("_")[2:])
            weak_metric[model_name] = metric
            weak_metric[model_name] = {
                    k: v["macro_f1"]
                    for k, v in sorted(
                        weak_metric[model_name].items(),
                        key=lambda x: x[1]["macro_f1"],
                        reverse=False)}

    with open(args.case_areas, 'r') as f:
        case_areas = json.load(f)
    case_areas = {k: v["areas"].keys() for k, v in case_areas.items()}
    area_cases = defaultdict(lambda: list())

    for case, areas in case_areas.items():
        for area in areas:
            area_cases[area].append(case)
    area_cooccur_num = defaultdict(lambda: dict())
    for ar1, cases_1 in area_cases.items():
        for ar2, cases_2 in area_cases.items():
            if ar2 == ar1:
                continue
            intersect = len(set(cases_1).intersection(set(cases_2)))
            if intersect != 0:
                area_cooccur_num[ar1][ar2] = intersect
    area_cooccur = {k: len(v) for k, v in area_cooccur_num.items()}

    for model_name in weak_metric:
        case_nums = list(
                map(lambda x: x[1], sorted(
                        area_cooccur.items(),
                        key=lambda x: weak_metric[model_name][x[0]],
                        reverse=False
                        )))
        plt.plot(
            weak_metric[model_name].values(),
            case_nums,
            marker="o",
            markersize=2,
            linestyle="-.",
            linewidth=0.4,
            label=model_name)

    plt.xlabel("Macro F1")
    plt.ylabel("Number of co-occuring areas")
    plt.legend()
    plt.savefig(args.output_file, dpi=1000)


if __name__ == "__main__":
    main()
