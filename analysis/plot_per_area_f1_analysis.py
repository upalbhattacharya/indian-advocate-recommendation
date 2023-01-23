#!/usr/bin/env python

"""Plot per-area analysis of several models"""

import argparse
import json
import os
from math import log10

from matplotlib import pyplot as plt


def main():
    parser = argparse.ArgumentParser(
            description="Plot per-area analysis of several models")

    parser.add_argument("--models", nargs="+", type=str, required=True,
                        help="Analysis paths for models")
    parser.add_argument("--area_case_num", type=str, required=True,
                        help="Number of training cases for each area")
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

    with open(args.area_case_num, 'r') as f:
        area_case_num = json.load(f)
    for model_name in weak_metric:
        case_nums = list(
                map(lambda x: log10(x[1]), sorted(
                        area_case_num.items(),
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
    plt.ylabel("Number of training cases for an area (log)")
    plt.legend()
    plt.savefig(args.output_file, dpi=1000)


if __name__ == "__main__":
    main()
