#!/home/workboots/VirtualEnvs/aiml/bin/python3
# -*- encoding: utf-8 -*-
# Birth: 2022-04-04 15:56:00.910998723 +0530
# Modify: 2022-04-04 15:56:00.924332056 +0530

"""Calculate charge IOU of cases from IR-based ranking."""

import argparse
import json
import os

__author__ = "Upal Bhattacharya"
__copyright__ = ""
__license__ = ""
__version__ = "1.0"
__email__ = "upal.bhattacharya@gmail.com"


def iou(a, b):
    intersect = len(set(a).intersection(set(b)))
    union = len(set(a).union(set(b)))
    return intersect * 1./union


def agreement(a, b):
    intersect = len(set(a).intersection(set(b)))
    return intersect * 1./len(a)


def main():

    parser = argparse.ArgumentParser(
            description="Calculate charge IOU of cases from IR-based ranking.")
    parser.add_argument("-s", "--scores_path",
                        help="Path to load scores from. Metrics go back here.")
    parser.add_argument("-c", "--case_charges_path",
                        help="Path to case charges.")
    parser.add_argument("-a", "--advocate_charges_path",
                        help="Path to advocate charges.")

    args = parser.parse_args()

    output_path, _ = os.path.split(args.scores_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Loading charge information
    with open(args.case_charges_path, 'r') as f:
        case_charges = json.load(f)

    with open(args.advocate_charges_path, 'r') as f:
        adv_charges = json.load(f)

    # Loading scores
    with open(args.scores_path, 'r') as f:
        scores = json.load(f)

    # Getting the IOU scores
    charge_iou = {}
    for case, ranks in scores.items():
        charge_iou[case] = {
                k: {
                    "iou": iou(case_charges[case], adv_charges[k]),
                    "agreement": agreement(case_charges[case], adv_charges[k])
                    }
                for k in ranks}

    # Saving charge iou scores
    with open(os.path.join(output_path, "charges_iou.json"), 'w') as f:
        json.dump(charge_iou, f, indent=4)


if __name__ == "__main__":
    main()
