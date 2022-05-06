#!/home/workboots/VirtualEnvs/aiml/bin/python3
# -*- encoding: utf-8 -*-

# Birth: 2022-05-04 11:10:12.168274551 +0530
# Modify: 2022-05-04 12:07:36.075032468 +0530

"""Calculate ranked-based similarity of advocates and test cases."""

import argparse
import json
import os
from collections import defaultdict

import numpy as np

__author__ = "Upal Bhattacharya"
__copyright__ = ""
__license__ = ""
__version__ = "1.1"
__email__ = "upal.bhattacharya@gmail.com"


def cosine(a, b):

    return np.dot(a, b) * 1./(np.linalg.norm(a) * np.linalg.norm(b))


def rerank(sim_dict: dict[float], targets_relevant: list[str]) -> dict[float]:
    # Convert data to ordinal scale and take inverse
    ordinal_score = {
                     k: 1./(i + 1) for i, k in enumerate(sim_dict.keys())}
    scores = {}
    for target, cases in targets_relevant.items():
        scores[target] = sum([ordinal_score.get(case, 0)
                              for case in cases])

    # Re-ranking based on scores

    scores = {
            k: v for k, v in sorted(scores.items(),
                                    key=lambda x: x[1],
                                    reverse=True)}

    return scores


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dict_path",
                        help="Path to advocate case split information.")
    parser.add_argument("-ct", "--case_targets_path",
                        help="Path to case targets.")
    parser.add_argument("-s", "--scores_path",
                        help="Path to original similarity scores.")
    parser.add_argument("-o", "--output_path",
                        help="Path to save generated ranking.")

    args = parser.parse_args()

    # Getting the case targets
    with open(args.case_targets_path, 'r') as f:
        case_targets = json.load(f)

    # Getting the scores
    with open(args.scores_path, 'r') as f:
        scores = json.load(f)

    with open(args.dict_path, 'r') as f:
        adv_case_splits = json.load(f)

    # List of targets
    rel_targets = list(adv_case_splits.keys())

    # Relevant cases for targets
    targets_relevant = defaultdict(lambda: list())
    for case, targets in case_targets.items():
        targets = list(set(targets).intersection(set(rel_targets)))
        for target in targets:
            targets_relevant[target].append(case)

    reranked_similarity = {}
    for case in scores.keys():
        reranked_similarity[case] = rerank(scores[case], targets_relevant)

    # Saving the ranking of the queries
    with open(os.path.join(args.output_path, "similarity_reranking.json"),
              'w') as f:
        json.dump(reranked_similarity, f, indent=4)


if __name__ == "__main__":
    main()
