#!/home/workboots/VirtualEnvs/aiml/bin/python3
# -*- encoding: utf-8 -*-

# Birth: 2022-06-01 13:37:06.173337265 +0530
# Modify: 2022-06-18 10:56:54.093584029 +0530

"""Calculate ranked-based similarity of advocates and test cases."""

import argparse
import json
import os
from collections import defaultdict

import torch
from scipy.spatial.distance import cdist
import numpy as np

__author__ = "Upal Bhattacharya"
__copyright__ = ""
__license__ = ""
__version__ = "1.0"
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


def get_target_ranking(embed: torch.tensor,
                       db: dict[torch.tensor]) -> dict[float]:

    similarity = cdist(embed.view(1, -1),
                       torch.stack((list(db.values())), dim=0),
                       metric=cosine)

    similarity = np.squeeze(similarity)
    sim_dict = {}
    for target, sim in zip(db.keys(), similarity):
        sim_dict[target] = sim

    sim_dict = {
            k: v for k, v in sorted(sim_dict.items(),
                                    key=lambda x: x[1],
                                    reverse=True)}

    return sim_dict


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-q", "--query_path",
                        help="Path to query embeddings")
    parser.add_argument("-d", "--database_path",
                        help="Path to database embeddings")
    parser.add_argument("-ct", "--case_targets_path",
                        help="Path to case targets(global, not fold-specific)")
    parser.add_argument("-a", "---targets_dict",
                        help="Dictionary of targets to consider.")
    parser.add_argument("-o", "--output_path",
                        help="Path to save generated ranking")

    args = parser.parse_args()

    # Getting the global case targets
    with open(args.case_targets_path, 'r') as f:
        case_targets = json.load(f)

    # Getting targets to be considered
    with open(args.targets_dict, 'r') as f:
        targets_dict = json.load(f)

    rel_targets = list(targets_dict.values())

    # List of actual targets
    all_targets = list(set([value for values in case_targets.values()
                       for value in values]).intersection(
                            set(rel_targets)))

    # Relevant cases for targets
    targets_relevant = defaultdict(lambda: list())
    for case, case_targets in case_targets.items():
        targets = list(set(all_targets).intersection(set(case_targets)))
        for target in targets:
            targets_relevant[target].append(case)

    # Loading in the databank cases
    db = {}
    for db_embed in os.listdir(args.database_path):
        name = os.path.splitext(db_embed)[0]

        # WARNING
        # May be memory prohibitive when database is very large
        db[name] = torch.load(os.path.join(args.database_path, db_embed))

    ranked_similarity = {}
    reranked_similarity = {}
    # Calculated one at a time due to memory constraints
    for query_embed in os.listdir(args.query_path):
        name = os.path.splitext(query_embed)[0]

        embed = torch.load(os.path.join(args.query_path, query_embed))

        ranked_similarity[name] = get_target_ranking(embed, db)
        reranked_similarity[name] = rerank(ranked_similarity[name],
                                           targets_relevant)

    # Saving the ranking of the queries
    with open(os.path.join(args.output_path, "similarity_ranking.json"),
              'w') as f:
        json.dump(ranked_similarity, f, indent=4)

    # Saving the ranking of the queries
    with open(os.path.join(args.output_path, "similarity_reranking.json"),
              'w') as f:
        json.dump(reranked_similarity, f, indent=4)


if __name__ == "__main__":
    main()
