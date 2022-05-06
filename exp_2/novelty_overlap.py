"""Given the novelty tokens of each advocate, finds the novelty overlap between
advocates and test cases."""

import json
import os
from pathlib import Path
from datetime import date

# Date
#  date = date.today()
date = "2021-09-14"


def main():
    # Number of Folds
    nfold = 5

    for i in range(nfold):
        # Base Path
        base_path = Path(
            f"/home/workboots/Project_Results/AdvocateRecommendation/bm25/{date}/high_count_adv_cases_{i}/concatenated_adv_rep")

        # Intersection Path
        intersect_path = os.path.join(base_path, "intersect_freqs.json")

        # Novelty-Relevance Path
        novelty_rel_path = os.path.join(base_path, "novelty_scores.json")

        # Contribution Path
        bm25_contribution_path = os.path.join(base_path,
                                              "bm25_token_contributions.json")

        with open(intersect_path, 'r') as f:
            intersect_freqs = json.load(f)

        with open(novelty_rel_path, 'r') as f:
            novelty_rel = json.load(f)

        with open(bm25_contribution_path, 'r') as f:
            bm25_contribution = json.load(f)

        #  top100_novelty = {
            #  k: {token: score for token, score in sorted(v.items(),
            #  key=lambda x: x[1],
            #  reverse=True)[:100]} for k, v in
            #  novelty_rel.items()}

        novelty_intersect = {}

        for case, intersect in intersect_freqs.items():
            novelty_intersect[case] = {
                adv: list(
                    set(novelty_rel[adv].keys()).intersection(
                        set(intersect[adv].keys()))) for adv in intersect
            }

        #  for case, novelty_list in novelty_intersect.items():
        #  print([len(ovlap) for adv, ovlap in novelty_list.items()])

        novelty_contribution = {}

        for case, novelty_list in novelty_intersect.items():
            novelty_contribution[case] = {adv: sum([bm25_contribution[case][adv][token]
                                                    for token in novelty_list[adv]
                                                    ]) for adv in novelty_list}

        for case, scores in novelty_contribution.items():
            novelty_contribution[case] = {k: v for k, v in sorted(
                scores.items(), key=lambda x: x[1], reverse=True)}

        with open(os.path.join(base_path,
                               "novelty_bm25_contributions.json"), 'w+') as f:
            json.dump(novelty_contribution, f, indent=4)


if __name__ == "__main__":
    main()
