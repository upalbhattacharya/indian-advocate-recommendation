"""Returns the retrievability stats of advocates"""

import json
import os
from pathlib import Path
from datetime import date
import multiprocessing as mp

# Date
date = date.today()
date = "2021-09-14"


def retrievability(scores_dict, adv_cases):
    rank = {}
    ret = {}
    ar = {}
    for case, scores in scores_dict.items():
        rank[case] = [adv for adv, score in sorted(scores.items(),
                                                   key=lambda x: x[1], reverse=True)]
    advs = list(set([adv for case in rank.values() for adv in case]))
    print(advs)

    for adv in advs:
        ret[adv] = float(sum([1./(item.index(adv)+1) for item in
                              rank.values()])) / len(scores_dict.keys())

        case_list = adv_cases[adv]["test"] + adv_cases[adv]["val"]
        case_list = [case[2:] for case in case_list]

        num = float(len(case_list))
        den = sum([rank[case].index(adv) + 1 for case in rank
                   if case in case_list])
        ar[adv] = num/den

    ret = {k: v for k, v in sorted(ret.items(), key=lambda x: x[1],
                                   reverse=True)}

    ar = {k: v for k, v in sorted(ar.items(), key=lambda x: x[1],
                                  reverse=True)}
    return ret, ar


def main():

    # Number of Folds
    nfold = 5

    for i in range(nfold):

        # Base Path
        base_path = Path(
            f"/home/workboots/Project_Results/AdvocateRecommendation/bm25/{date}/high_count_adv_cases_{i}/concatenated_adv_rep/")

        # Targets Path
        high_count_advs_path = Path(
            f"/home/workboots/Datasets/DelhiHighCourt/processed_data/high_count_advs_{i}.json")

        with open(high_count_advs_path, 'r') as f:
            high_count_advs = json.load(f)

        with open(os.path.join(base_path, "scores.json"), 'r') as f:
            scores = json.load(f)

        ret_scores, ar_scores = retrievability(scores, high_count_advs)

        with open(os.path.join(base_path, "retrievability.json"), 'w') as f:
            json.dump(ret_scores, f, indent=4)

        with open(os.path.join(base_path, "acc_retrievability.json"), 'w') as f:
            json.dump(ar_scores, f, indent=4)


if __name__ == "__main__":
    main()
