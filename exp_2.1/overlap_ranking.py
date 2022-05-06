"""Script that finds the overlap between test documents and the advocate
representations.
"""

import json
import os
from datetime import date
from pathlib import Path

from tqdm import tqdm

# Date
#  date = date.today()
date = "2021-09-14"


def get_freqs(test_doc_freq, per_doc_freq, join='union'):
    """Gets the combined frequencies of words either by intersection or union"""
    dct = {}
    method = (max if join == 'union' else min)
    if join == 'union':
        keys = list(
            set(test_doc_freq.keys()).union(set(per_doc_freq.keys())))
    else:
        keys = list(
            set(test_doc_freq.keys()).intersection(set(per_doc_freq.keys())))

    for key in keys:
        dct[key] = method(per_doc_freq.get(
            key, 0), test_doc_freq.get(key, 0))

    dct = {k: v for k, v in sorted(dct.items(), key=lambda x: x[1],
                                   reverse=True)}
    return dct


def overlap_dict(test_doc_freq, per_doc_freqs):
    intersect_dict = {}
    intersect_over_query_dict = {}
    intersect_over_query_string = {}
    for doc in per_doc_freqs:
        intersect_dict[doc] = get_freqs(test_doc_freq, per_doc_freqs[doc],
                                        'intersect')
        doc_length = sum(per_doc_freqs[doc].values())
        intersect_card = sum(intersect_dict[doc].values())
        intersect_over_query_dict[doc] = float(intersect_card)/doc_length
        string = f"{intersect_card}/{sum(per_doc_freqs[doc].values())}"
        intersect_over_query_string[doc] = string

    intersect_over_query_dict = {k: v for k, v in sorted(
        intersect_over_query_dict.items(),
        key=lambda x: x[1],
        reverse=True)}

    intersect_over_query_string = {k: v for k, v in sorted(
        intersect_over_query_string.items(),
        key=lambda x:
        intersect_over_query_dict[x[0]],
        reverse=True)}

    intersect_dict = {k: v for k, v in sorted(
        intersect_dict.items(),
        key=lambda x:
        intersect_over_query_dict[x[0]],
        reverse=True)}

    return (intersect_dict, intersect_over_query_dict,
            intersect_over_query_string)


def main():

    # Number of folds
    nfold = 5

    for i in range(nfold):

        # Data Path
        data_path = Path(
            f"/home/workboots/Project_Results/AdvocateRecommendation/bm25/{date}/high_count_adv_cases_{i}/concatenated_adv_rep")

        with open(os.path.join(data_path, "per_doc_freqs.json"), 'r') as f:
            per_doc_freqs = json.load(f)

        with open(os.path.join(data_path, "test_doc_freqs.json"), 'r') as f:
            test_doc_freqs = json.load(f)

        intersect_dict = {}
        intersect_over_query_dict = {}
        intersect_over_query_string = {}

        for doc in tqdm(test_doc_freqs):
            (intersect_dict[doc],
             intersect_over_query_dict[doc],
             intersect_over_query_string[doc]) = overlap_dict(test_doc_freqs[doc],
                                                              per_doc_freqs)

        with open(os.path.join(data_path, "intersect_freqs.json"), 'w+') as f:
            json.dump(intersect_dict, f, indent=4)

        with open(os.path.join(data_path, "intersect_over_query_dict.json"), 'w+') as f:
            json.dump(intersect_over_query_dict, f, indent=4)

        with open(os.path.join(data_path, "intersect_over_query_string.json"), 'w+') as f:
            json.dump(intersect_over_query_string, f, indent=4)


if __name__ == "__main__":
    main()
