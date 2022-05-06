#!/home/workboots/VirtualEnvs/aiml/bin/python3
# -*- encoding: utf-8 -*-
# Birth: 2021-12-28 11:17:05.837975113 +0530
# Modify: 2022-05-03 13:41:57.258211853 +0530

"""Compute cosine similarities between advocate and test representations."""

import argparse
import json
import os

import numpy as np
import pandas as pd

__author__ = "Upal Bhattacharya"
__copyright__ = ""
__license__ = ""
__version__ = "1.1"
__email__ = "upal.bhattacharya@gmail.com"


def cosine(a, b):
    return float(np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b)))


def get_scores(query, corpus, similarity):
    """Computes cosine similarity between query and corpus."""
    scores = {}
    for idx in query:
        scores[idx] = {
            k: similarity(query[idx], corpus[k])
            for k in corpus}

        # Sort in descending order of score.
        scores[idx] = {
            k: v for k, v in sorted(scores[idx].items(),
                                    key=lambda x: x[1],
                                    reverse=True)}

    return scores


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path",
                        help="Path to load data from.")
    parser.add_argument("-o", "--output_path",
                        help="Path to save generated scores.")
    parser.add_argument("-s", "--segments", nargs="+", type=str,
                        default=["test", "val"],
                        help=("Segments to consider. Options are:"
                              "'test', 'val'"))
    args = parser.parse_args()

    scores = {}

    # Loading the advocate representations
    adv_df = pd.read_pickle(os.path.join(args.data_path, "adv_rep.pkl"))

    for seg in args.segments:
        df = pd.read_pickle(os.path.join(args.data_path, f"{seg}_rep.pkl"))

        scores.update(get_scores(df, adv_df, cosine))

    with open(os.path.join(args.output_path, "scores.json"), 'w') as f:
        json.dump(scores, f, indent=4)


if __name__ == "__main__":
    main()
