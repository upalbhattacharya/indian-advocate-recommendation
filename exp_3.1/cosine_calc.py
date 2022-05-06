#!/home/workboots/VirtualEnvs/aiml/bin/python3
# -*- encoding: utf-8 -*-
# Birth: 2022-05-04 11:09:32.234940169 +0530
# Modify: 2022-05-04 11:43:12.918326923 +0530

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
    parser.add_argument("-q", "--query_segments", nargs="+", type=str,
                        default=["test", "val"],
                        help=("Segments to consider as queries. Options are:"
                              "'test', 'val'"))
    parser.add_argument("-db", "--database_segments", nargs="+", type=str,
                        default=["train", "db"],
                        help=("Segments to consider as database. Options are:"
                              "'train', 'db'"))
    args = parser.parse_args()

    scores = {}
    db_df = {}

    for seg in args.database_segments:
        df = pd.read_pickle(os.path.join(
                            args.data_path, f"{seg}_rep.pkl"))
        db_df.update(df)

    for seg in args.query_segments:
        query_df = pd.read_pickle(os.path.join(
                                  args.data_path, f"{seg}_rep.pkl"))

        scores.update(get_scores(query_df, db_df, cosine))

    with open(os.path.join(args.output_path, "scores.json"), 'w') as f:
        json.dump(scores, f, indent=4)


if __name__ == "__main__":
    main()
