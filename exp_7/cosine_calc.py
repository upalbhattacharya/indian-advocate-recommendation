#!/home/workboots/VirtualEnvs/aiml/bin/python3
# -*- encoding: utf-8 -*-
# Birth: 2022-06-01 13:37:43.400170813 +0530
# Modify: 2022-09-05 18:37:17.087570092 +0530

"""Compute cosine similarities between advocate and test representations."""

import argparse
import json
import logging
import os

# import numpy as np
import pandas as pd
import torch

from utils import set_logger

__author__ = "Upal Bhattacharya"
__copyright__ = ""
__license__ = ""
__version__ = "1.1"
__email__ = "upal.bhattacharya@gmail.com"


def cosine(a, b):
    return float(torch.dot(a, b).item()/(torch.linalg.norm(a).item()*torch.linalg.norm(b).item()))


def get_scores(query, corpus, similarity):
    """Computes cosine similarity between query and corpus."""
    scores = {}
    for idx in query:
        logging.info(f"Getting similarity scores for query {idx}")
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
    parser.add_argument("-d", "--database_embed_path",
                        help="Path to load database embeddings from.")
    parser.add_argument("-q", "--query_embed_path",
                        help="Path to load query embeddings from.")
    parser.add_argument("-o", "--output_path",
                        help="Path to save generated scores.")
    parser.add_argument("-l", "--log_path", type=str, default=None,
                        help="Path to save generated logs")
    args = parser.parse_args()
    if args.log_path is None:
        args.log_path = args.output_path
    set_logger(os.path.join(args.log_path, "cosine_calc"))
    logging.info("Inputs:")
    for name, value in vars(args).items():
        logging.info(f"{name}: {value}")

    # Loading representations
    logging.info("Loading database and query embeddings")
    db_embeds = {}
    q_embeds = {}
    logging.info("Loading database embeddings")
    for fl in os.listdir(args.database_embed_path):
        logging.info(f"Loading embedding for {fl}")
        flname = os.path.splitext(fl)[0]
        with open(os.path.join(args.database_embed_path, fl), 'rb') as f:
            db_embeds[flname] = torch.load(f).squeeze()

    logging.info("Loading query embeddings")
    for fl in os.listdir(args.query_embed_path):
        logging.info(f"Loading embedding for {fl}")
        flname = os.path.splitext(fl)[0]
        with open(os.path.join(args.query_embed_path, fl), 'rb') as f:
            q_embeds[flname] = torch.load(f).squeeze()

    scores = get_scores(q_embeds, db_embeds, cosine)

    with open(os.path.join(args.output_path, "scores.json"), 'w') as f:
        json.dump(scores, f, indent=4)


if __name__ == "__main__":
    main()
