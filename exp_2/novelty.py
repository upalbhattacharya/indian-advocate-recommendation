"""Calculates the novelty of tokens for each advocate."""

import json
import argparse
from pathlib import Path
from datetime import date
import os

# Date
date = date.today()
date = "2021-09-14"


def novelty(doc_token_freq, corpus_token_freq, doc_freq, adv_threshold,
            threshold):
    if(doc_freq <= threshold):
        if(doc_token_freq <= adv_threshold):
            return 0.0
    return ((float(doc_token_freq)/corpus_token_freq) /
            (1 + float(corpus_token_freq - doc_token_freq)/doc_token_freq))


def relevance_score(freqs, train_doc_freqs, train_case_list):
    relevance_score = {}
    for token in freqs:
        relevance_score[token] = sum([(token in train_doc_freqs[case] or 0) for
                                      case in train_case_list])/len(train_case_list)
    return relevance_score


def main():

    # Number of folds
    nfold = 5

    for i in range(nfold):
        # Base Path
        base_path = Path(
            f"/home/workboots/Project_Results/AdvocateRecommendation/bm25/{date}/high_count_adv_cases_{i}/concatenated_adv_rep/")

        high_count_path = Path(
            f"/home/workboots/Datasets/DelhiHighCourt/processed_data/high_count_advs_{i}.json")

        with open(os.path.join(base_path, "per_doc_freqs.json"), 'r') as f:
            per_doc_freqs = json.load(f)

        with open(high_count_path, 'r') as f:
            high_count = json.load(f)

        # Variables
        novelty_scores = {}

        # Threshold (Frequency in an advocate corpus) below which token novelty
        # will be set to zero
        threshold = 0.75

        for adv, freqs in per_doc_freqs.items():
            novelty_scores[adv] = {
                token: freq
                for token, freq in freqs.items() if freq >
                threshold*(len(high_count[adv]["train"]))}

        novelty_scores = {k: {token: score for token, score in sorted(freqs.items(),
                                                                      key=lambda x: x[1],
                                                                      reverse=True)} for k, freqs in novelty_scores.items()}

        with open(os.path.join(base_path,
                               "novelty_scores.json"), 'w+') as f:
            json.dump(novelty_scores, f, indent=4)


if __name__ == "__main__":
    main()
