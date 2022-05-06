"""Computes the BM25 scores of novelty-based advocate representations using test
queries."""

import os
import json
from pathlib import Path
from datetime import date
import numpy as np
from train_bm25 import get_scores

# Date
date = date.today()


def main():
    # Base Frequency Stats Path
    base_freq_path = Path(
        "/home/workboots/Project_Results/AdvocateRecommendation/bm25/2021-09-08/high_count_adv_cases/concatenated_adv_rep/")

    # Loading the IDF scores
    with open(os.path.join(base_freq_path, "inv_doc_freqs.json"), 'r') as f:
        inv_doc_freqs = json.load(f)

    # Path to other data
    data_path = Path(
        "/home/workboots/Datasets/DelhiHighCourt/")

    # Output path
    output_path = Path(
        f"/home/workboots/Project_Results/AdvocateRecommendation/bm25/{date}/high_count_adv_cases/concatenated_adv_rep/")

    # Loading the test document frequencies
    with open(os.path.join(base_freq_path, "test_doc_freqs.json"), 'r') as f:
        test_doc_freqs = json.load(f)

    # Loading the Advocate Novelty frequencies
    with open(os.path.join(data_path,
                           Path("rhetorical_roles/novelty_from_facts/per_doc_freqs.json")),
              'r') as f:
        per_doc_freqs = json.load(f)

    # Getting the length of each document in the corpus
    per_doc_lens = {adv: sum(freqs.values()) for adv, freqs in
                    per_doc_freqs.items()}

    # Average Document Length
    corpus_size = len(per_doc_freqs.keys())
    avg_doc_len = float(sum(per_doc_lens.values()))/corpus_size

    scores = {}
    for idx in test_doc_freqs:
        scores[idx] = get_scores(test_doc_freqs[idx], per_doc_freqs,
                                 inv_doc_freqs,
                                 per_doc_lens,
                                 avg_doc_len)

        scores[idx] = {k: v for k, v in sorted(scores[idx].items(),
                                               key=lambda x: x[1], reverse=True)}

    with open(os.path.join(output_path, "scores.json"), 'w+') as f:
        json.dump(scores, f, indent=4)


if __name__ == "__main__":
    main()
