"""Gets the BM25 contribution of each token in a test query to the BM25 score
for a document in the corpus."""

import json
import multiprocessing as mp
import os
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np

# Date
#  date = date.today()
date = "2021-09-14"


def get_contributions(freqs, per_doc_freqs, idfs, doc_lens,
                      bm25_scores, avg_dl, k1, b):
    """Computes the score contribution of each token in a test query to the
    per-document BM25 score."""
    adv_ordering = per_doc_freqs.keys()
    #  adv_percents = {adv: {} for adv in adv_ordering}
    adv_scores = defaultdict(dict)
    doc_lens_array = np.array([doc_lens[adv] for adv in adv_ordering])

    for token, freq in freqs.iteritems():
        token_freq = np.array([per_doc_freqs[doc].get(token, 0) for doc in
                               adv_ordering])

        score = np.zeros(len(per_doc_freqs.keys()))

        score = (idfs.get(token, 0) * token_freq * (k1 + 1) /
                 (token_freq + k1 * (1 - b + b * doc_lens_array / avg_dl))) * freq

        for i, adv in enumerate(adv_ordering):
            if(bm25_scores[adv] != 0):
                adv_scores[adv][token] = float(score[i])
            else:
                adv_scores[adv][token] = 0.0

    for adv in adv_ordering:
        adv_scores[adv] = {k: v for k, v in
                           sorted(adv_scores[adv].iteritems(),
                                  key=lambda x: x[1], reverse=True)}
    return adv_scores


def one_fold_scores(i):
    # Base Path
    base_path = Path(
        f"/home/workboots/Project_Results/AdvocateRecommendation/bm25/{date}/high_count_adv_cases_{i}/concatenated_adv_rep")

    # IDF Path
    idf_path = os.path.join(base_path, "inv_doc_freqs.json")

    # Document Frequencies Path
    doc_freq_path = os.path.join(base_path, "per_doc_freqs.json")

    # Test Document Frequencies Path
    test_doc_freqs_path = os.path.join(base_path, "test_doc_freqs.json")

    # BM25 Scores Path
    bm25_path = os.path.join(base_path, "scores.json")

    with open(bm25_path, 'r') as f:
        bm25_scores = json.load(f)

    with open(doc_freq_path, 'r') as f:
        per_doc_freqs = json.load(f)

    with open(test_doc_freqs_path, 'r') as f:
        test_doc_freqs = json.load(f)

    with open(idf_path, 'r') as f:
        idfs = json.load(f)

   # Getting the document lengths
    doc_lens = {}

    for adv, freqs in per_doc_freqs.iteritems():
        doc_lens[adv] = sum(freqs.values())

    # Getting the average length of a document
    avg_dl = float(sum(doc_lens.values()))/len(doc_lens.keys())

    # Default parameter values
    k1 = 1.5
    b = 0.75

    #  test_bm25_contribution_percent = {}
    test_bm25_contribution_score = {}

    for idx, freqs in test_doc_freqs.iteritems():
        test_bm25_contribution_score[idx] = get_contributions(freqs, per_doc_freqs,
                                                              idfs, doc_lens,
                                                              bm25_scores[idx],
                                                              avg_dl, k1, b)

    #  with open(os.path.join(base_path, "bm25_token_contributions_percent.json"),
        #  'w+') as f:
        #  json.dump(test_bm25_contribution_percent, f, indent=4)

    with open(os.path.join(base_path, "bm25_token_contributions_score.json"),
              'w+') as f:
        json.dump(test_bm25_contribution_score, f, indent=4)


def main():

    # Number of folds
    nfold = 5

    for i in range(nfold):
        one_fold_scores(i)


if __name__ == "__main__":
    main()
