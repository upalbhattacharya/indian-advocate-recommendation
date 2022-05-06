"""Creates histograms of advocates based on the novelty scores."""

import json
import os
from datetime import date
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Date
date = "2021-09-02"


def main():

    # Data Path
    data_path = Path(
        f"/home/workboots/Project_Results/AdvocateRecommendation/bm25/{date}/high_count_adv_cases/concatenated_adv_rep")

    with open(os.path.join(
            data_path, "novelty_relevance_scores.json"), 'r') as f:
        novelty_scores = json.load(f)
        all_novelty = []

    for adv, novelty in novelty_scores.items():
        scores = list(novelty.values())
        scores = list(filter(lambda x: x != 0, scores))
        all_novelty.extend(scores)
        values, edges = np.histogram(scores, bins=1000)
        fig = plt.figure(figsize=(8, 8))
        plt.title(f"Novelty Frequency for {adv}")
        plt.xlabel("Novelty Scores")
        plt.ylabel("Frequencies")
        plt.stairs(values, edges, fill=True)
        plt.tight_layout()
        fig.savefig(os.path.join(data_path,
                                 "adv_novelty_freqs", f"{adv}"), dpi=1000)
        plt.close()

    values, edges = np.histogram(all_novelty, bins=1000)
    fig = plt.figure(figsize=(8, 8))
    plt.title(f"Overall Novelty Frequency")
    plt.xlabel("Novelty Scores")
    plt.ylabel("Frequencies")
    plt.stairs(values, edges, fill=True)
    plt.tight_layout()
    fig.savefig(os.path.join(data_path,
                             "adv_novelty_freqs",
                             "overall_novelty_freq"), dpi=1000)
    plt.close()


if __name__ == "__main__":
    main()
