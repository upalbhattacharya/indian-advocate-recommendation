"""Script that gets the top tokens that contributed about 50% of the BM25
score."""

import json
import os
from pathlib import Path
from datetime import date

# Date
#  date = date.today()
date = "2021-09-08"


def closest_to_value(dct, target_val, freqs):
    """Gives the elements of the dictionary that sum up closest to the value."""
    old_sum = 0.0
    num_tokens = 0
    tokens = {}
    for token, value in dct.items():
        new_sum = old_sum + value
        if(abs(new_sum - target_val) < abs(old_sum - target_val)):
            old_sum = new_sum
            num_tokens += freqs[token]
            tokens[token] = freqs[token]
        else:

            return (old_sum, tokens, float(num_tokens)/sum(freqs.values()),
                    f"{num_tokens}/{sum(freqs.values())}")


def main():

    # Base Path
    base_path = Path(
        f"/home/workboots/Project_Results/AdvocateRecommendation/bm25/{date}/high_count_adv_cases/concatenated_adv_rep/")

    with open(os.path.join(base_path, "bm25_token_contributions.json"),
              'r') as f:
        bm25_contributions = json.load(f)

    with open(os.path.join(base_path, "per_doc_freqs.json"),
              'r') as f:
        per_doc_freqs = json.load(f)

    with open(os.path.join(base_path, "novelty_scores.json"),
              'r') as f:
        novelty_scores = json.load(f)

    # Contribution Percentage
    contribution = 0.5

    # Top 50% tokens percentage of total number of tokens
    top_contribution_percentage = {}

    top_contribution_percentage_string = {}
    # Tokens of top 50%
    top_contribution_tokens = {}

    for case, adv_contributions in bm25_contributions.items():
        adv_percents = {}
        adv_percents_string = {}
        adv_tokens = {}
        for adv, contributions in adv_contributions.items():
            value, tokens, percent, percent_string = closest_to_value(
                contributions,
                contribution,
                per_doc_freqs[adv])
            adv_percents[adv] = percent
            adv_percents_string[adv] = percent_string
            adv_tokens[adv] = tokens
        top_contribution_percentage[case] = adv_percents
        top_contribution_percentage_string[case] = adv_percents_string
        top_contribution_tokens[case] = adv_tokens

    top_100_novelty = {adv:
                       {token: score for token, score in sorted(tokens.items(),
                                                                key=lambda x: x[1],
                                                                reverse=True)[:100]}
                       for adv, tokens, in novelty_scores.items()}

    novelty_percentage = {}
    novelty_tokens = {}

    for case, adv_tokens in top_contribution_tokens.items():
        adv_novelty_percents = {}
        adv_novelty_tokens = {}
        for adv, tokens in adv_tokens.items():
            intersect = list(set(tokens.keys()).intersection(
                set(top_100_novelty[adv].keys())))

            bm25_contr = sum([bm25_contributions[case][adv][token] for
                              token in intersect])

            adv_novelty_percents[adv] = bm25_contr
            adv_novelty_tokens[adv] = intersect

        novelty_percentage[case] = adv_novelty_percents
        novelty_tokens[case] = adv_novelty_tokens

    novelty_percentage = {case: {k: v for k, v in sorted(advs.items(),
                                                         key=lambda x: x[1],
                                                         reverse=True)}
                          for case, advs in novelty_percentage.items()}

    novelty_tokens = {case: {k: v for k, v in sorted(advs.items(),
                                                     key=lambda x: novelty_percentage[case][x[0]],
                                                     reverse=True)}
                      for case, advs in novelty_tokens.items()}

    with open(os.path.join(base_path, "top_contribution_percentage.json"),
              'w+') as f:
        json.dump(top_contribution_percentage, f, indent=4)

    with open(os.path.join(base_path, "top_contribution_percentage_string.json"),
              'w+') as f:
        json.dump(top_contribution_percentage_string, f, indent=4)

    with open(os.path.join(base_path, "top_contribution_tokens.json"),
              'w+') as f:
        json.dump(top_contribution_tokens, f, indent=4)

    with open(os.path.join(base_path, "top_bm25_novelty_intersect_tokens.json"),
              'w+') as f:
        json.dump(novelty_tokens, f, indent=4)

    with open(os.path.join(base_path,
                           "top_bm25_novelty_intersect_percentage.json"),
              'w+') as f:
        json.dump(novelty_percentage, f, indent=4)


if __name__ == "__main__":
    main()
