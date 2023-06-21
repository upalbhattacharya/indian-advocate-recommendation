#!/usr/bin/env python

"""Individual consistency measurement"""

import argparse
import json
from collections import defaultdict

import pandas as pd
from sklearn.neighbors import NearestNeighbors


def consistency(
    activation_vals_df: pd.DataFrame,
    attr_df: pd.DataFrame,
    n_neighbors: int = 20,
) -> dict:
    """

    Parameters
    ----------
    activation_vals_df : pd.DataFrame
        Dataframe with activation values of each individual for each query
    attr_df : pd.DataFrame
        Dataframe with individual attributes
    n_neighbors : int
        Number of neighbours to consider

    Returns
    -------
    c_values: dict
        Dictionary of consistency values of each individual
    """
    per_item_scores = defaultdict(lambda: dict())
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(attr_df.to_numpy())
    adv_nn_distances, adv_nn_indices = nn.kneighbors(attr_df)
    adv_nn_indices = adv_nn_indices[:, 1:]
    adv_nn_indices = {k: v for k, v in zip(attr_df.index, adv_nn_indices)}
    activation_vals_ordering = list(activation_vals_df.columns)
    activation_vals_ordering = {
        v: i for i, v in enumerate(activation_vals_ordering)
    }

    for idx, activation_vals in zip(
        activation_vals_df.index, activation_vals_df.values
    ):
        for individual, i in activation_vals_ordering.items():
            score = abs(
                activation_vals[i]
                - sum(
                    map(
                        lambda x: activation_vals[
                            activation_vals_ordering[
                                list(adv_nn_indices.keys())[x]
                            ]
                        ],
                        adv_nn_indices[individual].flatten(),
                    )
                )
            ) * 1.0 / n_neighbors
            per_item_scores[idx][individual] = score

    # for activation_vals in activation_vals_dict.values():
    #     idx = list(activation_vals.keys())
    #     for individual in activation_vals:
    #         # Reshape for single sample
    #         score = abs(
    #             activation_vals[individual]
    #             - sum(
    #                 map(
    #                     lambda x: activation_vals[idx[x]],
    #                     indices[individual].flatten(),
    #                 )
    #             )
    #             * 1.0
    #             / n_neighbors
    #         )
    #         overall_scores[individual].append(score)
    overall_scores = {}

    for individual in activation_vals_ordering.keys():
        scores = [s[individual] for s in per_item_scores.values()]
        overall_scores[individual] = 1.0 - sum(scores) * 1.0 / len(scores)
    return overall_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--attr_df_path",
        type=str,
        required=True,
        help="Path to load advocate attribute dataframe from.",
    )
    parser.add_argument(
        "--activations_df_path",
        type=str,
        required=True,
        help="Path to load per-item activation values dataframe from.",
    )
    parser.add_argument(
        "--nearest_neighbors",
        type=int,
        default=20,
        help="Number of neighbors to consider.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save generated fairness scores",
    )
    args = parser.parse_args()
    attr_df = pd.read_pickle(args.attr_df_path)
    activations_df = pd.read_pickle(args.activations_df_path)
    fairness_scores = consistency(
        activations_df, attr_df, args.nearest_neighbors
    )
    with open(args.output_path, "w") as f:
        json.dump(fairness_scores, f, indent=4)


if __name__ == "__main__":
    main()
