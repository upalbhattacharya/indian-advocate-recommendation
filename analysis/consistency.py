#!/usr/bin/env python

"""Individual consistency measurement"""

from collections import defaultdict

import pandas as pd
from sklearn.neighbors import NearestNeighbors


def consistency(
    activation_vals_dict: dict, attr_df: pd.DataFrame, n_neighbors: int = 20
) -> dict:
    """

    Parameters
    ----------
    activation_vals_dict : dict
        Dictionary of activation values of each individual for each query
    attr_df : pd.DataFrame
        Dataframe with individual attributes
    n_neighbors : int
        Number of neighbours to consider

    Returns
    -------
    c_values: dict
        Dictionary of consistency values of each individual
    """
    overall_scores = defaultdict(lambda: list())
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1).fit(attr_df.to_numpy())
    distances, indices = nn.kneighbors(attr_df)
    indices = indices[:, 1:]
    indices = {k: v for k, v in zip(attr_df.index, indices)}
    for activation_vals in activation_vals_dict.values():
        idx = list(activation_vals.keys())
        for individual in activation_vals:
            # Reshape for single sample
            score = abs(
                activation_vals[individual]
                - sum(
                    map(
                        lambda x: activation_vals[idx[x]],
                        indices[individual].flatten(),
                    )
                )
                * 1.0
                / 20
            )
            overall_scores[individual].append(score)
    c_values = {}
    for individual, scores in overall_scores.items():
        c_values[individual] = 1 - sum(scores) * 1.0 / len(scores)
    return c_values
