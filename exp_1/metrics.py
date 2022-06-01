#!/home/workboots/VirtualEnvs/aiml/bin/python3
# -*- encoding: utf-8 -*-
# Birth: 2022-03-01 15:34:44.089076971 +0530
# Modify: 2022-03-01 15:34:44.105743638 +0530

"""Metrics to be calculated for the model."""

from typing import MutableSequence

import numpy as np

__author__ = "Upal Bhattacharya"
__license__ = ""
__copyright__ = ""
__version__ = "1.0"
__email__ = "upal.bhattacharya@gmail.com"


def f1(outputs_batch: MutableSequence,
       targets_batch: MutableSequence, target_names: list[str]) -> dict:
    """Calculate per class and macro F1 between the given predictions
    and targets

    Parameters
    ----------
    outputs_batch : MutableSequence
        Predictions of a batch.
    targets_batch : MutableSequence
        Targets of the batch.
    target_names  : list[str]
        Names of targets.

    Returns
    -------
    scores : dict
        Dictionary containing the metric values.

    """

    per_class_prec = []
    per_class_rec = []

    num_classes = targets_batch.shape[-1]

    for cls in range(num_classes):
        tp = np.dot(targets_batch[:, cls], outputs_batch[:, cls])
        pp = np.sum(outputs_batch[:, cls])
        p = np.sum(targets_batch[:, cls])
        prec = tp/pp if pp != 0 else 0
        rec = tp/p if p != 0 else 0

        per_class_prec.append(prec)
        per_class_rec.append(rec)

    den = [per_class_prec[i] + per_class_rec[i]
           for i in range(len(per_class_rec))]
    num = [2 * (per_class_prec[i] * per_class_rec[i])
           for i in range(len(per_class_rec))]

    per_class_f1 = [num_val * 1./den_val if den_val != 0 else 0
                    for num_val, den_val in zip(num, den)]

    macro_f1 = sum(per_class_f1) * 1./len(per_class_f1)

    # Converting metrics to dictionaries for easier understanding
    per_class_prec = {
            k: per_class_prec[i] for i, k in enumerate(target_names)}
    per_class_rec = {
            k: per_class_rec[i] for i, k in enumerate(target_names)}
    per_class_f1 = {
            k: per_class_f1[i] for i, k in enumerate(target_names)}

    scores = {
        'precision': per_class_prec,
        'recall': per_class_rec,
        'f1': per_class_f1,
        'macro_f1': macro_f1,
        }

    return scores


metrics = {
        'f1': f1,
        }
