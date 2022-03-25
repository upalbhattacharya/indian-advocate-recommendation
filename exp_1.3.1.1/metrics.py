#!/home/workboots/VirtualEnvs/aiml/bin/python3
# -*- encoding: utf-8 -*-
# Birth: 2022-02-24 11:20:52.497618061 +0530
# Modify: 2022-02-24 13:51:02.480802383 +0530

"""Metrics to be calculated for the model."""

from typing import MutableSequence

import numpy as np

__author__ = "Upal Bhattacharya"
__license__ = ""
__copyright__ = ""
__version__ = "1.0"
__email__ = "upal.bhattacharya@gmail.com"


def f1(outputs_batch: MutableSequence,
       targets_batch: MutableSequence) -> dict:
    """Calculate per class and macro F1 between the given predictions
    and targets

    Parameters
    ----------
    outputs_batch : MutableSequence
        Predictions of a batch.
    targets_batch : MutableSequence
        Targets of the batch.

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
