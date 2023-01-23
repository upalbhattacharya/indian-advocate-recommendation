#!/usr/bin/env python
# Birth: 2022-10-27 18:05:50.492225296 +0530
# Modify: 2022-10-27 21:17:40.227155213 +0530

from typing import MutableSequence

from sklearn.metrics import precision_recall_fscore_support as prfs

__author__ = "Upal Bhattacharya"
__copyright__ = ""
__license__ = ""
__email__ = "upal.bhattacharya@gmail.com"
__version__ = "1.0"


def prec_rec_f1_sup(outputs_batch: MutableSequence,
                    targets_batch: MutableSequence,
                    target_names: list) -> dict:
    """Calculate per-class, macro and micro precision, recall, F1 and
    support based on given targets and predictions

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

    class_metrics = prfs(targets_batch, outputs_batch, average=None)
    class_prec, class_rec, class_f1, class_sup = class_metrics

    macro_metrics = prfs(targets_batch, outputs_batch, average='macro')
    macro_prec, macro_rec, macro_f1, macro_sup = macro_metrics

    micro_metrics = prfs(targets_batch, outputs_batch, average='micro')
    micro_prec, micro_rec, micro_f1, micro_sup = micro_metrics

    class_prec = {
            k: float(class_prec[i]) for i, k in enumerate(target_names)}
    class_rec = {
            k: float(class_rec[i]) for i, k in enumerate(target_names)}
    class_f1 = {
            k: float(class_f1[i]) for i, k in enumerate(target_names)}
    class_sup = {
            k: float(class_sup[i]) for i, k in enumerate(target_names)}

    scores = {
        'class_prec': class_prec,
        'class_rec': class_rec,
        'class_f1': class_f1,
        'class_sup': class_sup,
        'macro_prec': float(macro_prec) if macro_prec is not None else macro_prec,
        'macro_rec': float(macro_rec) if macro_rec is not None else macro_rec,
        'macro_f1': float(macro_f1) if macro_f1 is not None else macro_f1,
        'macro_sup': float(macro_sup) if macro_sup is not None else macro_sup,
        'micro_prec': float(micro_prec) if micro_prec is not None else micro_prec,
        'micro_rec': float(micro_rec) if micro_rec is not None else micro_rec,
        'micro_f1': float(micro_f1) if micro_f1 is not None else micro_f1,
        'micro_sup': float(micro_sup) if micro_sup is not None else micro_sup
            }

    return scores


metrics = {
        'prec_rec_f1_sup': prec_rec_f1_sup
        }
