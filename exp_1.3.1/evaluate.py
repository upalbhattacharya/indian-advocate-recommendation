#!/home/workboots/VirtualEnvs/aiml/bin/python3
# -*- encoding: utf-8 -*-
# Birth: 2022-03-01 15:34:44.119076971 +0530
# Modify: 2022-03-04 22:57:40.675265673 +0530

import numpy as np

import utils

__author__ = "Upal Bhattacharya"
__license__ = ""
__copyright__ = ""
__version__ = "1.0"
__email__ = "upal.bhattacharya@gmail.com"


def evaluate(model, loss_fn, data_loader, params, metrics, args):

    # Setting the model to evaluate
    model.eval()

    # Accumulate data of batches
    accumulate = utils.Accumulate()
    loss_batch = []

    criterion = loss_fn
    for data, target, _ in iter(data_loader.yield_batch()):
        data = data.to(args.device)
        target = target.to(args.device)

        y_pred = model(data)

        loss = criterion(y_pred.float(), target.float())

        # Output batch might need changing based on multi-class or
        # multi-label output
        outputs_batch = (y_pred.data.cpu().numpy()
                         > params.threshold).astype(np.int32)
        targets_batch = (target.data.cpu().numpy()).astype(np.int32)

        accumulate.update(outputs_batch, targets_batch)
        loss_batch.append(loss.item())

        del data
        del target
        del outputs_batch
        del targets_batch
        del y_pred

    outputs, targets = accumulate()

    summary_batch = {metric: metrics[metric](outputs,
                                             targets)
                     for metric in metrics}
    summary_batch['loss_avg'] = sum(loss_batch) * 1./(len(loss_batch))

    return summary_batch
