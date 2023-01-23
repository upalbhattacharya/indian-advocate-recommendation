#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Birth: 2022-09-12 19:41:15.490098971 +0530
# Modify: 2022-09-12 20:09:26.486354864 +0530

"""Evaluation script for BertMultiLabel"""

import numpy as np

import utils


def evaluate(model, loss_fn, data_loader, params, metrics, args, target_names):

    # Set model to eval mode
    model.eval()

    # Accumulate data of batches
    accumulate = utils.Accumulate()
    loss_batch = []

    criterion = loss_fn

    for data, target in data_loader():
        target = target.to(args.device)
        data = [d.float().to(args.device) for d in data]

        y_pred = model(*data)

        loss = criterion(y_pred.float(), target.float())

        outputs_batch = (y_pred.data.cpu().numpy()
                         > params.threshold).astype(np.int32)
        targets_batch = (target.data.cpu().detach().numpy()).astype(np.int32)

        accumulate.update(outputs_batch, targets_batch)
        loss_batch.append(loss.item())

        del data
        del target
        del y_pred
        del outputs_batch
        del targets_batch

    output, targets = accumulate()

    summary_batch = {metric: metrics[metric](output, targets, target_names)
                     for metric in metrics}

    summary_batch["loss_avg"] = sum(loss_batch) * 1./len(loss_batch)

    return summary_batch
