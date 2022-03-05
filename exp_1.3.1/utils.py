#!/home/workboots/VirtualEnvs/aiml/bin/python3
# -*- encoding: utf-8 -*-
# Birth: 2022-03-01 15:34:44.105743638 +0530
# Modify: 2022-03-01 15:34:44.105743638 +0530

"""Utilities for the model."""

import json
import logging
import os
from typing import MutableSequence

import numpy as np
import torch

__author__ = "Upal Bhattacharya"
__license__ = ""
__copyright__ = ""
__version__ = "1.0"
__email__ = "upal.bhattacharya@gmail.com"


class Params:
    """Class that loads hyperparameters from a json file."""

    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    @property
    def dict(self):
        """Give dictionary-like access to Params instace."""
        return self.__dict__


class Accumulate:
    """Maintain all data used in an epoch for metrics calculation."""

    def __init__(self):
        self.output_batch = []
        self.targets_batch = []

    def update(self, output_batch: MutableSequence[float],
               targets_batch: MutableSequence[float]):
        """Update the values of output_batch and targets_batch

        Parameters
        ----------
        output_batch : MutableSequence[float]
            New data to add to output_batch.
        targets_batch : MutableSequence[float]
            New data to add to targets_batch.
        """

        self.output_batch.extend(output_batch.tolist())
        self.targets_batch.extend(targets_batch.tolist())

    def __call__(self):
        """Return output_batch and targets_batch as numpy arrays.
        """

        return(np.stack(self.output_batch, axis=0),
               np.stack(self.targets_batch, axis=0))


def save_checkpoint(state, is_best: bool, save_path: str,
                    to_save: bool = False):
    """Save model state at specified path.

    Parameters
    ----------
    state :
        Model checkpoint to save.
    is_best : bool
        Whether the checkpoint is of the best performance.
    save_path : str
        Path to save model checkpoint to.
    to_save : bool
        Whether to save the checkpoint.
    """

    if not(os.path.exists(save_path)):
        os.makedirs(save_path)
    if is_best:
        torch.save(state, os.path.join(save_path, "best.pth.tar"))
    if to_save:
        torch.save(state, os.path.join(save_path,
                                       f"epoch_{state['epoch']}.pth.tar"))


def load_checkpoint(restore_path: str, model: torch.nn.Module,
                    optimizer: torch.optim = None) -> int:
    """Load model checkpoint and optimizer and return epoch number.

    Parameters
    ----------
    restore_path : str
        Path to load checkpoint.
    model : torch.nn.Module
        Model to apply checkpoint weights to.
    optimizer : torch.optim
        Optimizer to apply checkpoint state to.

    Returns
    -------
    epoch : int
        Epoch number of the checkpoint.
    """

    if not (os.path.exists(restore_path)):
        raise (f"File does note exist at {restore_path}.")
    ckpt = torch.load(restore_path)
    model.load_state_dict(ckpt['state_dict'])

    if optimizer:
        optimizer.load_state_dict(ckpt['optim_dict'])

    return ckpt['epoch']


def save_dict_to_json(dict_obj: dict, save_path: str):
    """Save dictionary object in json format to specified path.

    Parameters
    ----------
    dict_obj : dict
        Dictionary object to save.
    save_path : str
        Path to save dictionary object to.
    """
    if not(os.path.exists(os.path.split(save_path)[0])):
        os.makedirs(os.path.split(save_path)[0])
    with open(save_path, 'w') as f:
        json.dump(dict_obj, f, indent=4)


def set_logger(log_path: str):
    """Generate log file at given path.

    Parameters
    ----------
    log_path : str
        Path to save log file.
    """

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # FileHandler
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s : [%(levelname)s] %(message)s",
            "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(file_handler)

        # StreamHandler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(
            "%(asctime)s : [%(levelname)s] %(message)s",
            "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(stream_handler)
