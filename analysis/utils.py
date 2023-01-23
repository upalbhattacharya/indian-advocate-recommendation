#!/usr/bin/env python

"""Utility methods for analysis scripts"""

import json
import logging
import os
from time import strftime


def set_logger(log_path: str, filename: str):
    timestamp = strftime("%Y-%m-%d-%H-%M-%S")
    log_path = os.path.join(log_path, f"{filename}_{timestamp}.log")
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # File Handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s: [%(levelname)s] %(message)s",
            "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(file_handler)

        # Stream Handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(
            "%(asctime)s: [%(levelname)s] %(message)s",
            "%Y-%m-%m %H:%M:%S"))
        logger.addHandler(stream_handler)


def save_dict_to_json(
        save_path: str,
        save_name: str,
        obj: dict):

    with open(os.path.join(save_path, save_name), 'w') as f:
        json.dump(obj, f, indent=4)
