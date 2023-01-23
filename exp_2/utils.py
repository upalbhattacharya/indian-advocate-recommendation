#!/usr/bin/env python

"""Utilities for BM25"""

import logging
import time


def set_logger(log_path: str):
    """Logger"""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    timestamp = time.strftime("%Y-%m-%d-%H-%m-%S")
    log_path = log_path + "_" + timestamp + ".log"

    if not logger.handlers:

        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(
                "%(asctime)s: [%(levelname)s] %(message)s",
                "%Y-%m-%d %H:%M:%S"))

        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(
            "%(asctime)s : [%(levelname)s] %(message)s",
            "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(stream_handler)
