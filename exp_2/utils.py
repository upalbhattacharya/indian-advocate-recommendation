#!/usr/bin/env python

"""Utilities for BM25"""

import logging


def set_logger(log_path: str):
    """Logger"""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

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
