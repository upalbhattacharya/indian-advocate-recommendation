import logging


def set_logger(path):
    """Set logger to log information to the terminal and the specified path.

    Parameters
    ----------
    log_path : str
        Path to log run-stats to.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        file_handler = logging.FileHandler(path)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s : [%(levelname)s] %(message)s",
            "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter(
            "%(asctime)s : [%(levelname)s] %(message)s",
            "%Y-%m-%d %H:%M:%S"))
        logger.addHandler(stream_handler)
