import logging
import time
from functools import wraps
import re
from string import punctuation
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

stopwords = set(stopwords.words("english"))

pattern = rf"[{punctuation}\s]+"
lemmatizer = WordNetLemmatizer()


def time_logger(original_func):
    """Time logging for method."""

    @wraps(original_func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = original_func(*args, **kwargs)
        elapsed = time.time() - start
        logging.info(
            f"Method {original_func.__name__} ran for {elapsed:.3f} sec(s)")
        return result
    return wrapper


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


def clean_names(adv_list):
    """Takes a list of advocate names, a dictionary of advocates with their
    assigned cases and a case text and adds the case to the appropriate
    advocate with the right prefix for petitioner or respondent
    """

    salutations = ['Mr', 'Ms', 'Mrs', 'Dr', 'Mr.', 'Mrs.', 'Ms.', 'Dr.']
    cleaned_advs = []

    for adv in adv_list:
        adv = re.split(r',|\.|\s+', adv)

        adv = list(filter(None, adv))

        # Using replace instead of strip due to abbreviated names with
        adv = [token for token in adv
               if token[0].isupper() and token not in salutations]

        if(len(adv) <= 1):
            continue
        cleaned_advs.append("".join(adv))

    return list(set(cleaned_advs))


def update_dict(d, names_list, fl):
    for name in names_list:
        if(d.get(name, -1) == -1):
            d[name] = [fl, ]
        else:
            d[name].append(fl)
    return d


def process(text):
    """Carry out processing of given text."""
    processed = list(filter(None, [re.sub('[^0-9a-zA-Z]+', '',
                                          token.lower())
                                   for token in re.split(pattern, text)]))

    # Removing tokens of length 1
    processed = [lemmatizer.lemmatize(token)
                 for token in processed
                 if len(token) > 1 and token not in stopwords]

    return processed
