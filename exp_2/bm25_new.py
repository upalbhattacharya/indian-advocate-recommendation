#!/usr/bin/env python

"""BM25 computation using standard implementation"""

import argparse
import json
import logging
import os
import re
import sys
from string import punctuation

import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer

from utils import set_logger

pattern = rf"[{punctuation}\s]+"
stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def create_concat_text(doc_id_list, data_path):
    """Load documents and return concatenated document.

    Parameters
    ----------
    doc_id_list : list
        List of document IDs to load and concatenate together.
    data_path : str
        Path to load documents from.

    Returns
    -------
    doc_concat : str
        Concatenated string of advocate's cases.
    doc : dict
        Dictionary containing the processed text of each document.
    """

    docs = {}
    for doc_id in doc_id_list:
        flname = doc_id
        try:
            if (os.path.exists(os.path.join(data_path, f"{flname}.txt"))):
                with open(os.path.join(data_path, f"{flname}.txt"), 'r') as f:
                    #  docs[flname] = f.read().split()
                    docs[flname] = process(f.read())
                    if docs[flname] == '':
                        raise ValueError((f"Found empty document {flname}."
                                          "Documents cannot be empty"))
            else:
                raise FileNotFoundError(f"{flname}.txt not found")
        except FileNotFoundError as f:
            logging.error(repr(f))
            sys.exit(1)
        except ValueError as e:
            logging.error(repr(e))
            sys.exit(1)

    # Concatenating into one document
    doc_concat = [token for doc in docs.values() for token in doc]

    return doc_concat, docs


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path",
                        help="Path to load text data from")
    parser.add_argument("-s", "--split_info",
                        help="Path to load advocate split info from")
    parser.add_argument("-o", "--output_path",
                        help="Path to save generated scores")
    parser.add_argument("-e", "--embedding_path",
                        help="Path to save generated embeddings")
    parser.add_argument("-l", "--log_path", type=str, default=None,
                        help="Path to save generated logs")

    args = parser.parse_args()
    if args.log_path is None:
        args.log_path = args.output_path

    set_logger(os.path.join(args.log_path, "bm25_new"))
    logging.info("Inputs:")
    for name, value in vars(args).items():
        logging.info(f"{name}: {value}")

    with open(args.split_info, 'r') as f:
        split_info = json.load(f)

    adv_concat = {}
    test_doc_ids = set()
    train_texts = {}
    test_texts = {}

    for adv, cases in split_info.items():
        logging.info(f"Generating text representation of {adv}")
        adv_concat[adv], docs = create_concat_text(cases["train"],
                                                   args.data_path)
        train_texts = {**train_texts, **docs}
        test_doc_ids.update(cases["test"])

    for idx in test_doc_ids:
        logging.info(f"Loading text of query {idx}")
        with open(os.path.join(args.data_path, f"{idx}.txt"), 'r') as f:
            test_text = f.read()
        test_texts[idx] = process(test_text)

    bm25 = BM25Okapi(list(adv_concat.values()))
    scores = {}

    for idx, text in test_texts.items():
        logging.info(f"Getting scores for query {idx}")
        scores[idx] = bm25.get_scores(text)
        scores[idx] = {
                k: v for k, v in sorted(zip(adv_concat.keys(), scores[idx]),
                                        key=lambda x: x[1],
                                        reverse=True)}

    vectorizer = TfidfVectorizer()
    logging.info("Generating tf-idf vectors of advocate representations")
    adv_vectors = vectorizer.fit_transform([" ".join(text)
                                            for text in adv_concat.values()])
    logging.info("Generating tf-idf vectors of test representations")
    test_vectors = vectorizer.transform([" ".join(text)
                                         for text in test_texts.values()])

    logging.info("Saving scores")
    with open(os.path.join(args.output_path, "scores.json"), 'w') as f:
        json.dump(scores, f, indent=4)

    adv_embed_path = os.path.join(args.embedding_path, "adv_rep")
    if not os.path.exists(adv_embed_path):
        os.makedirs(adv_embed_path)

    for adv, vector in zip(adv_concat.keys(), adv_vectors):
        logging.info(f"Saving tf-idf rep of {adv}")
        with open(os.path.join(adv_embed_path, f"{adv}.npy"), 'wb') as f:
            np.save(f, np.array(vector))

    test_embed_path = os.path.join(args.embedding_path, "test_rep")
    if not os.path.exists(test_embed_path):
        os.makedirs(test_embed_path)

    for idx, vector in zip(test_texts.keys(), test_vectors):
        logging.info(f"Saving tf-idf rep of {idx}")
        with open(os.path.join(test_embed_path, f"{idx}.npy"), 'wb') as f:
            np.save(f, np.array(vector))


if __name__ == "__main__":
    main()
