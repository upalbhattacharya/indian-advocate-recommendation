#!/home/workboots/VirtualEnvs/aiml/bin/python3
# -*- encoding: utf8 -*-
# Birth: 2022-06-01 13:37:43.216156496 +0530
# Modify: 2022-09-05 12:07:51.320249315 +0530

"""
Script that trains a bm25 model on the train dataset case files.
Takes the training documents of the high count advocates as given by the
high_count_advs dictionary and trains the bm25 model on it.
"""

import argparse
import json
import logging
import math
import os
import re
import string
import sys
import time
from string import punctuation

import numpy as np
from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from utils import set_logger

__author__ = "Upal Bhattacharya"
__license__ = ""
__copyright__ = ""
__version__ = "1.1"
__email__ = "upal.bhattacharya@gmail.com"

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


def idf(corpus_size, doc_freq):
    """Calculate IDF given document frequency and number of documents.

    Parameters
    ----------
    corpus-size : int
        Number of documents.
    doc_freq : int
        Document frequency of token.

    Returns
    -------
    idf_score : float
        IDF score of the given token.
    """

    return math.log((corpus_size - doc_freq + 0.5)/(doc_freq + 0.5))


def convert_to_token_dict(corpus_dict, idx_dict):
    """Return dictionary after converting keys to appropriate format.

    Parameters
    ----------
    corpus_dict : dict
        Dictionary mapping IDs to other format.
    idx_dict : dict
        ID based dictionary to converting.

    Returns
    -------
    formatted_dict : dict
        Dictionary mapping tokens to their corresponding values.
    """
    formatted_dict = {
        corpus_dict[idx]: value
        for idx, value in iter(sorted(
            idx_dict.items(), key=lambda x: x[1], reverse=True))}

    return formatted_dict


def write_to_dir(obj, paths, name, ext="json"):
    """Write object to path with given extenstion.

    Parameters
    ----------
    obj : object
        Object to be saved.
    paths : list
        Path to save file.
    name : str
        Name to use for saving.
    ext : str, default "json"
        Extension to use for saving.
    """

    with open(os.path.join(*paths, f"{name}.{ext}"), 'w') as f:
        if (ext == 'json'):
            json.dump(obj, f, indent=4)
        else:
            for item in obj:
                print(item, file=f, end='\n')


def get_scores(freqs, per_doc_freqs, per_doc_lens, avg_doc_len,
               idfs, k1=1.5, b=0.75):
    """Calculate BM25 scores of a document against documents in the corpus.

    Parameters
    ----------
    freqs : dict
        Dictionary of tokens and frequencies of document.
    per_doc_freqs : dict
        Dictionary of frequencies of documents in the corpus.
    per_doc_lens : dict
        Length of documents in the corpus.
    avg_doc_len : float
        Average length of documents in the corpus.
    idfs : dict
        IDFs of all tokens.
    k1 : float, default=1.5
        Parameter for BM25.
    b : float, default=0.75
        Parameter for BM25.

    Returns
    -------
    scores : dict
        BM25 scores of documents.
    """

    adv_ordering = [*per_doc_freqs]
    doc_len_array = np.array([per_doc_lens[adv] for adv in adv_ordering])

    scores_array = np.zeros(len(adv_ordering))

    for token, freq in freqs.items():
        token_freq = np.array([per_doc_freqs[doc].get(token, 0)
                               for doc in adv_ordering])
        scores_array += compute_bm25(freq, idfs.get(token, 0),
                                     token_freq, doc_len_array, avg_doc_len,
                                     k1, b)
    scores = {adv: scores_array[i] for i, adv in enumerate(adv_ordering)}
    scores = {
        adv: score
        for adv, score in iter(sorted(
            scores.items(), key=lambda x: x[1], reverse=True))}

    return scores


def compute_bm25(freq, idf_score, token_freq,
                 doc_len_array, avg_doc_len, k1=1.5, b=0.75):
    """Return BM25 score of a token.

    Parameters
    ----------
    freq : int
        Frequency of token in test case.
    idf_score : float
        IDF score of token.
    token_freq : array_like
        Array of frequencies of token in corpus.
    doc_len_array : array_like
        Array of lengths of documents in corpus.
    avg_doc_len : float
        Average length of documents in corpus.
    k1 : float, default=1.5
        Parameter for BM25.
    b : float, default=0.75
        Parameter for BM25.

    Returns
    -------
    bm25_score : array
        BM25 score of token across all document in the corpus.
    """
    return ((idf_score * token_freq * (k1 + 1)
             / (token_freq + k1 * (1 - b + b * doc_len_array / avg_doc_len)))
            * freq)


def get_doc_freqs(doc_dict, corpus_dict, drop_tokens):
    """Return dictionary containing the frequencies of tokens in documents

    Parameters
    ----------
    doc_dict : dict
        Dictionary containing the documents.
    corpus_dict : dict
        Dictionary to convert documents to bow.
    drop_tokens : list
        Tokens to exclude.

    Returns
    -------
    dict_obj : dict
        Dictionary with the documents frequencies
    """

    dict_obj = {
        k: convert_to_token_dict(
            corpus_dict, {idx: v
                          for (idx, v) in corpus_dict.doc2bow(text)
                          if (corpus_dict[idx] not in drop_tokens)})
        for k, text in doc_dict.items()}

    return dict_obj


def get_tf_idf_vector(freqs, inv_doc_freqs, drop_tokens):
    """Create TF-IDF vector representation of document

    Parameters
    ----------
    freqs: dict
        Term frequencies of document
    inv_doc_freqs: dict
        Inverse document frequencies of tokens
    drop_tokens: list
        Tokens to not consider

    Returns
    -------
    tf_idf: numpy.array
        TF-IDF Representation of document
    """
    len = sum(freqs.values())
    tf_idf = {
            k: freqs[k] * 1./len * v if freqs.get(k, -1) != -1 else 0.0
            for k, v in inv_doc_freqs.items() if k not in drop_tokens}
    tf_idf = np.array(list(tf_idf.values()))
    return tf_idf


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file_path",
                        help="Path to load text files from.")
    parser.add_argument("-d", "--dict_path",
                        help="Path to load dicts.")
    parser.add_argument("-o", "--output_path",
                        help="Path to save generated BM25 scores.")
    parser.add_argument("-k1", "--k1_value", type=float, default=1.5,
                        help="k1 value for computing BM25 scores")
    parser.add_argument("-b", "--b_value", type=float, default=0.5,
                        help="b value for computing BM25 scores")
    parser.add_argument("-t", "--threshold", type=float, default=0.80,
                        help="Threshold for tokens to drop")
    parser.add_argument("-l", "--log_path", type=str, default=None,
                        help="Path to save generated logs")
    args = parser.parse_args()

    if args.log_path is None:
        args.log_path = args.output_path

    set_logger(os.path.join(args.log_path, "bm25"))
    for name, value in vars(args).items():
        logging.info(f"{name}: {value}")

    # Loading dictionary containing the train and test splits

    # of the high count advocates
    with open(args.dict_path, 'r') as f:
        high_count_advs = json.load(f)

    adv_concat = {}  # For storing the concatenated advocate texts
    test_doc_ids = set()  # For storing the IDs of the test documents
    train_texts = {}  # For storing the training documents
    test_texts = {}  # For storing the test documents
    scores = {}  # For storing the BM25 scores
    test_doc_freqs = {}  # For storing the word frequencies of test documenst
    train_doc_freqs = {}  # For storing the word frequencies of training
    # documents

    # Creating the concatenated advocate representations to pass to the BM25
    # model
    for adv, cases in high_count_advs.items():

        logging.info(f"Creating meta-document for {adv}")

        # Creating the training corpus
        adv_concat[adv], docs = create_concat_text(cases["train"],
                                                   args.file_path)
        train_texts = {**train_texts, **docs}

        test_doc_ids.update(cases["test"])

    # Getting number of documents in the corpus_freqs
    corpus_size = len([*adv_concat])

    # Creating a dictionary for the dfs and cfs
    logging.info("Creating dictionary for dfs and cfs")
    dictionary = corpora.Dictionary(adv_concat.values())

    # Getting all the frequencies needed for calculating the BM25 scores
    # Conversion from idx:value to token:value
    corpus_freqs = convert_to_token_dict(dictionary, dictionary.cfs)
    doc_freqs = convert_to_token_dict(dictionary, dictionary.dfs)

    # Getting the list of tokens to not consider
    drop_tokens = set(
        [token for token, freq in doc_freqs.items()
         if (freq >= args.threshold * corpus_size)])

    drop_tokens = []

    logging.info(f"{len(drop_tokens)} tokens are being removed")

    _ = [(corpus_freqs.pop(token), doc_freqs.pop(token)) for token in
         drop_tokens]

    # Inverse documents frequencies
    logging.info("Getting IDFs of tokens")
    idf_neg = {
        token: idf(corpus_size, value)
        for token, value in doc_freqs.items()}

    avg_idf = sum(idf_neg.values()) * 1./len(idf_neg.keys())
    epsilon = 0.25

    # Converting away negative idf values
    inv_doc_freqs = {}
    for k, v in idf_neg.items():
        if v < 0:
            inv_doc_freqs[k] = epsilon * avg_idf
        else:
            inv_doc_freqs[k] = v

    # Sorting
    inv_doc_freqs = {
        k: v
        for k, v in iter(sorted(
            inv_doc_freqs.items(), key=lambda x: x[1]))}

    # Per Document Frequencies
    logging.info("Getting per-document token freqs for meta-documents")
    per_doc_freqs = get_doc_freqs(
        adv_concat, dictionary, drop_tokens)

    # Document Lengths
    per_doc_lens = {
        adv: sum(freqs.values())
        for adv, freqs in per_doc_freqs.items()}

    # Number of unique tokens
    unique_doc_lens = {
        adv: len(freqs.keys())
        for adv, freqs in per_doc_freqs.items()}

    # Average Document Length
    avg_doc_len = float(sum(per_doc_lens.values()))/corpus_size

    # Computing the BM25 scores for the test documents
    for idx in test_doc_ids:
        logging.info(f"Computing BM25 scores for {idx}")
        with open(os.path.join(args.file_path, f"{idx}.txt"), 'r') as f:
            test_text = f.read()
        test_texts[idx] = process(test_text)
        # Error for empty test document
        try:
            if (test_texts[idx] == ''):
                raise ValueError((f"Found empty test document {idx}."
                                  "Documents cannot be empty"))
        except ValueError as e:
            logging.error(repr(e))
            sys.exit(1)
        #  test_texts[idx] = test_text.split()

    test_doc_freqs = get_doc_freqs(
        test_texts, dictionary, drop_tokens)

    scores = {
        idx: get_scores(test_doc_freq, per_doc_freqs,
                        per_doc_lens,
                        avg_doc_len,
                        inv_doc_freqs)
        for idx, test_doc_freq in test_doc_freqs.items()}

    train_doc_freqs = get_doc_freqs(
        train_texts, dictionary, drop_tokens)

    if not os.path.exists(os.path.join(args.output_path, "embeddings",
                                       "adv_rep")):
        os.makedirs(os.path.join(args.output_path, "embeddings", "adv_rep"))
        os.makedirs(os.path.join(args.output_path, "embeddings", "train_rep"))
        os.makedirs(os.path.join(args.output_path, "embeddings", "test_rep"))

    # Getting tf-idf vectors
    logging.info("Getting TF-IDF vectors of advocates")
    for adv, freqs in per_doc_freqs.items():
        rep = get_tf_idf_vector(freqs, inv_doc_freqs,
                                drop_tokens)
        with open(os.path.join(args.output_path, "embeddings", "adv_rep",
                               f"{adv}.npy"), 'wb') as f:
            np.save(f, rep)

    logging.info("Getting TF-IDF vectors of training documents")
    for doc, freqs in train_doc_freqs.items():
        rep = get_tf_idf_vector(freqs, inv_doc_freqs,
                                drop_tokens)
        with open(os.path.join(args.output_path, "embeddings", "train_rep",
                               f"{doc}.npy"), 'wb') as f:
            np.save(f, rep)

    logging.info("Getting TF-IDF vectors of test documents")
    for doc, freqs in test_doc_freqs.items():
        rep = get_tf_idf_vector(freqs, inv_doc_freqs,
                                drop_tokens)
        with open(os.path.join(args.output_path, "embeddings", "test_rep",
                               f"{doc}.npy"), 'wb') as f:
            np.save(f, rep)

    write_to_dir(doc_freqs, [args.output_path, "model"], "doc_freqs")
    write_to_dir(corpus_freqs, [args.output_path, "model"], "corpus_freqs")
    write_to_dir(inv_doc_freqs, [args.output_path, "model"], "inv_doc_freqs")
    write_to_dir(per_doc_freqs, [args.output_path, "model"], "per_doc_freqs")
    write_to_dir(per_doc_lens, [args.output_path, "model"], "per_doc_lens")
    write_to_dir(unique_doc_lens, [args.output_path, "model"],
                 "unique_doc_lens")
    write_to_dir(test_doc_freqs, [args.output_path, "model"], "test_doc_freqs")
    write_to_dir(train_doc_freqs, [args.output_path, "model"],
                 "train_doc_freqs")
    write_to_dir(drop_tokens, [args.output_path, "model"], "drop_tokens",
                 "txt")
    write_to_dir(scores, [args.output_path, "results"], "scores")
    dictionary.save(os.path.join(
        args.output_path, "model", "dictionary"))


if __name__ == "__main__":
    main()
