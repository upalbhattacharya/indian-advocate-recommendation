#!/home/workboots/VirtualEnvs/aiml/bin/python3
# -*- encoding: utf8 -*-
# Birth: 2022-06-01 13:37:43.400170813 +0530
# Modify: 2022-08-31 14:51:51.416631595 +0530

"""Train doc2vec on training documents."""

import argparse
import json
import logging
import os
import pickle
import re
from itertools import chain
from string import punctuation

import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from utils import process, set_logger

__author__ = "Upal Bhattacharya"
__license__ = ""
__copyright__ = ""
__version__ = "1.1"
__email__ = "upal.bhattacharya@gmail.com"

pattern = rf"[{punctuation}\s]+"
stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def create_pd(split_dict, base_path):
    """Create Pandas Series from dictionary of IDs and base path

    Parameters
    ----------
    split_dict : dict
        Dictionary containing train, db, test and val splits of advocates
    base_path : str
        Base path where texts of the data are stored

    Returns
    -------
    df : pandas.Series
        Pandas Series containing the texts of the training documents

    Remarks
    -------
    Takes a dictionary containing the train, test and validation document
    splits of each advocate and creates a Pandas Series object from the
    training documents. Applies pre-processing on the documents.
    See `process` for more information on pre-processing.
    """

    train_doc_ids = set()
    train_docs_dict = {}

    train_doc_ids = set(chain.from_iterable([cases["train"]
                                             for cases in
                                             split_dict.values()]))
    logging.info(f"{len(train_doc_ids)} documents to be used for training")

    # Loading the document texts into a dictionary
    for idx in train_doc_ids:
        with open(os.path.join(base_path, f"{idx}.txt"), 'r') as f:
            train_docs_dict[idx] = f.read()

    df = pd.Series(train_docs_dict, name="FactText")
    df.index_name = "ID"
    df.reset_index()
    df = df.apply(process)
    return df


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--data_path",
                        help="Directory to load data from.")
    parser.add_argument("-d", "--dict_path",
                        help="Directory to load dictionary from.")
    parser.add_argument("-ep", "--epochs", type=int, default=50,
                        help="Number of epochs to train Doc2Vec models on.")
    parser.add_argument("-o", "--output_path",
                        help="Path to save generated embeddings.")
    parser.add_argument("-l", "--log_path", type=str, default=None,
                        help="Path to save generated logs")

    args = parser.parse_args()

    if args.log_path is None:
        args.log_path = args.output_path

    set_logger(os.path.join(args.log_path, "train"))
    logging.info("Inputs:")
    for name, value in vars(args).items():
        logging.info(f"{name}: {value}")

    with open(args.dict_path, 'r') as f:
        adv_case_splits = json.load(f)

    df = create_pd(adv_case_splits, args.data_path)
    tagged_docs = [TaggedDocument(txt, [idx]) for idx, txt in df.iteritems()]

    logging.info("Training model")
    model = Doc2Vec(
        vector_size=300, epochs=args.epochs)
    model.build_vocab(tagged_docs)
    model.train(
        tagged_docs, total_examples=model.corpus_count,
        epochs=model.epochs)

    logging.info("Saving model vectors and states")
    model.dv.save(os.path.join(args.output_path, "model", "d2v.docvectors"))
    model.wv.save(os.path.join(args.output_path, "model", "d2v.wordvectors"))
    model.save(os.path.join(args.output_path, "model", "d2v.model"))

    if not os.path.exists(os.path.join(args.output_path, "embeddings",
                                       "train_rep")):
        os.makedirs(os.path.join(args.output_path, "embeddings", "train_rep"))

    logging.info("Saving training representations")
    dv = model.dv
    for key in dv.index_to_key:
        with open(os.path.join(args.output_path, "embeddings", "train_rep",
                               f"{key}.npy"), 'wb') as f:
            np.save(f, dv[key])


if __name__ == "__main__":
    main()
