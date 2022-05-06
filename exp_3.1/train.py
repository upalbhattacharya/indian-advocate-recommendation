#!/home/workboots/VirtualEnvs/aiml/bin/python3
# -*- encoding: utf8 -*-
# Birth: 2021-12-28 11:17:05.197975110 +0530
# Modify: 2022-05-03 11:57:18.494711259 +0530

"""Train doc2vec on training documents."""

import argparse
import json
import os
import pickle
from string import punctuation
import re

import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

__author__ = "Upal Bhattacharya"
__license__ = ""
__copyright__ = ""
__version__ = "1.1"
__email__ = "upal.bhattacharya@gmail.com"

pattern = rf"[{punctuation}\s]+"
stopwords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def process(text):
    """Carry out processing of given text."""
    processed = list(filter(None, [re.sub('[^0-9a-zA-Z]+', '',
                                          token.lower())
                                   for token in re.split(pattern, text)]))

    # Removing tokens of length 1
    processed = [token for token in processed if len(token) > 1]

    return processed


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

    Notes
    -----
    Takes a dictionary containing the train, test and validation document
    splits of each advocate and creates a Pandas Series object from the
    training documents. Applies pre-processing on the documents.
    See `process` for more information on pre-processing.
    """

    train_doc_ids = set()
    train_docs_dict = {}

    _ = [train_doc_ids.update([
        idx for idx in [*cases["train"], *cases["db"]]])
        for adv, cases in split_dict.items()]

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

    args = parser.parse_args()

    with open(args.dict_path, 'r') as f:
        adv_case_splits = json.load(f)

    db_cases = list(set([case for adv in adv_case_splits.values()
                         for case in adv["db"]]))

    df = create_pd(adv_case_splits, args.data_path)
    tagged_docs = [TaggedDocument(txt, [idx]) for idx, txt in df.iteritems()]

    model = Doc2Vec(
        vector_size=300, epochs=args.epochs)
    model.build_vocab(tagged_docs)
    model.train(
        tagged_docs, total_examples=model.corpus_count,
        epochs=model.epochs)

    model.dv.save(os.path.join(args.output_path, "d2v.docvectors"))
    model.wv.save(os.path.join(args.output_path, "d2v.wordvectors"))
    model.save(os.path.join(args.output_path, "d2v.model"))

    dv = model.dv
    db_dict = {}
    train_dict = {}
    for key in dv.index_to_key:
        if (key in db_cases):
            db_dict[key] = dv[key]
        else:
            train_dict[key] = dv[key]

    with open(os.path.join(args.output_path, "db_rep.pkl"), 'wb') as f:
        pickle.dump(db_dict, f)

    with open(os.path.join(args.output_path, "train_rep.pkl"), 'wb') as f:
        pickle.dump(train_dict, f)


if __name__ == "__main__":
    main()
