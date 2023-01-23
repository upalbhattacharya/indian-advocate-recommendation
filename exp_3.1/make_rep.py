#!/home/workboots/VirtualEnvs/aiml/bin/python3
# -*- encoding: utf-8 -*-
# Birth: 2022-06-01 13:37:43.400170813 +0530
# Modify: 2022-08-31 17:48:03.739493253 +0530

"""Create representations of advocates and test and validation cases"""

import argparse
import json
import logging
import os
import pickle
from itertools import chain

import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec

from utils import process, set_logger

__author__ = "Upal Bhattacharya"
__license__ = ""
__copyright__ = ""
__version__ = "1.1"
__email__ = "Upal Bhattacharya"


def create_adv_df(dict_obj, data_path):

    adv_concat_text = {}
    case_list = list(chain.from_iterable([cases["train"]
                                          for cases in dict_obj.values()]))
    case_texts = {}

    # Get case texts to avoid double reading
    for case in case_list:
        with open(os.path.join(data_path, f"{case}.txt"), 'r') as f:
            case_texts[case] = f.read()

    for adv, cases in dict_obj.items():
        adv_concat_text[adv] = " ".join([case_texts[case]
                                         for case in cases["train"]])

    df = pd.Series(adv_concat_text, name="AdvText")
    del adv_concat_text
    df.index_name = "Adv"
    df.reset_index()
    df = df.apply(process)
    return df


def create_df(dict_obj, data_path, split):

    cases = set(chain.from_iterable([cases[split]
                                     for cases in dict_obj.values()]))
    case_dict = {}
    for case in cases:
        with open(os.path.join(data_path, f"{case}.txt"), 'r') as f:
            case_dict[case] = f.read()

    df = pd.Series(case_dict, name="CaseText")
    del case_dict
    del cases
    df.index_name = "ID"
    df.reset_index()
    df = df.apply(process)
    return df


def infer_vec(text, model):
    return model.infer_vector(text, epochs=10)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dict_path",
                        help="Path to load dictionary data from.")
    parser.add_argument("-f", "--data_path",
                        help="Directory to load data from.")
    parser.add_argument("-m", "--model_path",
                        help="Path to load trained model from")
    parser.add_argument("-o", "--output_path", default=None,
                        help=("Directory to save generated representations."
                              "Model is loaded from this directory."))
    parser.add_argument("-l", "--log_path", type=str, default=None,
                        help="Path to save generated logs")

    args = parser.parse_args()
    if args.log_path is None:
        args.log_path = args.output_path
    set_logger(os.path.join(args.log_path, "make_rep"))
    logging.info("Inputs:")
    for name, value in vars(args).items():
        logging.info(f"{name}: {value}")

    with open(args.dict_path, 'r') as f:
        adv_case_splits = json.load(f)

    adv_df = create_adv_df(adv_case_splits, args.data_path)
    test_df = create_df(adv_case_splits, args.data_path, "test")

    model = Doc2Vec.load(os.path.join(args.model_path, "d2v.model"))
    adv_df = adv_df.apply(infer_vec, args=(model,))

    test_df = test_df.apply(infer_vec, args=(model,))

    adv_dict = adv_df.to_dict()
    test_dict = test_df.to_dict()

    if not os.path.exists(os.path.join(args.output_path, "adv_rep")):
        os.makedirs(os.path.join(args.output_path, "adv_rep"))

    if not os.path.exists(os.path.join(args.output_path, "test_rep")):
        os.makedirs(os.path.join(args.output_path, "test_rep"))

    logging.info("Saving advocate representations")
    for key in adv_dict:
        logging.info(f"Saving representation for {key}")
        with open(os.path.join(args.output_path,
                               "adv_rep", f"{key}.npy"), 'wb') as f:
            np.save(f, adv_dict[key])

    logging.info("Saving test representations")
    for key in test_dict:
        logging.info(f"Saving representation for {key}")
        with open(os.path.join(args.output_path,
                               "test_rep", f"{key}.npy"), 'wb') as f:
            np.save(f, test_dict[key])


if __name__ == "__main__":
    main()
