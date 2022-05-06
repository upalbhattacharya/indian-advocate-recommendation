#!/home/workboots/VirtualEnvs/aiml/bin/python3
# -*- encoding: utf-8 -*-
# Birth: 2021-12-28 11:17:05.757975113 +0530
# Modify: 2022-05-03 13:10:00.818160750 +0530

"""Create representations of advocates and test and validation cases"""

import argparse
import json
import os
import pickle
import pandas as pd
from gensim.models.doc2vec import Doc2Vec

from utils import process

__author__ = "Upal Bhattacharya"
__license__ = ""
__copyright__ = ""
__version__ = "1.1"
__email__ = "Upal Bhattacharya"


def create_adv_df(dict_obj, data_path):

    adv_dict = {
        k: " ".join([open(os.path.join(data_path,
                                       f"{case}.txt"), 'r').read()
                     for case in v["db"]
                     if os.path.exists(os.path.join(data_path,
                                                    f"{case}.txt"))])
        for k, v in dict_obj.items()}

    df = pd.Series(adv_dict, name="AdvText")
    del adv_dict
    df.index_name = "Adv"
    df.reset_index()
    df = df.apply(process)
    return df


def create_df(dict_obj, data_path, split):

    cases = set()
    _ = cases.update([case for _, v in dict_obj.items()
                      for case in v[split]])

    case_dict = {
        case: open(os.path.join(
            data_path, f"{case}.txt"), 'r').read()
        for case in cases
        if os.path.exists(os.path.join(data_path,
                                       f"{case}.txt"))}

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
    parser.add_argument("-o", "--output_path", default=None,
                        help=("Directory to save generated representations."
                              "Model is loaded from this directory."))

    args = parser.parse_args()

    with open(args.dict_path, 'r') as f:
        adv_case_splits = json.load(f)

    adv_df = create_adv_df(adv_case_splits, args.data_path)
    test_df = create_df(adv_case_splits, args.data_path, "test")
    val_df = create_df(adv_case_splits, args.data_path, "val")

    model = Doc2Vec.load(os.path.join(args.output_path, "d2v.model"))
    adv_df = adv_df.apply(infer_vec, args=(model,))

    test_df = test_df.apply(infer_vec, args=(model,))

    val_df = val_df.apply(infer_vec, args=(model,))

    adv_dict = adv_df.to_dict()
    test_dict = test_df.to_dict()
    val_dict = val_df.to_dict()

    with open(os.path.join(args.output_path, "adv_rep.pkl"), 'wb') as f:
        pickle.dump(adv_dict, f)

    with open(os.path.join(args.output_path, "test_rep.pkl"), 'wb') as f:
        pickle.dump(test_dict, f)

    with open(os.path.join(args.output_path, "val_rep.pkl"), 'wb') as f:
        pickle.dump(val_dict, f)


if __name__ == "__main__":
    main()
