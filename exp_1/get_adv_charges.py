#!/home/workboots/VirtualEnvs/aiml/bin/python3
# -*- encoding: utf-8 -*-
# Birth: 2022-03-07 13:04:28.004734858 +0530
# Modify: 2022-03-07 13:25:33.828149824 +0530

"""Get charges of advocates from case charges."""

import argparse
import json
import os

__author__ = "Upal Bhattacharya"
__license__ = ""
__copyright__ = ""
__version__ = "1.0"
__email__ = "upal.bhattacharya@gmail.com"


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--split_info_path",
                        help="Path to split information.")
    parser.add_argument("-c", "--case_charges_path",
                        help="Path to charges of cases.")
    parser.add_argument("-o", "--output_path",
                        help="Path to save generate data.")

    args = parser.parse_args()

    # Loading advocate split information to get training cases
    with open(args.split_info_path, 'r') as f:
        split_info = json.load(f)

    # Loading the case charges
    with open(args.case_charges_path, 'r') as f:
        case_charges = json.load(f)

    # Getting all the charges
    adv_charges = {
            k: list(set([charge for charges in [case_charges.get(case, [])
                         for case in v["train"]]
                         for charge in charges]))
            for k, v in split_info.items()}

    # Saving
    with open(os.path.join(args.output_path, "adv_charges.json"), 'w') as f:
        json.dump(adv_charges, f, indent=4)


if __name__ == "__main__":
    main()
