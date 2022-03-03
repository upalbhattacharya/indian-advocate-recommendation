#!/home/workboots/workEnv/bin/python3

"""make_survey_template.py: Create markdown file survey template

__author__ = "Upal Bhattacharya"
__copyright__ = ""
__licencse__ = ""
__version__ = "1.0"
__email__ = "upal.bhattacharya@gmail.com"
"""

import argparse
import json
import logging
import os

from utils import set_logger

parser = argparse.ArgumentParser(description="Create templates of survey.")
parser.add_argument("-a", "--advocate_path",
                    help="Path to load advocate index dictionary from.")
parser.add_argument("-s", "--advocate_section_path",
                    help="Path to load advocate section information from.")
parser.add_argument("-f", "--start", type=int, required=True,
                    help="Advocate Index to start considering from.")
parser.add_argument("-t", "--end", type=int, required=True,
                    help="Last advocate index to consider.")
parser.add_argument("-o", "--output_path",
                    help="Path to save generated template.")


def main():
    args = parser.parse_args()

    set_logger(os.path.join(args.output_path, "make_survey_template.log"))
    logging.info(f"Start index: {args.start}")
    logging.info(f"Last index: {args.end}")

    # Loading required advocates list
    with open(os.path.join(
            args.advocate_path, "selected_advs.json"), 'r') as f:
        advs = json.load(f)

    # Loading advocate section relevant counts
    with open(os.path.join(
            args.advocate_section_path, "adv_num_counts.json"), 'r') as f:
        adv_num_counts = json.load(f)

    # Loading advocate section information
    with open(os.path.join(
            args.advocate_section_path, "adv_sections_pretty.json"), 'r') as f:
        adv_sections = json.load(f)

    # Printing to file
    with open(os.path.join(
            args.output_path, f"survey_advs_{args.start}_{args.end}.md"),
            'w') as f:

        print(r"\newpage", file=f)
        # Heading
        print("# Identification of Legal Areas Advocates "
              f"{args.start} to {args.end}", file=f, end="\n")

        for i in range(args.start, args.end + 1):
            adv = advs[f"{i}"]
            print(f"## Advocate {i}", file=f, end="\n")

            rel = adv_num_counts[adv]["rel"]
            total = adv_num_counts[adv]["total"]

            print(f"Total number of cases: {total}", file=f, end="\n\n")
            print(f"Number of cases with statutes cited: {rel}",
                  file=f, end="\n\n")

            print("### Statutes", file=f, end="\n")

            for section, freq in adv_sections[adv].items():
                print(f"- {section} : {freq}", file=f, end="\n")

            print("\n", file=f)
            print(r"\newpage", file=f)


if __name__ == "__main__":
    main()
