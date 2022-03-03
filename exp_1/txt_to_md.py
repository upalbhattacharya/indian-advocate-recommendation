#!/home/workboots/VirtualEnvs/aiml/bin/python3
# -*- encoding: utf8 -*-
"""
Convert a given text file to markdown by separating along the newline character
"""

import os
import argparse

__author__ = "Upal Bhattacharya"
__copyright__ = ""
__license__ = ""
__version__ = "1.0"
__email__ = "upal.bhattacharya@gmail.com"


def main():
    parser = argparse.ArgumentParser(description="Convert txt to md.")
    parser.add_argument("-i", "--input_path",
                        help="Path to load data from.")
    parser.add_argument("-o", "--output_path",
                        help="Path to store generated data.")
    parser.add_argument("-t", "--title", type=str, required=True,
                        help="Title of the document being created.")

    args = parser.parse_args()

    # Reading the file
    with open(args.input_path, 'r') as f:
        raw = f.read()
    split = raw.split("\n")

    name = os.path.splitext(os.path.basename(args.input_path))[0]

    with open(os.path.join(args.output_path, f"{name}.md"), 'w') as f:
        print(f"# {args.title}", file=f, end="\n\n")
        for line in split:
            print(f"- {line}", file=f, end="\n")


if __name__ == "__main__":
    main()
