#!/usr/bin/env python

"""Area co-occurence of areas"""

import argparse
import json
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(
            description="Area co-occurence")
    parser.add_argument("--case_areas", type=str, required=True,
                        help="Areas of cases")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Path so save data")

    args = parser.parse_args()

    with open(args.case_areas, 'r') as f:
        case_areas = json.load(f)
    

