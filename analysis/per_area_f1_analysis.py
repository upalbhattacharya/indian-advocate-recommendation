#!/usr/bin/env python

"""Calculation of F1 when considering only advocates of a particular
area"""

import argparse
import json
import logging
import os
from collections import defaultdict
from itertools import chain

import numpy as np
from sklearn.metrics import precision_recall_fscore_support as prfs

from utils import save_dict_to_json, set_logger


def main():
    parser = argparse.ArgumentParser(
            description=("Calculation of F1 when considering only advocates "
                         "of a particular area."))

    parser.add_argument("--areas", type=str, required=True,
                        help="Path to list of areas")
    parser.add_argument("--test_cases", type=str, required=True,
                        help="Path to test cases for each advocate")
    parser.add_argument("--train_cases", type=str, required=True,
                        help="Path to training cases for each advocate")
    parser.add_argument("--case_areas", type=str, required=True,
                        help="Path to areas of cases")
    parser.add_argument("--adv_areas", type=str, required=True,
                        help="Path to areas of advocates")
    parser.add_argument("--area_adv_test", type=str, required=True,
                        help="Path to advocates for each area in the test set")
    parser.add_argument("--predictions", type=str, required=True,
                        help="Path to model predictions")
    parser.add_argument("--percent", type=float, default=None,
                        help="Area case percentage to consider advocate")
    parser.add_argument("--targets", type=str, required=True,
                        help="Target advocates for cases")
    parser.add_argument("--output_path", type=str, required=True,
                        help="Directory to save generate outputs")

    args = parser.parse_args()
    set_logger(args.output_path, "per_area_f1_analysis")

    for arg, val in vars(args).items():
        logging.info(f"{arg}: {val}")

    with open(args.areas, 'r') as f:
        areas = f.readlines()
    areas = list(filter(None, map(lambda x: x.strip("\n"), areas)))
    logging.info(f"Number of areas: {len(areas)}")

    with open(args.test_cases, 'r') as f:
        adv_test_cases = json.load(f)
    test_cases = list(set(list(
        chain.from_iterable(adv_test_cases.values()))))
    logging.info(f"Number of test cases: {len(test_cases)}")

    with open(args.train_cases, 'r') as f:
        adv_train_cases = json.load(f)
    train_cases = list(set(list(
        chain.from_iterable(adv_train_cases.values()))))
    logging.info(f"Number of training cases: {len(train_cases)}")

    adv_train_cases_num = {
            k: len(v)
            for k, v in adv_train_cases.items()}

    advs = list(sorted(adv_train_cases_num.keys()))
    logging.info(f"Number of advocates: {len(advs)}")

    with open(args.case_areas, 'r') as f:
        case_areas = json.load(f)
    case_areas = {
            k: v["areas"]
            for k, v in case_areas.items()}

    with open(args.adv_areas, 'r') as f:
        adv_areas = json.load(f)
    adv_areas = {
            k: v["areas"]
            for k, v in adv_areas.items()}

    with open(args.area_adv_test, 'r') as f:
        area_adv_test = json.load(f)

    with open(args.predictions, 'r') as f:
        predictions = json.load(f)

    model_name = os.path.splitext(os.path.basename(args.predictions))[0]

    with open(args.targets, 'r') as f:
        targets = json.load(f)

    area_ranked_advocates = get_ranking(
            areas, case_areas, adv_areas,
            adv_train_cases_num)

    weak_marginal_f1_top_stats = get_marginal(
            targets=targets,
            predictions=predictions,
            advs=advs,
            adv_test_cases=adv_test_cases,
            test_cases=test_cases,
            case_areas=case_areas,
            adv_areas=adv_areas,
            area_adv_test=area_adv_test,
            area_ranked_advocates=area_ranked_advocates,
            percent=args.percent,
            type="weak")

    weak_marginal_f1_top_stats = {
            k: v for k, v in sorted(
                weak_marginal_f1_top_stats.items(),
                key=lambda x: x[1]["macro_f1"],
                reverse=True)}

    hard_marginal_f1_top_stats = get_marginal(
            targets=targets,
            predictions=predictions,
            advs=advs,
            adv_test_cases=adv_test_cases,
            test_cases=test_cases,
            case_areas=case_areas,
            adv_areas=adv_areas,
            area_adv_test=area_adv_test,
            area_ranked_advocates=area_ranked_advocates,
            percent=args.percent,
            type="hard")

    hard_marginal_f1_top_stats = {
            k: v for k, v in sorted(
                hard_marginal_f1_top_stats.items(),
                key=lambda x: x[1]["macro_f1"],
                reverse=True)}

    if args.percent is None:
        save_dict_to_json(
                args.output_path,
                f"weak_marginal_{model_name}.json",
                weak_marginal_f1_top_stats)

        save_dict_to_json(
                args.output_path,
                f"hard_marginal_{model_name}.json",
                hard_marginal_f1_top_stats)
    else:
        save_dict_to_json(
                args.output_path,
                f"weak_marginal_percent_{args.percent}_{model_name}.json",
                weak_marginal_f1_top_stats)

        save_dict_to_json(
                args.output_path,
                f"hard_marginal_percent_{args.percent}_{model_name}.json",
                hard_marginal_f1_top_stats)


def get_ranking(
        areas: list,
        case_areas: dict,
        adv_areas: dict,
        adv_train_cases_num: dict):
    """Rank advocates based on number of cases for each area"""

    area_prop = defaultdict(lambda: dict())

    for adv in adv_areas:
        for area, num in adv_areas[adv].items():
            area_prop[area][adv] = num * 1./adv_train_cases_num[adv]

    for area, vals in area_prop.items():
        area_prop[area] = {
                k: v
                for k, v in sorted(vals.items(),
                                   key=lambda x: x[1],
                                   reverse=True)}
    return area_prop


def get_marginal(
        targets: dict,
        predictions: dict,
        advs: list,
        adv_test_cases: dict,
        test_cases: list,
        case_areas: dict,
        adv_areas: dict,
        area_adv_test: dict,
        area_ranked_advocates: dict,
        percent: float,
        type: str = "weak"):
    """Marginalisation per-area test F1"""
    area_adv_stats = {}

    for area, vals in area_ranked_advocates.items():
        if percent is None:
            rel_adv = area_adv_test[area]
        else:
            rel_adv = [
                    adv
                    for adv in area_adv_test[area]
                    if area_ranked_advocates[area].get(adv, 0) >= percent]
        rel_test_cases = list(set(list(chain.from_iterable(
            [adv_test_cases[adv] for adv in rel_adv]))))
        rel_test_cases = [
                case
                for case in rel_test_cases
                if predictions.get(case, -1) != -1]

        if type == "hard":
            rel_test_cases = [
                    case
                    for case in rel_test_cases
                    if area in case_areas[case]]

        if len(rel_test_cases) == 0:
            stats = {
                    "class_prec": {k: 0.0 for k in rel_adv},
                    "class_rec": {k: 0.0 for k in rel_adv},
                    "class_f1": {k: 0.0 for k in rel_adv},
                    "class_sup": {k: 0.0 for k in rel_adv},
                    "macro_prec": 0.0,
                    "macro_rec": 0.0,
                    "macro_f1": 0.0,
                    "macro_sup": "null",
                    "micro_prec": 0.0,
                    "micro_rec": 0.0,
                    "micro_f1": 0.0,
                    "micro_sup": "null",
                    }
            area_adv_stats[area] = stats
            continue

        vector_pred = np.stack([
            convert_to_vector(predictions[case], rel_adv)
            for case in rel_test_cases], axis=0)
        vector_target = np.stack([
            convert_to_vector(targets[case], rel_adv)
            for case in rel_test_cases], axis=0)

        area_adv_stats[area] = get_stats(
                vector_target,
                vector_pred,
                rel_adv)
    return area_adv_stats


def convert_to_vector(
        pred: list,
        advs: list) -> np.ndarray:
    """Convert prediction to vector format"""

    pred = np.array([
        int(adv in pred)
        for adv in advs])
    return pred


def get_stats(
        targets: np.ndarray,
        predictions: np.ndarray,
        target_names: list) -> dict:
    """Get prediction statistics"""

    class_metrics = prfs(targets, predictions, average=None)
    class_prec, class_rec, class_f1, class_sup = class_metrics

    macro_metrics = prfs(targets, predictions, average='macro')
    macro_prec, macro_rec, macro_f1, macro_sup = macro_metrics

    micro_metrics = prfs(targets, predictions, average='micro')
    micro_prec, micro_rec, micro_f1, micro_sup = micro_metrics

    class_prec = {
            k: float(class_prec[i]) for i, k in enumerate(target_names)}
    class_rec = {
            k: float(class_rec[i]) for i, k in enumerate(target_names)}
    class_f1 = {
            k: float(class_f1[i]) for i, k in enumerate(target_names)}
    class_sup = {
            k: float(class_sup[i]) for i, k in enumerate(target_names)}

    scores = {
        'class_prec': class_prec,
        'class_rec': class_rec,
        'class_f1': class_f1,
        'class_sup': class_sup,
        'macro_prec': float(macro_prec) if macro_prec is not None else macro_prec,
        'macro_rec': float(macro_rec) if macro_rec is not None else macro_rec,
        'macro_f1': float(macro_f1) if macro_f1 is not None else macro_f1,
        'macro_sup': float(macro_sup) if macro_sup is not None else macro_sup,
        'micro_prec': float(micro_prec) if micro_prec is not None else micro_prec,
        'micro_rec': float(micro_rec) if micro_rec is not None else micro_rec,
        'micro_f1': float(micro_f1) if micro_f1 is not None else micro_f1,
        'micro_sup': float(micro_sup) if micro_sup is not None else micro_sup
            }

    return scores


if __name__ == "__main__":
    main()
