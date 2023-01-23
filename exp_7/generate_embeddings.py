#!/home/workboots/VirtualEnvs/aiml/bin/python3
# -*- encoding: utf-8 -*-
# Birth: 2022-09-17 23:55:22.686297970 +0530
# Modify: 2022-09-18 03:36:26.348057405 +0530

import argparse
import logging
import os
import pickle
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import utils
from data_generator import MultiLabelDataset
from model.net import EmbedMultiLabel

__author__ = "Upal Bhattacharya"
__license__ = ""
__copyright__ = ""
__version__ = "1.0"
__email__ = "upal.bhattacharya@gmail.com"


def generate_embeddings(model, data_loader, params, args):

    if args.restore_file is not None:

        # Loading trained model
        logging.info(f"Found checkpoint at {args.restore_file}. Loading.")
        _ = utils.load_checkpoint(args.restore_file, model, device_id=0) + 1

    # Setting the model to evaluate
    model.eval()

    # For getting the embeddings

    # Generating embeddings
    for idx in range(len(data_loader)):
        doc, data = data_loader[idx]
        data = data.to(args.device)
        logging.info(f"Generating embeddings for {doc}.")

        y_pred, embed = model(data)

        # Removing from computation graph
        embed = embed.cpu().detach()

        # Saving the generated embeddings
        torch.save(embed, os.path.join(args.save_path, f"{doc}.pt"))

        del data
        del y_pred
        del embed
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_path", nargs="+", type=str,
                        default=["data/"],
                        help="Directory containing training embeds")
    parser.add_argument("-t", "--targets_paths", nargs="+", type=str,
                        default=["targets/targets.json"],
                        help="Path to target files.")
    parser.add_argument("-x", "--exp_dir", default="experiments/",
                        help=("Directory to load parameters "
                              " from and save metrics and model states"))
    parser.add_argument("-n", "--name", type=str, required=True,
                        help="Name of model")
    parser.add_argument("-p", "--params", default="params.json",
                        help="Name of params file to load from exp+_dir")
    parser.add_argument("-de", "--device", type=str, default="cuda",
                        help="Device to train on.")
    parser.add_argument("-id", "--device_id", type=int, default=0,
                        help="Device ID to run on if using GPU.")
    parser.add_argument("-r", "--restore_file", default=None,
                        help="Restore point to use.")
    parser.add_argument("-ul", "--unique_labels", type=str, default=None,
                        help="Labels to use as targets.")
    parser.add_argument("-s", "--save_path",
                        help="Path to save generated embeddings")

    args = parser.parse_args()

    # Setting logger
    utils.set_logger(os.path.join(args.exp_dir, f"{args.name}.log"))

    # Selecting correct device to train and evaluate on
    if not torch.cuda.is_available() and args.device == "cuda":
        logging.info("No CUDA cores/support found. Switching to cpu.")
        args.device = "cpu"

    if args.device == "cuda":
        args.device = f"cuda:{args.device_id}"

    logging.info(f"Device is {args.device}.")

    # Loading parameters
    params_path = os.path.join(args.exp_dir, "params", f"{args.params}")
    assert os.path.isfile(params_path), f"No params file at {params_path}"
    params = utils.Params(params_path)

    # Setting seed for reproducability
    torch.manual_seed(47)
    if "cuda" in args.device:
        torch.cuda.manual_seed(47)

    # Datasets
    dataset = MultiLabelDataset(data_paths=args.data_path,
                                targets_paths=args.targets_paths,
                                input_type="np",
                                mode="generate")

    model = EmbedMultiLabel(labels=dataset.unique_labels,
                            device=args.device,
                            input_dim=params.input_dim,
                            embed_dim=params.embed_dim,
                            mode="generate")

    model.to(args.device)

    generate_embeddings(model=model,
                        data_loader=dataset,
                        params=params, args=args)


if __name__ == "__main__":
    main()
