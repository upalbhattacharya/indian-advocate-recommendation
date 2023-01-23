#!/home/workboots/VirtualEnvs/aiml/bin/python3
# -*- encoding: utf-8 -*-
# Birth: 2022-09-12 20:55:37.555695841 +0530
# Modify: 2022-09-12 21:56:45.227007590 +0530

import argparse
import logging
import os

import torch

import utils
from data_generator import EnsembleDataset, EnsembleDataLoader
from model.net import EnsembleSelfAttn

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
    for i, (data, doc) in enumerate(data_loader()):
        data = [d.float().to(args.device) for d in data]
        doc = doc[0]
        logging.info(f"Generating embeddings for {doc}.")

        embed = model(*data)

        # Removing from computation graph
        embed = embed.cpu().detach()
        embed = torch.squeeze(embed, dim=0)

        # Saving the generated embeddings
        logging.info(f"Saving representation of {doc}")
        torch.save(embed, os.path.join(args.save_path, f"{doc}.pt"))

        del data
        del embed
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--embed_paths", nargs="+", type=str,
                        default=["data/"],
                        help=("Directory containing training "
                              "and testing embeddings."))
    parser.add_argument("-en", "--embed_names", nargs="+", type=str,
                        help="Names of embeddings")
    parser.add_argument("-ed", "--embed_dims", nargs="+", type=int,
                        help="Dimensionality of different embeddings")
    parser.add_argument("-t", "--target_path", type=str,
                        default="targets/targets.json",
                        help="Path to target file.")
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
                        help="Path to save embeddings")

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
    dataset = EnsembleDataset(embed_paths=args.embed_paths,
                              target_path=args.target_path,
                              mode="generate")

    # Dataloaders
    data_loader = EnsembleDataLoader(dataset=dataset,
                                     batch_size=1,
                                     mode="generate")

    model = EnsembleSelfAttn(proj_dim=params.proj_dim,
                             names=args.embed_names,
                             device=args.device,
                             labels=dataset.unique_labels,
                             input_dims=args.embed_dims
                             )

    model.to(args.device)
    print(model)

    generate_embeddings(model=model,
                        data_loader=data_loader,
                        params=params, args=args)
    logging.info("="*80)


if __name__ == "__main__":
    main()
