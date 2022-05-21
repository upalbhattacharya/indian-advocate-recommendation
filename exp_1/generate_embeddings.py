#!/home/workboots/VirtualEnvs/aiml/bin/python3
# -*- encoding: utf-8 -*-
# Birth: 2022-03-05 09:58:40.698068918 +0530
# Modify: 2022-03-07 12:21:58.097904021 +0530

import argparse
import logging
import os
import pickle
from pathlib import Path

import torch

import utils
from data_generator import DataGenerator
from model.net import HANPrediction

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

    # For registering hooks
    # Need to learn more
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook

    # Registering the forward hook
    model.han.register_forward_hook(get_activation('han'))

    # For getting the embeddings

    # Generating embeddings
    for idx, data, _, _ in iter(data_loader.yield_batch()):
        logging.info(f"Generating embeddings for {idx}.")
        data = data.to(args.device)

        y_pred = model(data)

        del data
        del y_pred

        embedding = activation['han'].cpu()

        for doc, embed in zip(idx, embedding):

            # Saving the generated embeddings
            torch.save(embed, os.path.join(args.save_path, f"{doc}.pt"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data_dirs", nargs="+", type=str,
                        default=["data/"],
                        help=("Directory containing training "
                              "and testing cases."))
    parser.add_argument("-t", "--targets_paths", nargs="+", type=str,
                        default=["targets/targets.json"],
                        help="Path to targets json file. Give junk data.")
    parser.add_argument("-e", "--embed_path",
                        default="embeddings/embeddings.pkl",
                        help="Path to embeddings json file.")
    parser.add_argument("-x", "--exp_dir", default="experiments/",
                        help=("Directory to load parameters and to save "
                              "model states and metrics."))
    parser.add_argument("-p", "--params", default="params.json",
                        help="Name of params file to load from exp_dir.")
    parser.add_argument("-de", "--device", type=str, default="cuda",
                        help="Device to run on.")
    parser.add_argument("-r", "--restore_file", default=None,
                        help=("File to reload a saved model."))
    parser.add_argument("-n", "--name", type=str, default="gen_embeds",
                        help="Name of model.")
    parser.add_argument("-s", "--save_path", default="generated/", type=str,
                        help="Path to save generated embeddings.")

    args = parser.parse_args()

    utils.set_logger(os.path.join(args.save_path, f"{args.name}.log"))

    # Selecting the right device to train and evaluate on
    if not torch.cuda.is_available() and args.device == "cuda":
        logging.info("No CUDA cores/support found. Switching to cpu.")
        args.device = "cpu"

    logging.info(f"Device is {args.device}.")

    # Loading model parameters
    params_path = os.path.join(args.exp_dir, "params", f"{args.params}")
    assert os.path.isfile(
            params_path), f"No configuration file found at {params_path}."
    params = utils.Params(params_path)

    # Setting a seed for reproducability
    torch.manual_seed(47)
    if(args.device == "cuda"):
        torch.cuda.manual_seed(47)

    # Creating the training and testing data generators
    data_loader = DataGenerator(
                            data_paths=args.data_dirs,
                            targets_paths=args.targets_paths,
                            embed_path=args.embed_path,
                            batch_size=params.batch_size,
                            max_sent_len=params.max_sent_len,
                            max_sent_num=params.max_sent_num,
                            neg_ratio=params.neg_ratio,
                            give_ids=True)

    model = HANPrediction(input_size=params.input_size,
                          hidden_dim=params.hidden_dim,
                          labels=data_loader.unique_labels,
                          device=args.device)

    model.to(args.device)

    generate_embeddings(model=model,
                        data_loader=data_loader,
                        params=params, args=args)


if __name__ == "__main__":
    main()
