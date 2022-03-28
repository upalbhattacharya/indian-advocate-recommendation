#!/home/workboots/VirtualEnvs/aiml/bin/python3
# -*- encoding: utf-8 -*-
# Birth: 2022-02-21 13:51:41.024560553 +0530
# Modify: 2022-02-24 12:10:42.584234470 +0530

"""Generate batches of documents in accordance with the requirements of batches
for HANs."""

import json
import os
import pickle
from collections import defaultdict
from copy import copy
from random import choices, sample
from typing import Union

import torch


class DataGenerator:
    """Generate batches of data."""

    def __init__(self, data_path, targets_path, embed_path, batch_size,
                 max_sent_len, max_sent_num,):
        self.data_path = data_path
        self.targets_path = targets_path
        self.embedding_path = embed_path
        self.batch_size = batch_size
        self.max_sent_len = max_sent_len
        self.max_sent_num = max_sent_num
        self.text_data = self.get_data()
        self.bins, self.num_sent = self.get_num_sent()

        with open(self.embedding_path, 'rb') as f:
            self.embeddings = pickle.load(f)

        with open(self.targets_path, 'r') as f:
            self.targets_dict = json.load(f)

        self.unique_labels = self.get_unique_labels()
        self.targets = self.create_targets()

    def get_data(self) -> dict:
        """Get textual data from path.

        Returns
        -------
        texts : dict
            Dictionary containing the texts of each document as lists.

        """
        texts = defaultdict(lambda: list)
        for filename in os.listdir(self.data_path):
            # Loading the data
            with open(os.path.join(self.data_path, filename), 'r') as f:
                text = f.read()
            text = text.split("\n")
            text = list(filter(lambda x: x != "", text))
            texts[os.path.splitext(filename)[0]] = text

        return texts

    def get_num_sent(self) -> dict:
        """Get number of sentences of each document.

        Returns
        -------
        num_sent : dict
            Dictionary with the number of sentences for each document.

        """
        num_sent = {}
        for doc in self.text_data:
            num_sent[doc] = len(self.text_data[doc])

        # Sorting the documents by document length
        num_sent = {
                k: v for k, v in sorted(num_sent.items(),
                                        key=lambda x: x[1])}

        bins = sorted(list(set([v for v in num_sent.values()])))

        return bins, num_sent

    def get_unique_labels(self):

        # Keeping it as a list for ordering ??
        unique_labels = list(set([label
                                  for labels in self.targets_dict.values()
                                  for label in labels]))

        # Extra step to ensure consistency with test dataset
        unique_labels = sorted(unique_labels)

        return unique_labels

    def create_targets(self) -> dict:
        """Create target tensors given categorical targets.

        Parameters
        ----------

        Returns
        -------
        targets : dict
            Dictionary containing the target tensors.
        """

        targets = {}
        for doc, v in self.targets_dict.items():
            targets[doc] = torch.tensor([int(label in v)
                                         for label in self.unique_labels])

        return targets

    def fetch_targets(self, batch: list) -> torch.Tensor:
        """Return target tensors for given batch

        Parameters
        ----------
        batch : list
            List of document IDs for a batch.

        Returns
        -------
        targets : torch.nn.Tensor
            Tensor containing the target tensors.
        """

        targets = []

        for doc in batch:
            targets.append(self.targets[doc])

        targets = torch.stack(targets, dim=0)

        return targets

    def remove_docs(self, bins: list, num_sent: dict,
                    batch: list) -> Union[list, dict]:
        """Remove documents from the diven dictionary and recompute the bins.

        Parameters
        ----------
        bins : list
            List of bins.
        num_sent : dict
            Dictionary of documents and their lengths.
        batch : list
            List of documents to remove.

        Returns
        -------
        bins : list
            Updated list of bins.
        num_sent : dict
            Updated list of documents.

        """
        for doc in batch:
            num_sent.pop(doc, None)

        bins = sorted(list(set([v for v in num_sent.values()])))

        return bins, num_sent

    def get_batch_data(self, batch: list) -> list:
        # Getting the text for each document of the batch
        batch_data = []
        for doc in batch:
            with open(os.path.join(self.data_path, f"{doc}.txt"), 'r') as f:
                text = f.read()
            text = text.split("\n")
            text = list(filter(lambda x: x != "", text))
            batch_data.append(text)

        # Pruning number of sentences to least of the shortest document and
        # the maximum number of sentences
        shortest = min([len(v) for v in batch_data])
        if (shortest < self.max_sent_num):
            batch_data = [v[:shortest] for v in batch_data]
        else:
            batch_data = [v[:self.max_sent_num] for v in batch_data]
        return batch_data

    def get_batch_embeddings(self, batch):
        """Generate batch embeddings of the documents in the batch. Prunes
        sentence length to the maximum specified. Pads shorter sentences with
        zeros. Generates for each batch, a 4D tensor of shape
        [batch_size, sentence_num, sentence_len, embedding_dim]

        Parameters
        ----------
        batch :
            Batch of documents whose embeddings are to be generated.
        """
        embedding_dim = list(self.embeddings.values())[0].size(dim=0)
        # Getting the maximum sentence length
        maximum = 0
        for doc in batch:
            for sent in doc:
                words = sent.split()
                length = len(words)
                maximum = max(length, maximum)
        if (maximum > self.max_sent_len):
            maximum = self.max_sent_len

        batch_embed = []
        # Creating embeddings of each document
        for doc in batch:
            doc_embed = []
            for sent in doc:
                # Creating embedding of each sentence in a document
                sent_embed = []
                sent = sent.lower()
                words = sent.split()
                words = list(filter(lambda x: x != "", words))
                length = len(words)
                words = [word.strip("./-?!") for word in words]

                # Pruning to maximum sentence length
                if(len(words) > self.max_sent_len):
                    words = words[:self.max_sent_len]

                # Getting the embedding of each word
                for word in words:
                    sent_embed.append(
                                self.embeddings.get(word,
                                                    torch.zeros(embedding_dim)
                                                    ))
                sent_embed = torch.stack(sent_embed, dim=0)
                # Padding
                if (sent_embed.size(0) < maximum):
                    pad_num = maximum - sent_embed.size(0)
                    padding = torch.zeros(pad_num, embedding_dim)
                    sent_embed = torch.cat((sent_embed, padding), 0)
                # Appending each sentence embedding
                doc_embed.append(sent_embed)
            # Stacking to create document embedding
            doc_embed = torch.stack(doc_embed, dim=0)

            # Appending to batch embedding list
            batch_embed.append(doc_embed)

        # Stacking to get batch_embedding
        batch_embed = torch.stack(batch_embed, dim=0)
        return batch_embed

    def yield_batch(self) -> list:
        """Generate batch IDs based on HAN protocols.

        Returns
        -------

        batch : list
            List of documents for a batch.
        """
        bins_cp = copy(self.bins)
        num_sent_cp = copy(self.num_sent)
        # Yield batches till no other bin remains
        while (len(bins_cp) != 0):

            # Bin dictionary for faster access
            bins_dict = defaultdict(lambda: list())
            for doc, length in num_sent_cp.items():
                bins_dict[length].append(doc)

            bins_dict = {
                    k: v for k, v in sorted(bins_dict.items(),
                                            key=lambda x: x[0])}

            # Get frequency of remaining bins
            freq = [len(v) for v in bins_dict.values()]
            total = len(num_sent_cp.keys())
            weights = [v * 1./total for v in freq]

            # Select a bin
            sel_bin = choices(bins_cp, weights, k=1)
            sel_bin = sel_bin[0]
            idx = bins_cp.index(sel_bin)

            # Check if selected bin has adequate number of documents
            if (freq[idx] >= self.batch_size):
                # Getting the documents of 'sel_bin' length
                docs = bins_dict[sel_bin]
                batch = sample(docs, k=self.batch_size)
                # Need to check if necessary here
                batch = list(batch)
            else:
                # When number of documents for selected bin not sufficient

                # Check if total number of cases remaining is adequate
                if (total <= self.batch_size):
                    docs = [doc for doc in num_sent_cp.keys()]
                    batch = docs

                else:
                    docs = bins_dict[sel_bin]
                    batch = docs
                    to_select = self.batch_size - len(batch)

                    # If bin is close to the end, choose previous bin
                    if (idx >= len(freq) - 1 - self.batch_size):
                        next_bin = -1
                    else:
                        next_bin = 1

                    # Iterate over bins till adequate number of documents
                    # are found for the batch
                    next_idx = idx
                    while(len(batch) < self.batch_size):
                        next_idx = next_idx + next_bin
                        n_bin = bins_cp[next_idx]
                        docs = bins_dict[n_bin]
                        # Check if bin has adequate cases
                        if (len(docs) >= to_select):
                            addn = sample(docs, k=to_select)
                            batch.extend(addn)
                            batch = list(batch)
                        else:
                            batch.extend(docs)
                            batch = list(batch)
                            to_select = self.batch_size - len(batch)

            bins_cp, num_sent_cp = self.remove_docs(bins_cp,
                                                    num_sent_cp,
                                                    batch)
            targets = self.fetch_targets(batch)
            batch = self.get_batch_data(batch)
            batch = self.get_batch_embeddings(batch)
            yield batch, targets


def main():
    data_path = ("/home/workboots/Datasets/DHC/variations/var_1.1/"
                 "data/ipc_data/sentences/")
    embed_path = ("/home/workboots/Datasets/DHC/variations/var_1.1/"
                  "embeddings/word2vec/word2vec_dict_200.pkl")
    targets_path = ("/home/workboots/Datasets/DHC/variations/var_1.1/"
                    "targets/ipc_case_offences.json")
    data_gen = DataGenerator(data_path=data_path,
                             targets_path=targets_path,
                             embed_path=embed_path,
                             batch_size=8,
                             max_sent_len=400,
                             max_sent_num=100)

    for i, (batch, target) in enumerate(data_gen.yield_batch()):
        print(i, batch.shape, target)


if __name__ == "__main__":
    main()
