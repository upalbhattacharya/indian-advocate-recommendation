#!/home/workboots/VirtualEnvs/aiml/python3
# -*- encoding: utf8 -*-
# Birth: 2022-03-01 15:34:37.229076934 +0530
# Modify: 2022-03-01 15:34:37.375743601 +0530

"""
Hierarchical model for processing long texts.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SentenceEncoder(nn.Module):
    """Word to Sentence Encoder"""

    def __init__(self, input_size, hidden_dim, device):
        super(SentenceEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.device = device

        self.gru = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_dim,
                          batch_first=True,
                          bidirectional=True)

        self.wordAttn = WordAttn(input_dim=2 * self.hidden_dim,
                                 device=self.device)

    def forward(self, ip):

        # Reshaping the 4D tensor to a 3D tensor
        batch_size = ip.size(dim=0)
        num_sentences = ip.size(dim=1)
        sentence_length = ip.size(dim=2)
        input_size = ip.size(dim=3)
        ip = ip.view(batch_size*num_sentences, sentence_length, input_size)
        output, _ = self.gru(ip)
        # [batch_size * num_sentences, sentence_length, 2 * hidden_dim]

        # Getting attention-based sentence embeddings
        sentence_embeds = self.wordAttn(output)
        # [batch_size * num_sentences, 2 * hidden_dim]

        # Reshaping to [batch_size, num_sentences, 2 * hidden_dim]
        sentence_embeds = sentence_embeds.view(batch_size, num_sentences, -1)
        # Need to check if this works
        return sentence_embeds


class WordAttn(nn.Module):
    """Word Level Attention for Sentences"""

    def __init__(self, input_dim, device):
        super(WordAttn, self).__init__()
        self.input_dim = input_dim
        self.device = device
        self.linear = nn.Linear(in_features=self.input_dim,
                                out_features=self.input_dim,
                                bias=True)
        self.tanh = nn.Tanh()
        self.context = nn.Parameter(torch.Tensor(self.input_dim, 1))
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.context.data.normal_(mean, std)

    def forward(self, ip):
        output = self.linear(ip)
        # [batch_size * num_sentences, sentence_length, 2 * hidden_dim]
        output = self.tanh(output)
        output = torch.matmul(output, self.context)
        # [batch_size * num_sentences, sentence_length, 1]

        output = torch.squeeze(output, dim=-1)
        # [batch_size * num_sentences, sentence_length]

        attn_weights = F.softmax(output, dim=1)
        # [batch_size * num_sentences , sentence_length]

        sent_embeds = self.element_wise_multiply(ip, attn_weights)

        return sent_embeds

    def element_wise_multiply(self, ip, attn_weights):
        sent_embeds = torch.tensor([])
        sent_embeds = sent_embeds.to(self.device)
        for sentence, weights in zip(ip, attn_weights):
            weights = weights.view(1, -1)
            sentence = torch.squeeze(sentence, dim=0)
            # [1, sentence_length, 2 * hidden_dim]
            # -> [sentence_length, 2 * hidden_dim]

            sent_embed = torch.matmul(weights, sentence)
            # [1, 2 * hidden_dim]

            sent_embeds = torch.cat((sent_embeds, sent_embed), dim=0)
            # [batch_size * num_sentences, 2 * hidden_dim]

        return sent_embeds


class DocumentEncoder(nn.Module):
    """Sentence to Document Encoder"""

    def __init__(self, input_size, hidden_dim, device):
        super(DocumentEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.device = device
        self.gru = nn.GRU(input_size=self.input_size,
                          hidden_size=self.hidden_dim,
                          batch_first=True,
                          bidirectional=True)
        self.sentAttn = SentAttn(input_dim=2 * self.hidden_dim,
                                 device=self.device)

    def forward(self, ip):
        output, _ = self.gru(ip)
        # Output shape = [batch_size, num_sentences, 2 * hidden_dim]

        # Getting attention-based document embeddings
        document_embeds = self.sentAttn(output)
        # [batch_size, 2 * hidden_dim]

        return document_embeds


class SentAttn(nn.Module):
    """Sentence Level Attention for Documents"""

    def __init__(self, input_dim, device):
        super(SentAttn, self).__init__()
        self.input_dim = input_dim
        self.device = device
        self.linear = nn.Linear(in_features=self.input_dim,
                                out_features=self.input_dim,
                                bias=True)
        self.tanh = nn.Tanh()
        self.context = nn.Parameter(torch.Tensor(self.input_dim, 1))
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.context.data.normal_(mean, std)

    def forward(self, ip):
        output = self.linear(ip)
        # [batch_size , num_sentences, 2 * hidden_dim]

        output = self.tanh(output)
        output = torch.matmul(output, self.context)
        # [batch_size, num_sentences, 1]

        output = torch.squeeze(output, dim=-1)
        # [batch_size, num_sentences]

        attn_weights = F.softmax(output, dim=1)
        # [batch_size, num_sentences]

        doc_embeds = self.element_wise_multiply(ip, attn_weights)

        return doc_embeds

    def element_wise_multiply(self, ip, attn_weights):
        doc_embeds = torch.tensor([])
        doc_embeds = doc_embeds.to(self.device)
        for doc, weights in zip(ip, attn_weights):
            weights = weights.view(1, -1)
            # Only squeeze when the first dimension is not 1
            if weights.size(dim=-1) != 1:
                doc = torch.squeeze(doc, dim=0)
            # [1, num_sentences, 2 * hidden_dim]
            # -> [num_sentences, 2 * hidden_dim]

            doc_embed = torch.matmul(weights, doc)
            # [1, 2 * hidden_dim]

            doc_embeds = torch.cat((doc_embeds, doc_embed), dim=0)
            # [batch_size, 2 * hidden_dim]
        return doc_embeds


class HAN(nn.Module):
    """Hierarchical Attention Network Model"""

    def __init__(self, input_size, hidden_dim, device):
        super(HAN, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.device = device
        self.sentEncoder = SentenceEncoder(input_size=self.input_size,
                                           hidden_dim=self.hidden_dim,
                                           device=self.device)

        self.docEncoder = DocumentEncoder(input_size=2 * self.hidden_dim,
                                          hidden_dim=self.hidden_dim,
                                          device=self.device)

    def forward(self, ip):
        sent_embeds = self.sentEncoder(ip)
        doc_embeds = self.docEncoder(sent_embeds)

        return doc_embeds
