# Text Encoder-Decoder model

import torch
import torch.nn as nn  # Neural networks like fully connected layers, CNNs, RNNs, LSTMs, GRUs
import torch.optim as optim  # Optimiations like SGD, Adam
import torch.nn.functional as F  # For activation functions like RELU, Tanh, sigmoid
from torch.utils.data import DataLoader  # For dataset management
import torchvision.datasets as datasets  # Standard datasets, has COCO
import torchvision.transforms as transforms  # transformations on dataset
import torchvision.models as models
from CustomDatasets import *  # Datasets involving captions
import numpy as np
import spacy
import random

torch.autograd.set_detect_anomaly(True)

# Device setup for runtime
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


class TextEncoder(nn.Module):

    def __init__(self, embedding_dim=300, hidden_size=512, num_layers=2, vocab_size=1000, output_dim=1000, p=0.5):
        super(TextEncoder, self).__init__()

        self.encoded_features = None
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_dim = output_dim
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, embedding_dim)

        self.dropout = nn.Dropout(p)

        # Encoder LSTM
        self.encoderLSTM = nn.LSTM(input_size=self.embedding_dim,
                                   hidden_size=self.hidden_size,
                                   num_layers=self.num_layers,
                                   dropout=p,
                                   batch_first=True)

        self.linearProject = nn.Linear(self.hidden_size, output_dim)

    def forward(self, x):
        # dim x = batch_size, seq_len
        # print("x dim")
        # print(x.shape)
        x = self.dropout(self.embed(x))
        # print("x dim embed")
        # print(x.shape)
        init_hidden_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        init_cell_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        outputs, (hidden_states, _) = self.encoderLSTM(x, (init_hidden_state, init_cell_state))
        # dim outputs = (batch_size, sequence_len, hidden_size)

        encoded_text = self.linearProject(outputs.sum(dim=1))
        # print("outputs.shape")
        # print(outputs.shape)

        return encoded_text


class TextDecoder(nn.Module):
    def __init__(self, embed, feature_dim=1000, embedding_dim=300, vocab_size=1000, num_layers=2, dropout=0.5,
                 teacher_force_ratio=0.5):
        super(TextDecoder, self).__init__()
        self.hidden_size = vocab_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.feature_dim = feature_dim
        self.embed = embed
        self.teacher_force_ratio = teacher_force_ratio
        # Decoder LSTM
        self.decoderLSTM = nn.LSTM(input_size=self.feature_dim+self.vocab_size,
                                   hidden_size=self.vocab_size,
                                   num_layers=self.num_layers,
                                   dropout=dropout,
                                   batch_first=True)

    def forward(self, features, captions):
        batch_size = features.shape[0]
        target_len = captions.shape[1]
        target_vocab_size = self.vocab_size

        outputs = torch.zeros(batch_size, target_len, target_vocab_size)

        hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device=device)
        cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device=device)

        # dim features = (batch_size, feature_dim)
        # print("features.shape")
        # print(features.shape)

        input = captions[:, 0]
        input_onehot = F.one_hot(input, num_classes=vocab_size)

        # print("input_onehot.shape")
        # print(input_onehot.shape)
        input_cat = torch.cat((features, input_onehot), dim=1)
        input_cat = input_cat.unsqueeze(1)

        for t in range(1, target_len):
            output, (hidden, cell) = self.decoderLSTM(input_cat, (hidden, cell))
            outputs[:, t, :] = output.squeeze(1)
            # output dim = (batch_size, vocab_size)
            outputx = output.squeeze(1).clone().detach()

            if random.random() < self.teacher_force_ratio:
                captions_onehot = F.one_hot(captions[:, t], num_classes=vocab_size)
                next_input = torch.cat((features, captions_onehot), dim=1)
            else:
                outputx = outputx.argmax(dim=1)
                output_onehot = F.one_hot(outputx, num_classes=vocab_size)
                next_input = torch.cat((features, output_onehot), dim=1)

            next_input.unsqueeze_(1)
            input_cat = next_input.clone().detach()

        return outputs


class TextEncoderDecoder(nn.Module):

    def __init__(self, feature_dim, embedding_dim, en_hidden_size, en_num_layers, de_num_layers, vocab_size, p=0.5,
                 teacher_force_ratio=0.5):
        super(TextEncoderDecoder, self).__init__()

        self.encoder = TextEncoder(embedding_dim=embedding_dim,
                                   hidden_size=en_hidden_size,
                                   num_layers=en_num_layers,
                                   output_dim=feature_dim,
                                   vocab_size=vocab_size,
                                   p=p)

        self.decoder = TextDecoder(embed=self.encoder.embed,
                                   feature_dim=feature_dim,
                                   embedding_dim=embedding_dim,
                                   vocab_size=vocab_size,
                                   num_layers=de_num_layers,
                                   dropout=p,
                                   teacher_force_ratio=teacher_force_ratio)

    def forward(self, x):
        # dim x = (batch_size, sequence_len)

        features = self.encoder(x)
        prediction = self.decoder(features, x)

        return prediction

def test(dataloader, model, device):
    input, _ = next(iter(dataloader))
    input = input[3]
    input = input.unsqueeze(0).to(device)
    output = model(input)
    output = output.squeeze(0)
    output = output.argmax(1)

    print(input)
    print(output)

    return None


if __name__ == "__main__":

    # Hyper parameters for learning
    learning_rate = 0.001
    batch_size = 32
    num_epochs = 10

    # Hyper parameters for network
    feature_dim = 1000
    en_num_layers = 2
    de_num_layers = 2
    en_hidden_size = 512
    embedding_dim = 300
    vocab_size = None  # to be defined after datasets are loaded
    dropout_p = 0.5
    teacher_force_ratio = 0.5

    # Initializing dataloader
    UCM_train_loader, UCM_test_loader, pad_idx, vocab_size = getTextUCMDataLoader(batch_size=batch_size)

    model = TextEncoderDecoder(feature_dim=feature_dim,
                               embedding_dim=embedding_dim,
                               en_hidden_size=en_hidden_size,
                               en_num_layers=en_num_layers,
                               de_num_layers=de_num_layers,
                               vocab_size=vocab_size,
                               p=dropout_p,
                               teacher_force_ratio=teacher_force_ratio).to(device=device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_vector = []
    outputs = []

    for epoch in range(num_epochs):
        for (data, ground_truth) in UCM_train_loader:
            data = data.to(device=device)
            ground_truth = ground_truth.to(device=device)

            output = model(data)
            output = output.permute(0, 2, 1).to(device=device)

            loss = criterion(output, ground_truth)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            optimizer.step()

        print(f'Epoch:{epoch + 1}, Loss:{loss.item():.4f}')
        outputs.append([epoch, data, output, loss.item()])
