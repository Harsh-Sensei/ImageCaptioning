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
from torch.nn.functional import normalize

torch.manual_seed(73)

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
        self.encoderLSTM = nn.LSTM(input_size=self.vocab_size,
                                   hidden_size=self.hidden_size,
                                   num_layers=self.num_layers,
                                   dropout=p,
                                   batch_first=True)

        self.linearProject = nn.Linear(self.hidden_size, output_dim)

    def forward(self, x):

        x_onehot = F.one_hot(x, num_classes=self.vocab_size)

        init_hidden_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        init_cell_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        outputs, (hidden_states, cell_states) = self.encoderLSTM(x_onehot.float(), (init_hidden_state, init_cell_state))
        # dim outputs = (batch_size, sequence_len, hidden_size)

        encoded_text = self.linearProject(outputs.sum(dim=1))

        return encoded_text, hidden_states, cell_states


class TextDecoder(nn.Module):
    def __init__(self, feature_dim=1000, vocab_size=1000, num_layers=2, dropout=0.5,
                 hidden_size=1024, teacher_force_ratio=0.5):
        super(TextDecoder, self).__init__()
        self.hidden_size = vocab_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.feature_dim = feature_dim
        self.hidden_size = hidden_size
        self.teacher_force_ratio = teacher_force_ratio
        # Decoder LSTM
        self.decoderLSTM = nn.LSTM(input_size=self.vocab_size,
                                   hidden_size=self.hidden_size,
                                   num_layers=self.num_layers,
                                   dropout=dropout,
                                   batch_first=True)

        self.linear = nn.Linear(feature_dim, vocab_size)
        self.reverse_linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, hidden, cell, captions):
        batch_size = captions.shape[0]
        target_len = captions.shape[1]
        target_vocab_size = self.vocab_size

        outputs = torch.zeros(batch_size, target_len, target_vocab_size)

        # hidden = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device=device)
        # cell = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device=device)

        # dim features = (batch_size, feature_dim)
        # print("features.shape")
        # print(features.shape)

        input = captions[:, 0]
        input_onehot = F.one_hot(input, num_classes=self.vocab_size)
        outputs[:, 0, :] = input_onehot

        # print("input_onehot.shape")
        # print(input_onehot.shape)
        # input_cat = torch.cat((features, input_onehot), dim=1)
        input_onehot = input_onehot.unsqueeze(1)

        for t in range(1, target_len):
            output, (hidden, cell) = self.decoderLSTM(input_onehot.float(), (hidden, cell))
            
            output = self.reverse_linear(output.squeeze(1))
            outputs[:, t, :] = output
                # output dim = (batch_size, vocab_size)
            # outputx = output.squeeze(1).clone().detach()

            if random.random() < self.teacher_force_ratio:
                input_onehot = F.one_hot(captions[:, t], num_classes=self.vocab_size)
            else:
                output = output.squeeze(1).argmax(dim=1)
                input_onehot = F.one_hot(output, num_classes=self.vocab_size)

            input_onehot = input_onehot.unsqueeze(1)

        return outputs

    def inference(self, hidden, cell):
        max_target_len = 30
        target_vocab_size = self.vocab_size

        outputs = torch.zeros(1, max_target_len, target_vocab_size)

        input = torch.tensor([1])
        input_onehot = F.one_hot(input, num_classes=self.vocab_size)
        #dim input_onehot = 1, vocab_size
        outputs[:, 0, :] = input_onehot.float()

        input_onehot = input_onehot.unsqueeze(1)
        # dim input_onehot = 1, 1, num_classes

        for t in range(1, max_target_len):
            output, (hidden, cell) = self.decoderLSTM(input_onehot.float().to(device), (hidden, cell))
            #dim output = 1, 1, hidden_size
            output = self.reverse_linear(output.squeeze(1))
            # dim output = 1, 1, vocab_size

            outputs[:, t, :] = output.squeeze(1)
            output = output.unsqueeze(1)
            output = output.argmax(dim=2)
            # dim output = 1, 1
            input_onehot = F.one_hot(output, num_classes=self.vocab_size)
            # dim input_onehot = 1, 1, num_classes

            if input_onehot[0, 0, 2] == 1:
                break

        return output


class TextEncoderDecoder(nn.Module):

    def __init__(self, feature_dim, embedding_dim, en_hidden_size, num_layers,
                 de_hidden_size,vocab_size, p=0.5, teacher_force_ratio=0.5):
        super(TextEncoderDecoder, self).__init__()

        self.encoder = TextEncoder(embedding_dim=embedding_dim,
                                   hidden_size=en_hidden_size,
                                   num_layers=num_layers,
                                   output_dim=feature_dim,
                                   vocab_size=vocab_size,
                                   p=p)

        self.decoder = TextDecoder(feature_dim=feature_dim,
                                   vocab_size=vocab_size,
                                   num_layers=num_layers,
                                   hidden_size=de_hidden_size,
                                   dropout=p,
                                   teacher_force_ratio=teacher_force_ratio)

    def forward(self, x):
        # dim x = (batch_size, sequence_len)

        features, hidden, cell = self.encoder(x)
        prediction = self.decoder(hidden, cell, x)

        return prediction

    def inference(self, x):
        features, hidden, cell = self.encoder(x)
        prediction = self.decoder.inference(hidden, cell)

        return prediction

def test(dataloader, model, device, i=3):
    input, _ = next(iter(dataloader))
    input = input[1]
    input = input.unsqueeze(0).to(device)
    output = model.inference(input)
    output = output.squeeze(0)
    output = output.argmax(1)

    print(input)
    print(output)

    return None


def infer(model, dataloader, i=5):
    x, _ = next(iter(dataloader))
    x = x[i]
    x = x.unsqueeze(0)
    x = x.to(device)
    features, hidden, cell = model.encoder(x)
    max_target_len = 30
    target_vocab_size = model.encoder.vocab_size
    outputs = torch.zeros(1, max_target_len, target_vocab_size)
    projected_features = model.decoder.linear(features)
    input = torch.tensor([1])
    input_onehot = F.one_hot(input, num_classes=target_vocab_size)
    # dim input_onehot = 1, vocab_size
    outputs[:, 0, :] = input_onehot.float()
    input_onehot = input_onehot.unsqueeze(1)
    # dim input_onehot = 1, 1, num_classes
    for t in range(1, max_target_len):
        output, (hidden, cell) = model.decoder.decoderLSTM(input_onehot.float().to(device), (hidden, cell))
        # dim output = 1, 1, hidden_size
        output = model.decoder.reverse_linear(output.squeeze(1))
        # dim output = 1, vocab_size
        output = output.unsqueeze(1)
        outputs[:, t, :] = output.squeeze(1)
        # print(output.shape)
        output = output.argmax(dim=2)
        # dim output = 1, 1
        input_onehot = F.one_hot(output, num_classes=target_vocab_size)
        # dim input_onehot = 1, 1, num_classes
        if input_onehot[0, 0, 2] == 1:
            break
    print("Ground Truth:")
    print(x)
    print("Predicted:")
    print(outputs.argmax(dim=2))
    return None


def save_model(model, filename="./saved_models/e_LSTM_d_LSTM_epochs20_UCM.pth.tar"):
    state = {'state_dict': model.state_dict()}
    torch.save(state, filename)

    return None


if __name__ == "__main__":

    # Hyper parameters for learning
    learning_rate = 0.001
    batch_size = 32
    num_epochs = 20

    # Hyper parameters for network
    feature_dim = 1000
    en_num_layers = 2
    de_num_layers = 2
    en_hidden_size = 1024
    de_hidden_size = 1024
    embedding_dim = 300
    vocab_size = None  # to be defined after datasets are loaded
    dropout_p = 0.5
    teacher_force_ratio = 0.5

    # Initializing dataloader
    UCM_train_loader, UCM_test_loader, pad_idx, vocab_size = getTextUCMDataLoader(batch_size=batch_size)

    model = TextEncoderDecoder(feature_dim=feature_dim,
                               embedding_dim=embedding_dim,
                               en_hidden_size=en_hidden_size,
                               num_layers=en_num_layers,
                               vocab_size=vocab_size,
                               de_hidden_size=de_hidden_size,
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

    test(UCM_train_loader, model, device)
