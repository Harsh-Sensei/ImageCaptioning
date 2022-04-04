6# Using embeddings instead of one hot representations for the text

import torch
import torch.nn as nn  # Neural networks like fully connected layers, CNNs, RNNs, LSTMs, GRUs
import torch.optim as optim  # Optimiations like SGD, Adam
import torch.nn.functional as F  # For activation functions like RELU, Tanh, sigmoid
from torch.utils.data import DataLoader  # For dataset management
import torchvision.datasets as datasets  # Standard datasets, has COCO
import torchvision.transforms as transforms  # transformations on dataset
import torchvision.models as models
from PIL import Image
from CustomDatasets import *
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(73)
torch.autograd.set_detect_anomaly(True)

# Device setup for runtime
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# Hyper parameters for network
num_layers = 2
en_hidden_size = 1024
de_hidden_size = 1024
embedding_dim = 300
vocab_size = 322  # to be assigned again after datasets are loaded
dropout_p = 0.5
teacher_force_ratio = 0.5

# Hyper parameters for learning
learning_rate = 0.001
batch_size = 32
num_epochs = 20


class EmbeddingTextEncoder(nn.Module):

    def __init__(self, embedding_dim=300, hidden_size=512, num_layers=2, vocab_size=1000, pad_idx=0, p=0.5):
        super(EmbeddingTextEncoder, self).__init__()

        # Attributes
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.embed = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(p)

        # Encoder LSTM
        self.encoderLSTM = nn.LSTM(input_size=self.embedding_dim,
                                   hidden_size=self.hidden_size,
                                   num_layers=self.num_layers,
                                   dropout=p,
                                   batch_first=True)

    def forward(self, x):
        # x = batch_size, seq_len
        embed_x = self.embed(x)

        # embed_x = batch_size, seq_len, embedding_dim
        init_hidden_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        init_cell_state = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        outputs, (hidden_states, cell_states) = self.encoderLSTM(embed_x, (init_hidden_state, init_cell_state))
        # dim outputs = (batch_size, sequence_len, hidden_size)

        return hidden_states, cell_states


class EmbeddingTextDecoder(nn.Module):
    def __init__(self, embed, input_size, num_layers=2,
                 hidden_size=1024, teacher_force_ratio=0.5, dropout=0.5):
        super(EmbeddingTextDecoder, self).__init__()
        self.embed = embed
        self.input_size = input_size
        self.vocab_size = embed.num_embeddings
        self.embedding_dim = embed.embedding_dim
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.teacher_force_ratio = teacher_force_ratio
        self.dropout = dropout
        self.dropout_layer = nn.Dropout(dropout)

        self.linear_embed_projection = nn.Linear(hidden_size, embedding_dim)
        self.linear_vocab_projection = nn.Linear(hidden_size, vocab_size)
        # Decoder LSTM
        self.decoderLSTM = nn.LSTM(input_size=self.input_size,
                                   hidden_size=self.hidden_size,
                                   num_layers=self.num_layers,
                                   dropout=self.dropout,
                                   batch_first=True)

    def forward(self, hidden, cell, captions):
        batch_size = captions.shape[0]
        target_len = captions.shape[1]
        # target_vocab_size = self.vocab_size

        outputs = torch.zeros(batch_size, target_len, self.vocab_size)

        caption_embed = self.dropout_layer(self.embed(captions))
        lstm_input = caption_embed[:, 0, :].unsqueeze(1)
        # caption_embed batch_size, seq_len, embed_dim

        for t in range(1, target_len):
            output, (hidden, cell) = self.decoderLSTM(lstm_input, (hidden, cell))

            # output dim = (batch_size, 1,vocab_size)
            output_embedding = self.linear_embed_projection(output.squeeze(1))
            output_vocab_probability = self.linear_vocab_projection(output.squeeze(1))
            outputs[:, t, :] = output_vocab_probability

            if random.random() < self.teacher_force_ratio:
                lstm_input = caption_embed[:, t, :].unsqueeze(1)
            else:
                lstm_input = output_embedding.unsqueeze(1)

        return outputs

    def inference(self, hidden, cell, batch_size, target_len=30):

        outputs = torch.zeros(batch_size, target_len, self.vocab_size)

        start_token = torch.ones(batch_size, 1).int()
        lstm_input = self.embed(start_token)
        # caption_embed batch_size, seq_len, embed_dim

        for t in range(1, target_len):
            output, (hidden, cell) = self.decoderLSTM(lstm_input, (hidden, cell))

            # output dim = (batch_size, 1, vocab_size)
            output_embedding = self.linear_embed_projection(output.squeeze(1))
            output_vocab_probability = self.linear_vocab_projection(output.squeeze(1))
            outputs[:, t, :] = output_vocab_probability

            lstm_input = output_embedding.unsqueeze(1)

        return outputs


class EmbeddingTextEncoderDecoder(nn.Module):

    def __init__(self, embedding_dim, en_hidden_size, num_layers,
                 de_hidden_size, vocab_size, pad_idx=0, p=0.5, teacher_force_ratio=0.5):
        super(EmbeddingTextEncoderDecoder, self).__init__()

        self.encoder = EmbeddingTextEncoder(embedding_dim=embedding_dim,
                                            hidden_size=en_hidden_size,
                                            num_layers=num_layers,
                                            vocab_size=vocab_size,
                                            pad_idx=pad_idx,
                                            p=p)

        self.decoder = EmbeddingTextDecoder(embed=self.encoder.embed,
                                            input_size=embedding_dim,
                                            num_layers=num_layers,
                                            hidden_size=de_hidden_size,
                                            dropout=p,
                                            teacher_force_ratio=teacher_force_ratio)

    def forward(self, x):
        # dim x = (batch_size, sequence_len)

        hidden, cell = self.encoder(x)
        prediction = self.decoder(hidden, cell, x)

        return prediction

    def inference(self, x, max_target_length=30):
        hidden, cell = self.encoder(x)
        prediction = self.decoder.inference(hidden, cell, x.size(0), target_len=max_target_length)

        return prediction


def test(dataloader, model, device, i=3):
    input, _ = next(iter(dataloader))
    input = input[i]
    input = input.unsqueeze(0).to(device)
    output = model.inference(input)
    output = output.squeeze(0)
    output = output.argmax(1)

    print(input)
    print(output)

    return None


def save_model(model, filename="./saved_models/Embed_e_LSTM_d_LSTM_UCM.pth.tar"):
    state = {'state_dict': model.state_dict()}
    torch.save(state, filename)

    return None


if __name__ == "__main__":

    # Initializing dataloader
    UCM_train_loader, UCM_test_loader, pad_idx, vocab_size = getTextUCMDataLoader(batch_size=batch_size)

    model = EmbeddingTextEncoderDecoder(embedding_dim=embedding_dim,
                                        en_hidden_size=en_hidden_size,
                                        num_layers=num_layers,
                                        vocab_size=vocab_size,
                                        de_hidden_size=de_hidden_size,
                                        pad_idx=pad_idx,
                                        p=dropout_p,
                                        teacher_force_ratio=teacher_force_ratio).to(device=device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total number of parameters in the model")
    print(total_params)

    loss_vector = []

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

    test(UCM_train_loader, model, device)
