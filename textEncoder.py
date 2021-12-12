# Text Encoder-Decoder model

import torch
import torch.nn as nn  # Neural networks like fully connected layers, CNNs, RNNs, LSTMs, GRUs
import torch.optim as optim  # Optimiations like SGD, Adam
import torch.nn.functional as F  # For activation functions like RELU, Tanh, sigmoid
from torch.utils.data import DataLoader  # For dataset management
import torchvision.datasets as datasets  # Standard datasets, has COCO
import torchvision.transforms as transforms  # transformations on dataset
import torchvision.models as models
from CustomDatasets import Sydney_Captions, UCM_Captions  # Datasets involving captions
import numpy as np
import spacy

# Hyper parameters
learning_rate = 0.001
embedding_dim = 300
encoded_text_dim = 1000
encoder_hidden_size = 512
encoder_num_layers = 2
decoder_hidden_size = 300
decoder_num_layers = 1
batch_size = 32
num_epochs = 10

# Device setup for runtime
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# loading the UCM_Captions and Sydney_Captions datasets
UCM_dataset = UCM_Captions(ret_type="caption-caption")
Sydney_dataset = Sydney_Captions(ret_type="caption-caption")

UCM_len = UCM_dataset.__len__()
Sydney_len = Sydney_dataset.__len__()

# Setting up training and testing data
UCM_train_set, UCM_test_set = torch.utils.data.random_split(UCM_dataset,
                                                            [int(UCM_len * 0.8), UCM_len - int(UCM_len * 0.8)])
Sydney_train_set, Sydney_test_set = torch.utils.data.random_split(UCM_dataset, [int(Sydney_len * 0.8),
                                                                                Sydney_len - int(Sydney_len * 0.8)])

# Initializing dataloader
UCM_train_loader = DataLoader(dataset=UCM_train_set, batch_size=batch_size, shuffle=True)
UCM_test_loader = DataLoader(dataset=UCM_test_set, batch_size=batch_size, shuffle=True)
Sydney_train_loader = DataLoader(dataset=Sydney_train_set, batch_size=batch_size, shuffle=True)
Sydney_test_loader = DataLoader(dataset=Sydney_test_set, batch_size=batch_size, shuffle=True)


class TextEncoderDecoder(nn.Module):

    def __init__(self, dropout=0.5):
        super(TextEncoderDecoder, self).__init__()

        # Hyper parameters
        self.encoded_features = None
        self.embedding_dim = embedding_dim
        self.encoder_hidden_size = encoder_hidden_size
        self.encoder_num_layers = encoder_num_layers
        self.encoded_text_dim = encoded_text_dim
        self.decoder_hidden_size = decoder_hidden_size
        self.decoder_num_layers = decoder_num_layers
        self.bidirectional_encoder = True

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder LSTM
        self.encoderLSTM = nn.LSTM(input_size=self.embedding_dim,
                                   hidden_size=self.encoder_hidden_size,
                                   num_layers=self.encoder_num_layers,
                                   dropout=dropout,
                                   bidirectional=self.bidirectional_encoder,
                                   batch_first=True)
        self.linearProject = nn.Linear(self.encoder_hidden_size, encoded_text_dim)

        # Decoder LSTM
        self.decoderLSTM = nn.LSTM(input_size=self.encoded_text_dim,
                                   hidden_size=self.decoder_hidden_size,
                                   num_layers=self.decoder_num_layers,
                                   dropout=dropout,
                                   batch_first=True)


    def forward(self, x):
        # dim x = (batch_size, sequence_len, embedding_dim)
        print(x.shape)
        init_hidden_state_encoder = torch.zeros(self.encoder_num_layers, x.size(0), self.encoder_hidden_size).to(device)
        init_cell_state_encoder = torch.zeros(self.encoder_num_layers, x.size(0), self.encoder_hidden_size).to(device)
        outputs, (hidden_states, _) = self.encoderLSTM(x, (init_hidden_state_encoder, init_cell_state_encoder))
        # dim outputs = (batch_size, sequence_len, hidden_size)

        encoded_text = self.linearProject(outputs[:, -1, :])
        self.encoded_features = encoded_text

        print(outputs.shape)

        init_hidden_state_decoder = torch.zeros(self.decoder_num_layers, x.size(0), self.decoder_hidden_size).to(device)
        init_cell_state_decoder = torch.zeros(self.decoder_num_layers, x.size(0), self.decoder_hidden_size).to(device)

        hidden = init_hidden_state_decoder
        cell = init_cell_state_decoder

        encoded_text = encoded_text.unsqueeze(1)

        print(encoded_text.shape)

        for i in range(x.size(1)):

            output, (hidden, cell) = self.decoderLSTM(encoded_text, hidden, cell)
            # dim output = (N, 1, decoder_hidden_size)
            print(output.shape)

            output.unsqueeze(1)
            # dim output = (N, 1, decoder_hidden_size)








        return x


model = TextEncoderDecoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

loss_vector = []
outputs = []
for epoch in range(num_epochs):
    for (data, ground_truth) in UCM_train_loader:
        data = data.to(device=device)
        ground_truth = ground_truth.to(device=device)
        # print("data_shape:")
        # print(data.shape)
        output = model(data)
        loss = criterion(output, ground_truth)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch:{epoch + 1}, Loss:{loss.item():.4f}')
    outputs.append([epoch, data, output, loss.item()])
