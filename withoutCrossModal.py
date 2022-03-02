# training image captioning model without using cross modal-distillation

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
from imageEncoder import *
from textEncoder import *
import random

torch.manual_seed(73)
random.seed(73)

# Hyper parameters
learning_rate = 0.001
batch_size = 32
num_epochs = 20
num_classes = 17
feature_dim = 1000
en_num_layers = 2
de_num_layers = 2
en_hidden_size = 1024
de_hidden_size = 1024
embedding_dim = 300
vocab_size = 322  # to be defined after datasets are loaded
dropout_p = 0.5
teacher_force_ratio = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


class BasicImageCaptioner(nn.Module):
    def __init__(self, image_encoder, text_decoder, num_image_features):
        super(BasicImageCaptioner, self).__init__()
        self.image_encoder = image_encoder
        self.text_decoder = text_decoder
        self.project = nn.Linear(num_image_features, 4096)

    def forward(self, image, captions):
        image_features = self.image_encoder(image)
        projected_image_features = self.project(image_features)
        batch_size = projected_image_features.shape[0]

        hidden = projected_image_features[:, :2048].reshape(2, batch_size, 1024)
        cell = projected_image_features[:, 2048:].reshape(2, batch_size , 1024)
        predictions = self.text_decoder(hidden, cell, captions)

        return predictions





if __name__ == "__main__":

    num_epochs = 0
    batch_size = 32
    num_image_features = 1000

    # Preprocessor for the images
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1, 1, 1]),
    ])

    # get dataloader
    UCM_train_loader, UCM_test_loader, pad_idx, vocab_size = getImageTextUCMDataLoader(batch_size=batch_size,
                                                                                       transform=preprocess,
                                                                                       type="one-one")

    text_decoder = TextDecoder(feature_dim=feature_dim,
                               vocab_size=vocab_size,
                               num_layers=en_num_layers,
                               hidden_size=de_hidden_size,
                               dropout=dropout_p,
                               teacher_force_ratio=teacher_force_ratio)
    image_encoder = ImageEncoder()

    imageCaptioner = BasicImageCaptioner(image_encoder,
                                         text_decoder,
                                         num_image_features).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = optim.Adam(imageCaptioner.parameters(), lr=learning_rate)
    print("Number of trainable parameters(imageCaptioner): ", end="")
    total_params = sum(p.numel() for p in imageCaptioner.parameters() if p.requires_grad)
    print(total_params)

    loss_vector = []
    outputs = []
    for epoch in range(num_epochs):
        for (image, caption) in UCM_train_loader:
            image = image.to(device=device)
            caption = caption.to(device=device)

            predictions = imageCaptioner(image, caption)

            predictions = predictions.permute(0, 2, 1).to(device=device)
            loss = criterion(predictions, caption)

            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(image_encoder.parameters(), max_norm=0.1)

            optimizer.step()

        print(f'Epoch:{epoch + 1}, Loss:{loss.item():.4f}')



