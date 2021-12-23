# Image Encoder-Decoder Model

import torch
import torch.nn as nn  # Neural networks like fully connected layers, CNNs, RNNs, LSTMs, GRUs
import torch.optim as optim  # Optimiations like SGD, Adam
import torch.nn.functional as F  # For activation functions like RELU, Tanh, sigmoid
from torch.utils.data import DataLoader  # For dataset management
import torchvision.datasets as datasets  # Standard datasets, has COCO
import torchvision.transforms as transforms  # transformations on dataset
import torchvision.models as models
from PIL import Image
from CustomDatasets import Sydney_Captions, UCM_Captions
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(17)


class MultiLabel_ImageEncoder(nn.Module):
    def __init__(self, encoder_network, output_dim, num_classes):
        super(MultiLabel_ImageEncoder, self).__init__()
        self.encoder_network = encoder_network
        self.linear = nn.Linear(output_dim, num_classes)
        self.output_dim = output_dim
        self.num_classes = num_classes
        self.sigmoid = F.sigmoid()

    def forward(self, x):
        # dim x = (batch_size, output_dim)

        x = self.encoder_network(x)
        x = self.num_classes
        x = torch.sigmoid(x)

        return x


def getImageEncoder():
    return None


def getDataloaders():
    return None, None





if __name__ == "__main__":

    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # Hyper parameter
    learning_rate = 0.001
    batch_size = 32
    num_epochs = 20
    output_dim = 1000
    num_classes = 2048

    # get image encoder
    image_encoder = getImageEncoder()

    # get dataloaders
    train_dataloader, test_dataloader = getDataloaders()

    model = MultiLabel_ImageEncoder(encoder_network=image_encoder, output_dim=output_dim, num_classes=num_classes)
    model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    loss_vector = []
    outputs = []
    for epoch in range(num_epochs):
        for (data, ground_truth) in train_dataloader:
            data = data.to(device=device)
            ground_truth = ground_truth.to(device=device)

            output = model(data)
            # dim of output = (batch_size, feature_dim)

            loss = criterion(output, ground_truth)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch:{epoch + 1}, Loss:{loss.item():.4f}')
        outputs.append([epoch, data, output, loss.item()])

    # print("Average loss: " + str(evaluate_model(UCM_test_loader, model)))
