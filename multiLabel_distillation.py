# Cross-modal distillation using multi-label image encoder and text encoder

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
from multiLabel_ImageEncoder import MultiLabelImageEncoder
from textEncoder import TextEncoderDecoder
from embeddingText_En_De import *

torch.manual_seed(17)

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


def getDataloaders(transform=None):
    dataset = UCM_Captions(transform=transform, ret_type="image-caption")
    UCM_train_set, UCM_test_set = torch.utils.data.random_split(dataset,
                                                                [int(dataset.__len__() * 0.8), dataset.__len__() -
                                                                 int(dataset.__len__() * 0.8)])
    TrainLoader = DataLoader(UCM_train_set, batch_size=batch_size,
                             collate_fn=AuxPadClass(pad_idx=0, ret_type="image-caption"), shuffle=True)
    TestLoader = DataLoader(UCM_test_set, batch_size=batch_size,
                            collate_fn=AuxPadClass(pad_idx=0, ret_type="image-caption"), shuffle=True)
    return TrainLoader, TestLoader, dataset.vocab


class KL_MSE_Loss(nn.Module):
    def __init__(self):
        super(KL_MSE_Loss, self).__init__()
        self.kl_loss = nn.KLDivLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, output, groundtruth):
        loss1 = self.mse_loss(output, groundtruth)
        # loss2 = self.kl_loss(output, groundtruth)

        return loss1  # + loss2


def primarytest(image_en, text_dec, dataloader, linear):
    image, text = next(iter(dataloader))
    image = image[7].to(device)
    image = image.unsqueeze(0).to(device)
    text = text[7].to(device)

    image_encoding = linear(image_en(image))

    hidden = image_encoding[:, :2048].reshape(2, 1, 1024)
    cell = image_encoding[:, 2048:].reshape(2, 1, 1024)
    features = torch.zeros(1, 1000).to(device)

    predictions = text_dec.inference(features, hidden, cell)

    print("Predictions")
    print(predictions.argmax(dim=2))
    print("Ground Truth")
    print(text)

    return None

# batch_first
def labelsToEncoding(labels, text_model, vocabulary, label_to_string):
    # labels dim = batch_size, num_classes
    embedding = text_model.encoder.embed
    itos = vocabulary.itos
    stoi = vocabulary.stoi
    size_labels = labels.size

    result = torch.zeros((size_labels[0], embedding.embedding_dim))

    label_class = [label_to_string[j] for j in range(size_labels[1])]
    indices = [stoi.get(lc) if stoi.get(lc) else stoi['<PAD>'] for lc in label_class]

    indices_embedding = embedding(indices)

    for idx in range(size_labels[0]):
        result[idx, :] = torch.transpose(torch.matmul(torch.transpose(indices_embedding, 0, 1), labels[idx, :]))

    return result


if __name__ == "__main__":
    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1, 1, 1]),
    ])

    # get dataloaders
    train_dataloader, test_dataloader, vocabulary = getDataloaders(preprocess)

    image_model = MultiLabelImageEncoder(num_classes=num_classes)
    image_model.load_state_dict(torch.load("./saved_models/multi_label_kaggleNetwork.pth.tar")['state_dict'])
    image_model = image_model.to(device)

    text_model = EmbeddingTextEncoderDecoder(embedding_dim=embedding_dim,
                                             en_hidden_size=en_hidden_size,
                                             num_layers=num_layers,
                                             vocab_size=vocab_size,
                                             de_hidden_size=de_hidden_size,
                                             pad_idx=pad_idx,
                                             p=dropout_p,
                                             teacher_force_ratio=teacher_force_ratio).to(device=device)

    text_model.load_state_dict(torch.load("./saved_models/e_LSTM_d_LSTM_epochs20_UCM.pth.tar")['state_dict'])
    text_encoder_model = text_model.encoder.to(device)

    # text_encoder_model returns output, hidden_state, cell_state

    print("Number of trainable parameters(image encoder): ", end="")
    total_params = sum(p.numel() for p in image_model.parameters() if p.requires_grad)
    total_params += sum(p.numel() for p in text_encoder_model.parameters() if p.requires_grad)
    print(total_params)

    criterion = KL_MSE_Loss()

    parameters = list(image_model.parameters()) + list(text_model.parameters())
    optimizer = optim.Adam(parameters, lr=learning_rate)

    loss_vector = []
    outputs = []
    linear_layer = nn.Linear(embedding_dim, 4096).to(device)
    for epoch in range(num_epochs):
        for (image_data, text_data) in train_dataloader:
            image_data = image_data.to(device=device)
            text_data = text_data.to(device=device)

            image_encoding = image_model(image_data)
            _, text_encoding_hidden, text_encoding_cell = text_encoder_model(text_data)
            text_encoding_hidden = text_encoding_hidden.permute(1, 0, 2).to(device=device)
            text_encoding_hidden = text_encoding_hidden.reshape(text_encoding_hidden.shape[0], -1)
            text_encoding_cell = text_encoding_cell.permute(1, 0, 2).to(device=device)
            text_encoding_cell = text_encoding_cell.reshape(text_encoding_cell.shape[0], -1)

            text_encoding = torch.cat((text_encoding_cell, text_encoding_hidden), dim=1)
            # dim of output = (batch_size, feature_dim)

            image_encoding = linear_layer(image_encoding)
            # print(image_encoding.shape)
            # print(text_encoding.shape)

            loss = criterion(image_encoding, text_encoding)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch:{epoch + 1},'
              f' Loss:{loss.item():.4f}')
