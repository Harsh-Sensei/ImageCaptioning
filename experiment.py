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

from multiLabel_ImageEncoder import *
from textEncoder import *


torch.manual_seed(73)

torch.autograd.set_detect_anomaly(True)

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

learning_rate = 0.001

# Device setup for runtime
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')



def getImageEncoder(model_path="./saved_models/multi_label_kaggleNetwork.pth.tar"):
    model = MultiLabelImageEncoder(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path)['state_dict'])
    model = model.to(device)
    model.eval()

    return model

def getTextEncoder(model_path="./saved_models/e_LSTM_d_LSTM_epochs20_UCM.pth.tar"):
    model = TextEncoderDecoder(feature_dim=feature_dim,
                               embedding_dim=embedding_dim,
                               en_hidden_size=en_hidden_size,
                               num_layers=en_num_layers,
                               vocab_size=vocab_size,
                               de_hidden_size=de_hidden_size,
                               p=dropout_p,
                               teacher_force_ratio=teacher_force_ratio).to(device=device)
    model.load_state_dict(torch.load(model_path)['state_dict'])
    model.to(device)
    model.eval()

    return model

def getDataloaders(transform=None):


def testImage_TextEncoder_TextDecoder(model, image, groundtruth):
    image_encoding = model(image)



if __name__ == "__main__":

    # Preprocessor for the images
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1, 1, 1]),
    ])

    # get dataloader


    image_model = getImageEncoder()
    text_model = getTextEncoder()




