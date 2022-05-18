import pandas as pd
from sklearn.manifold import TSNE
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
from fineTunedImageClassifier import *
from textEncoder import TextEncoderDecoder
from embeddingText_En_De import *
from torch.nn.functional import normalize
import time


num_classes = 17
feature_dim = 1000
en_num_layers = 2
de_num_layers = 2
en_hidden_size = 1024
de_hidden_size = 1024
embedding_dim = 300
vocab_size = 309  # to be defined after datasets are loaded
pad_idx = 0
batch_size = 100


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

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

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1, 1, 1]),
])
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
train_dataloader, test_dataloader, vocabulary = getDataloaders(preprocess)
print(f"Number of datapoints in training dataset(approx): {train_dataloader.__len__() * batch_size}")
print(f"Number of datapoints in test dataset(approx): {test_dataloader.__len__() * batch_size}")
image_model = ResnetImageEncoder(num_classes=num_classes)
image_model.classifier.fc = Identity()
image_model.load_state_dict(
    torch.load("./saved_models/distilled_image_encoder.pth.tar")['state_dict'])
image_model = image_model.to(device)


text_model = EmbeddingTextEncoderDecoder(embedding_dim=embedding_dim,
                                         en_hidden_size=en_hidden_size,
                                         num_layers=num_layers,
                                         vocab_size=vocab_size,
                                         de_hidden_size=de_hidden_size,
                                         pad_idx=pad_idx,
                                         p=dropout_p,
                                         teacher_force_ratio=teacher_force_ratio).to(device=device)

text_model.load_state_dict(torch.load("./saved_models/distilled_text_encoder.pth.tar")['state_dict'])
text_encoder_model = text_model.encoder.to(device)
def tsne(image_model,text_model,dataloader, i=7,perple=30):
    tsne = TSNE(n_components=2,random_state=0,perplexity=perple)
    image_model.eval()
    text_model.eval()

    data1=[]
    data2=[]
    label=[]
    with torch.no_grad():
        for image_data,text_data in dataloader:
            image_data = image_data.to(device)
            text_data = text_data.to(device)

            image_encoding = image_model(image_data)
            text_encoding_cell = text_encoder_model(text_data)
            text_encoding_cell = text_encoding_cell.permute(1, 0, 2).to(device=device)
            text_encoding_cell = text_encoding_cell.reshape(text_encoding_cell.shape[0], -1)

            image_encoding = normalize(image_encoding, p=2, dim=1)
            text_encoding_cell = normalize(text_encoding_cell, p=2, dim=1)
            image_encoding=image_encoding.to('cpu').numpy()
            text_encoding_cell = text_encoding_cell.to('cpu').numpy()

            data1 = [*data1, *image_encoding]
            data2 = [*data2, *text_encoding_cell]
    data=data1+data2
    label=['image' if i<len(data1) else 'text' for i in range(len(data1)+len(data2))]
    tsne_data=tsne.fit_transform(data)
    plt.scatter(tsne_data[:len(data1),0],tsne_data[:len(data1),1])
    plt.scatter(tsne_data[len(data1):,0],tsne_data[len(data1):,1])

    print(tsne_data)

    plt.show()
    plt.pause(1)

    return tsne_data, data1


df2, data_1 = tsne(image_model, text_model, train_dataloader)