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
from fineTunedImageClassifier import *
from textEncoder import TextEncoderDecoder
from embeddingText_En_De import *
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import normalize
import time

script_start_time = time.time()

torch.manual_seed(17)

# Hyper parameters
learning_rate_img = 0.001
learning_rate_text = learning_rate_img/100

batch_size = 16
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
pad_idx = 0

writer = SummaryWriter(f"runs/distillation/model_summary_cosine")


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


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class Linear_Proj(nn.Module):
    def __init__(self):
        super(Linear_Proj, self).__init__()
        self.linear_p = nn.Linear(2048, 2048)

    def forward(self, x):
        return self.linear_p(x)




class KL_MSE_Loss(nn.Module):
    def __init__(self):
        super(KL_MSE_Loss, self).__init__()
        self.kl_loss = nn.KLDivLoss()
        self.mse_loss = nn.MSELoss()
        self.cosine = nn.CosineEmbeddingLoss()

    def forward(self, output, groundtruth):
        # loss1 = self.mse_loss(output, groundtruth)
        # loss2 = self.kl_loss(groundtruth, output)
        loss3 = self.cosine(output, groundtruth, torch.ones(output.shape[0]).to(device))
        return loss3  # + loss2


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


def save_model(model, filename="./saved_models/distilled_image_encoder.pth.tar"):
    state = {'state_dict': model.state_dict()}
    torch.save(state, filename)
    print("Model saved: ", filename)
    return None


def img2txt(img_encoder, txt_decoder, dataloader, all=False, i=7):
    if all:
        for (img_data, captions) in dataloader:
            img_encoding = img_encoder(img_data)
            output = txt_decoder.inference(img_encoding)
            predictions = output.argmax(dim=2)
            return predictions, captions
    else:
        input, captions = next(iter(dataloader))
        input = input[i]
        input = input.unsqueeze(0).to(device)
        captions = captions[i]
        img_encoding = img_encoder(input)
        img_encoding = img_encoding.reshape(2, 1, 1024)
        img_encoding = img_encoding.unsqueeze(0).to(device)
        output = txt_decoder.inference(img_encoding, 1)
        output = output.squeeze(0)
        output = output.argmax(1)

        print("Predicted")
        print(output)
        print("Ground Truth")
        print(captions)

        return output, captions


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
    print(f"Number of datapoints in training dataset(approx): {train_dataloader.__len__() * batch_size}")
    print(f"Number of datapoints in test dataset(approx): {test_dataloader.__len__() * batch_size}")
    image_model = ResnetImageEncoder(num_classes=num_classes)
    image_model.load_state_dict(
        torch.load("./saved_models/multi_label_image_classifier_resnet_fine_tuned.pth.tar")['state_dict'])
    image_model = image_model.to(device)

    image_model.classifier.fc = Linear_Proj().to(device)
    text_model = EmbeddingTextEncoderDecoder(embedding_dim=embedding_dim,
                                             en_hidden_size=en_hidden_size,
                                             num_layers=num_layers,
                                             vocab_size=vocab_size,
                                             de_hidden_size=de_hidden_size,
                                             pad_idx=pad_idx,
                                             p=dropout_p,
                                             teacher_force_ratio=teacher_force_ratio).to(device=device)

    text_model.load_state_dict(torch.load("./saved_models/Embed_e_LSTM_d_LSTM_UCM_wo_embed_layer.pth.tar")['state_dict'])
    text_encoder_model = text_model.encoder.to(device)

    # text_encoder_model returns output, hidden_state, cell_state
    # for param in text_encoder_model.parameters():
    #     param.requires_grad = False

    print("Number of trainable parameters(image encoder): ", end="")

    total_params = sum(p.numel() for p in image_model.parameters() if p.requires_grad)
    total_params += sum(p.numel() for p in text_encoder_model.parameters() if p.requires_grad)
    print(total_params)

    criterion = KL_MSE_Loss()

    parameters = list(image_model.parameters()) + list(text_model.parameters())
    optimizer_img = optim.Adam(image_model.parameters(), lr=learning_rate_img)
    optimizer_text = optim.Adam(text_encoder_model.parameters(), lr=learning_rate_text)

    loss_vector = []
    outputs = []

    step = 0

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        for (image_data, text_data) in train_dataloader:
            image_data = image_data.to(device=device)
            text_data = text_data.to(device=device)

            image_encoding = image_model(image_data)
            text_encoding_cell = text_encoder_model(text_data)
            text_encoding_cell = text_encoding_cell.permute(1, 0, 2).to(device=device)
            text_encoding_cell = text_encoding_cell.reshape(text_encoding_cell.shape[0], -1)

            image_encoding = normalize(image_encoding, p=2, dim=1)
            text_encoding_cell = normalize(text_encoding_cell, p=2, dim=1)

            loss = criterion(image_encoding, text_encoding_cell)

            optimizer_img.zero_grad()
            optimizer_text.zero_grad()
            loss.backward()
            optimizer_img.step()
            optimizer_text.step()

            # setting up tensorboard

            writer.add_scalar('MSE loss', loss, global_step=step)
            writer.add_histogram('FC_Classifier', image_model.classifier.fc.linear_p.weight)
            if step % 100 == 0:
                features = torch.cat((image_encoding, text_encoding_cell), dim=0)
                class_labels = ['Image' if id < len(image_encoding) else 'Text' for id in range(len(image_encoding)
                                                                                            + len(text_encoding_cell))]
                writer.add_embedding(features, metadata=class_labels, global_step=step)
            step += 1

        loss_vector.append(loss.item())
        print(f'Epoch:{epoch + 1},'
              f' Loss:{loss.item():.4f}, Epoch time: {time.time() - epoch_start_time}')

    save_model(image_model)
    save_model(text_encoder_model, filename="./saved_models/distilled_text_encoder.pth.tar")
    plt.plot(loss_vector, linestyle='--', marker='o', color='b')
    plt.xlabel("Epochs")
    plt.ylabel("Loss(MSE)")
    plt.show()

