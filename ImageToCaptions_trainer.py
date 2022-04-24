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
import cv2 as cv

torch.manual_seed(17)

# Hyper parameters
learning_rate = 0.001
batch_size = 16
num_epochs = 40
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


def save_model(model, filename="./saved_models/image_encoder_downstream.pth.tar"):
    state = {'state_dict': model.state_dict()}
    torch.save(state, filename)

    return None


def img2txt(img_encoder, txt_decoder, dataloader, itos, all=False, i=7):
    if all:
        for (img_data, captions) in dataloader:
            img_encoding = img_encoder(img_data)
            output = txt_decoder.inference(img_encoding)
            predictions = output.argmax(dim=2)
            return predictions, captions
    else:
        input, captions = next(iter(dataloader))
        input = input[i]
        input_img = input.detach().permute(1, 2, 0).numpy()
        input_img = input_img + 0.5

        input = input.unsqueeze(0).to(device)
        captions = captions[i]
        img_encoding = img_encoder(input)
        img_encoding = img_encoding.reshape(2, 1, 1024).to(device)
        output = txt_decoder.inference(img_encoding, 1)
        output = output.squeeze(0)
        output = output.argmax(1)
        cv.imshow("Input Image", input_img)

        print("Predicted")
        print([itos[int(e)] for e in output.tolist()])

        print("Ground Truth")
        print([itos[int(e)] for e in captions.tolist()])

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

    image_model = ResnetImageEncoder(num_classes=num_classes)

    image_model = image_model.to(device)
    image_model.classifier.fc = Identity()
    image_model.load_state_dict(
        torch.load("./saved_models/distilled_image_encoder.pth.tar")['state_dict'])

    text_model = EmbeddingTextEncoderDecoder(embedding_dim=embedding_dim,
                                             en_hidden_size=en_hidden_size,
                                             num_layers=num_layers,
                                             vocab_size=vocab_size,
                                             de_hidden_size=de_hidden_size,
                                             pad_idx=pad_idx,
                                             p=dropout_p,
                                             teacher_force_ratio=teacher_force_ratio).to(device=device)

    text_model.load_state_dict(torch.load("./saved_models/Embed_e_LSTM_d_LSTM_UCM.pth.tar")['state_dict'])
    text_encoder_model = text_model.encoder.to(device)
    text_model.decoder = text_model.decoder.to(device)

    # text_encoder_model returns output, hidden_state, cell_state
    ins_weights = torch.tensor(vocabulary.weights, requires_grad=False).to(device)
    print(len(ins_weights))
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, weight=ins_weights)
    params = list(image_model.parameters()) + list(text_model.decoder.parameters())
    optimizer = optim.Adam(params, lr=learning_rate)

    print("Number of trainable parameters(image encoder): ", end="")
    total_params = sum(p.numel() for p in image_model.parameters() if p.requires_grad)
    total_params += sum(p.numel() for p in text_encoder_model.parameters() if p.requires_grad)
    print(total_params)
    loss_vector = []
    outputs = []

    ext = False
    for epoch in range(num_epochs):
        if not ext:
            for (j, (image_data, text_data)) in enumerate(train_dataloader):
                image_data = image_data.to(device=device)
                text_data = text_data.to(device=device)
                batch_size = image_data.shape[0]
                image_encoding = image_model(image_data)
                image_encoding = image_encoding.reshape(2, batch_size, 1024).to(device)

                predictions = text_model.decoder.forward(image_encoding, text_data)
                predictions = predictions.permute(0, 2, 1).to(device=device)
                loss = criterion(predictions, text_data)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            if epoch % 10 == 0:
                x = ""
                while x != "C" and x != "M" and x != "E":
                    x = input("Commands(C:Continue;S:Save;T:Test;E:Exit;M:More Epochs)")
                    if x == "S":
                        save_model(image_model)
                        save_model(text_model.decoder, filename="./saved_models/text_decoder_downstream.pth.tar")
                    if x == "T":
                        img2txt(image_model, text_model.decoder, test_dataloader, vocabulary.itos)
                    if x == "E":
                        ext = True

            loss_vector.append(loss.item())
            img2txt(image_model, text_model.decoder, test_dataloader, vocabulary.itos)

            print(f'Epoch:{epoch + 1},'
                  f' Loss:{loss.item():.4f}')

    cv.waitKey(0)
    # closing all open windows
    cv.destroyAllWindows()
    img2txt(image_model, text_model.decoder, test_dataloader, vocabulary.itos)
