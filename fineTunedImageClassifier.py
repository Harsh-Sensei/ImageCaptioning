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
from CustomDatasets import *
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(17)

p = torch.ones(1, 3, 256, 256)


class ResnetImageEncoder(nn.Module):
    def __init__(self, num_classes):
        super(ResnetImageEncoder, self).__init__()
        self.drp = nn.Dropout(p=0.5)

        self.classifier = models.resnet18(pretrained=True)

        self.fully_connected_layers = nn.Sequential(self.drp(nn.Linear(2048, 1000)),
                                                    nn.ReLU(),
                                                    self.drp(nn.Linear(1000, 100)),
                                                    nn.ReLU(),
                                                    self.drp(nn.Linear(100, num_classes)))

        self.classifier.fc = self.fully_connected_layers

    def forward(self, x):
        x = self.classifier(x)
        x = torch.sigmoid(x)
        return x

    def inference(self, x):
        prediction = None
        with torch.no_grad():
            prediction = self.forward(x)

        return prediction


def save_model(model, filename="./saved_models/multi_label_kaggleNetwork.pth.tar"):
    state = {'state_dict': model.state_dict()}
    torch.save(state, filename)

    return None


def getDataloaders(transform=None):
    dataset = UCM_Captions(transform=transform, ret_type="image-labels")
    UCM_train_set, UCM_test_set = torch.utils.data.random_split(dataset,
                                                                [int(dataset.__len__() * 0.8), dataset.__len__() -
                                                                 int(dataset.__len__() * 0.8)])
    TrainLoader = DataLoader(UCM_train_set, batch_size=batch_size, shuffle=True)
    TestLoader = DataLoader(UCM_test_set, batch_size=batch_size, shuffle=True)

    return TrainLoader, TestLoader


def infer(model, dataloader, i=7):
    a, target = next(iter(dataloader))
    input = a[i]
    input = input.unsqueeze(0)
    input = input.to(device)
    output = model(input)
    prediction = [0 if elem < 0.4 else 1 for elem in output[0]]
    print("Output")
    print(output)
    print("Predicted")
    print(prediction)
    print("Ground Truth")
    print(target[i])
    for index in range(17):
        if abs(prediction[index] - target[i][index]) < 0.01 and prediction[index] == 1:
            print(index)


def evalF1Score(model, test_dataloader, threshold=0.6):
    total_true_positives = 0
    total_target_positives = 0
    total_predicted_positives = 0
    precision = None
    recall = None
    F1score = None

    model.eval()

    with torch.no_grad():
        for inputs, targets in test_dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            scores = model(inputs)

            # evaluating the predictions
            predictions = scores > threshold

            # evaluating parameters required for prcision and recall
            # precision = true_pos/(true_pos + true_neg) = true_pos/(total_pos_pred)
            # reccall = true_pos/(true_pos + false_neg) = true_pos/(total_pos_targets)

            # print(targets[2])
            # print(predictions[2])
            predictions = predictions.flatten().float()
            targets = targets.flatten()

            total_target_positives += torch.sum(targets == 0).float()
            total_predicted_positives += torch.sum(predictions == 0).float()

            for elem in range(len(targets)):
                if int(targets[elem]) == int(predictions[elem]) and int(targets[elem]) == 1:
                    total_true_positives += 1

        precision = total_true_positives / total_predicted_positives
        recall = total_true_positives / total_target_positives

        F1score = precision * recall / (precision + recall)

    model.train()

    print(f"Recall:{recall:.4f}, Precision:{precision:.4f}, F1score:{F1score:.4f}")

    return F1score


if __name__ == "__main__":
    # get device
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # Hyper parameters

    batch_size = 32
    num_epochs = 10
    num_classes = 17

    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1, 1, 1]),
    ])

    # get dataloaders
    train_dataloader, test_dataloader = getDataloaders(preprocess)

    a, target = next(iter(test_dataloader))
    input1, input2 = a[1], a[7]
    input1 = input1.unsqueeze(0)
    input2 = input2.unsqueeze(0)
    input1 = input1.to(device)
    input2 = input2.to(device)

    model = ResnetImageEncoder(num_classes=num_classes)
    print("Number of trainable parameters: ", end="")
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(total_params)
    model.to(device)
    criterion = nn.BCELoss()

    learning_rate1 = 0.0001
    learning_rate2 = 0.01

    optimizer_finetuning = optim.Adam(model.parameters(), lr=learning_rate1)
    optimizer_fullyconnected = optim.Adam(model.parameters(), lr=learning_rate2)

    loss_vector = []
    outputs1 = []
    outputs2 = []

    print("Outputs of trial inputs:")
    print(model(input1))
    print(model(input2))

    for epoch in range(num_epochs):
        for (data, ground_truth) in train_dataloader:
            data = data.to(device=device)
            ground_truth = ground_truth.to(device=device)

            output = model(data)
            # dim of output = (batch_size, feature_dim)

            loss = criterion(output, ground_truth)

            optimizer_finetuning.zero_grad()
            loss.backward()
            optimizer_finetuning.step()
        outputs1.append(loss.item)
        print(f'Epoch:{epoch + 1},'
              f' Loss:{loss.item():.4f}')
        evalF1Score(model, test_dataloader)

        print("Outputs of trial inputs:")
        print(model(input1))
        print(model(input2))



    for param in model.classifier.parameters():
        param.requires_grad = False

    for param in model.classifier.fc.parameters():
        param.requires_grad = True

    for epoch in range(num_epochs):
        for (data, ground_truth) in train_dataloader:
            data = data.to(device=device)
            ground_truth = ground_truth.to(device=device)

            output = model(data)
            # dim of output = (batch_size, feature_dim)

            loss = criterion(output, ground_truth)

            optimizer_fullyconnected.zero_grad()
            loss.backward()
            optimizer_fullyconnected.step()
        outputs2.append(loss.item())
        print(f'Epoch:{epoch + 1},'
              f' Loss:{loss.item():.4f}')
        evalF1Score(model, test_dataloader)
        print("Outputs of trial inputs:")
        print(model(input1))
        print(model(input2))



