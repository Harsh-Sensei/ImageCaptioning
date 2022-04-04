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
from CustomDatasets import UCM_Captions
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(17)


class ImageEncoder(nn.Module):

    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.encoder_model = models.resnet18(pretrained=False)

    def forward(self, x):
        x = self.encoder_model(x)
        x = torch.tanh(x)
        return x


class ImageDecoder(nn.Module):
    def __init__(self):
        super(ImageDecoder, self).__init__()

        self.dfc3 = nn.Linear(encoded_image_size, 4096)
        self.bn3 = nn.BatchNorm1d(4096)
        self.dfc2 = nn.Linear(4096, 4096)
        self.bn2 = nn.BatchNorm1d(4096)
        self.dfc1 = nn.Linear(4096, 256 * 6 * 6)
        self.bn1 = nn.BatchNorm1d(256 * 6 * 6)
        self.upsample1 = nn.Upsample(scale_factor=2)
        self.dconv5 = nn.ConvTranspose2d(256, 256, (3, 3), padding=(0, 0))
        self.dconv4 = nn.ConvTranspose2d(256, 384, (3, 3), padding=(1, 1))
        self.dconv3 = nn.ConvTranspose2d(384, 192, (3, 3), padding=(1, 1))
        self.dconv2 = nn.ConvTranspose2d(192, 64, (5, 5), padding=(2, 2))
        self.dconv1 = nn.ConvTranspose2d(64, 3, (12, 12), stride=(4, 4), padding=(4, 4))

    def forward(self, x):  # dim x = (batch_size, 1000)
        x = self.dfc3(x)
        x = F.relu(x)
        x = F.relu(self.bn3(x))
        x = self.dfc2(x)
        x = F.relu(self.bn2(x))
        x = F.relu(x)
        x = self.dfc1(x)
        x = F.relu(self.bn1(x))
        x = F.relu(x)
        x = x.view(x.shape[0], 256, 6, 6)
        x = self.upsample1(x)
        x = self.dconv5(x)
        x = F.relu(x)
        x = F.relu(self.dconv4(x))
        x = F.relu(self.dconv3(x))
        x = self.upsample1(x)
        x = self.dconv2(x)
        x = F.relu(x)
        x = self.upsample1(x)
        x = self.dconv1(x)

        x = torch.tanh(x)

        return x


class ImageEncoderDecoder(nn.Module):

    def __init__(self):
        super(ImageEncoderDecoder, self).__init__()
        self.encoder = ImageEncoder()
        self.decoder = ImageDecoder()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x


# Model evaluation
def evaluate_model(loader, model):
    model.eval()

    avg_loss = 0
    num_samples = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            avg_loss += criterion(model(x), y).item()
            num_samples += 1

    model.train()

    return avg_loss / num_samples


def save_model(model, filename="./saved_models/e_resnet18_d_custom_UCM.pth.tar"):
    state = {'state_dict': model.state_dict()}
    torch.save(state, filename)

    return None


if __name__ == "__main__":

    # Hyper parameter
    learning_rate = 0.001
    input_image_size = (224, 224, 3)
    batch_size = 32
    num_epochs = 20
    encoded_image_size = 1000

    # Loading the test image
    test_image = Image.open("./dataset/test_image.jpg")



    # Device setup for runtime
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print("cuda") if torch.cuda.is_available() else print("GiveUp")

    # Preprocessor for the images
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1, 1, 1]),
    ])

    # loading the UCM_Captions and Sydney_Captions datasets
    UCM_dataset = UCM_Captions(transform=preprocess)
    # Sydney_dataset = Sydney_Captions(transform=preprocess)

    UCM_len = UCM_dataset.__len__()
    # Sydney_len = Sydney_dataset.__len__()

    # Setting up training and testing data
    UCM_train_set, UCM_test_set = torch.utils.data.random_split(UCM_dataset,
                                                                [int(UCM_len * 0.8), UCM_len - int(UCM_len * 0.8)])
    # Sydney_train_set, Sydney_test_set = torch.utils.data.random_split(Sydney_dataset, [int(Sydney_len * 0.8),
    #                                                                                    Sydney_len - int(
    #                                                                                        Sydney_len * 0.8)])

    # Initializing dataloader
    UCM_train_loader = DataLoader(dataset=UCM_train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    UCM_test_loader = DataLoader(dataset=UCM_test_set, batch_size=batch_size, shuffle=True, num_workers=2)
    # Sydney_train_loader = DataLoader(dataset=Sydney_train_set, batch_size=batch_size, shuffle=True)
    # Sydney_test_loader = DataLoader(dataset=Sydney_test_set, batch_size=batch_size, shuffle=True)

    model = ImageEncoderDecoder()
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print(range(num_epochs))
    loss_vector = []
    outputs = []
    for epoch in range(num_epochs):
        print(UCM_train_loader.__len__())
        for (data, ground_truth) in UCM_train_loader:
            print("here3")
            data = data.to(device=device)
            ground_truth = ground_truth.to(device=device)
            output = model(data)
            loss = criterion(output, ground_truth)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print("here")
        print(f'Epoch:{epoch + 1}, Loss:{loss.item():.4f}')
        outputs.append([epoch, data, output, loss.item()])

    print("Average loss on UCM_captions dataset: " + str(evaluate_model(UCM_test_loader, model)))

    # Plotting the loss vs epoch
    plt.plot([i[3] for i in outputs])
    plt.ylabel("Epochs")
    plt.xlabel("Loss")
    plt.show()
