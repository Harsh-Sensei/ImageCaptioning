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
import time

script_start_time = time.time()

from imageEncoder import *
from textEncoder import *

torch.manual_seed(73)

torch.autograd.set_detect_anomaly(True)

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


class ImageEncoderResNet50(nn.Module):
    def __init__(self):
        super(ImageEncoderResNet50, self).__init__()
        self.encoder = models.resnet50(pretrained=False)
        self.encoder.fc = nn.Linear(in_features=2048, out_features=4096)
        self.same_projection = nn.Linear(4096, 4096)

    def forward(self, x):
        x = self.encoder(x)
        x = self.same_projection(x)
        return x

class ImageEncoderExtension(nn.Module):
    def __init__(self, encoder, output_dim, required_output_dim):
        super(ImageEncoderExtension, self).__init__()
        self.encoder = encoder
        self.linear_upscale = nn.Linear(output_dim, required_output_dim)
        self.linear = nn.Linear(required_output_dim, required_output_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.linear_upscale(x)
        x = F.relu(x)
        x = self.linear(x)
        return x


def getImageEncoder(model_path="./saved_models/e_resnet18_d_custom_UCM.pth.tar", pretrained=False):
    if pretrained:
        model = ImageEncoderDecoder()
        model.load_state_dict(torch.load(model_path)['state_dict'])
        encoder = ImageEncoderExtension(model.encoder, 1000, 4096)
        return encoder
    else:
        model = ImageEncoderResNet50().to(device=device)
        return model


def getTextEncoder(model_path="./saved_models/e_LSTM_d_LSTM_UCM.pth.tar"):
    model = TextEncoderDecoder(feature_dim=feature_dim,
                               embedding_dim=embedding_dim,
                               en_hidden_size=en_hidden_size,
                               num_layers=en_num_layers,
                               vocab_size=vocab_size,
                               de_hidden_size=de_hidden_size,
                               p=dropout_p,
                               teacher_force_ratio=teacher_force_ratio).to(device=device)
    model.load_state_dict(torch.load(model_path)['state_dict'])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    return model.encoder, model


def getEncodedImage(image_encoder_model, input):
    # dim input = batch_size, 3, 224, 224

    result = image_encoder_model(input)
    # dim result = batch_size, feature_dim

    return result


def getEncodedText(text_encoder_model, input):
    # dim input = batch_size, seq_len  contains indices of the words in the captions
    result1, result2, result3 = text_encoder_model(input)
    # dim result2, result3 = num_layers, batch_size, hidden_size
    result2 = result2.permute(1, 0, 2)
    result3 = result3.permute(1, 0, 2)
    result2 = result2.flatten(start_dim=1)
    result3 = result3.flatten(start_dim=1)
    result = torch.cat((result2, result3), dim=1)
    # dim = batch_size, 4096
    return result


def save_model(model, filename="./saved_models/e_resnet18_d_custom_UCM.pth.tar"):
    state = {'state_dict': model.state_dict()}
    torch.save(state, filename)

    return None


def inferImageText(image_encoder, text_decoder, dataloader, i=3):
    global vocab_size
    x, target = next(iter(dataloader))
    x = x[i]
    target = target[i]
    x = x.unsqueeze(0)
    x = x.to(device)
    output = image_encoder(x)
    hidden, cell = torch.split(output, [2048, 2048], dim=1)
    hidden = hidden.reshape(1, 2, 1024)
    cell = cell.reshape(1, 2, 1024)
    hidden = hidden.permute(1, 0, 2).to(device=device)
    cell = cell.permute(1, 0, 2).to(device=device)

    max_target_len = 30
    target_vocab_size = vocab_size
    outputs = torch.zeros(1, max_target_len, target_vocab_size)
    input = torch.tensor([1])
    input_onehot = F.one_hot(input, num_classes=target_vocab_size)
    # dim input_onehot = 1, vocab_size
    outputs[:, 0, :] = input_onehot.float()
    input_onehot = input_onehot.unsqueeze(1)
    # dim input_onehot = 1, 1, num_classes
    for t in range(1, max_target_len):
        output, (hidden, cell) = text_decoder.decoderLSTM(input_onehot.float().to(device), (hidden, cell))
        # dim output = 1, 1, hidden_size
        output = text_decoder.reverse_linear(output.squeeze(1))
        # dim output = 1, vocab_size
        output = output.unsqueeze(1)
        outputs[:, t, :] = output.squeeze(1)
        # print(output.shape)
        output = output.argmax(dim=2)
        # dim output = 1, 1
        input_onehot = F.one_hot(output, num_classes=target_vocab_size)
        # dim input_onehot = 1, 1, num_classes
        if input_onehot[0, 0, 2] == 1:
            break
    print("Ground Truth:")
    print(target)
    print("Predicted:")
    print(outputs.argmax(dim=2))

    return None


if __name__ == "__main__":

    num_epochs = 30
    batch_size = 32

    # Preprocessor for the images
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[1, 1, 1]),
    ])

    # get dataloader
    UCM_train_loader, UCM_test_loader, pad_idx, vocab_size = getImageTextUCMDataLoader(batch_size=batch_size,
                                                                                       transform=preprocess,
                                                                                       type="one-one")

    image_encoder = getImageEncoder()
    text_encoder, model = getTextEncoder()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(image_encoder.parameters(), lr=learning_rate)

    loss_vector = []
    outputs = []

    epoch_start_time = time.time()
    for epoch in range(num_epochs):
        for (image, caption) in UCM_train_loader:
            image = image.to(device=device)
            caption = caption.to(device=device)

            image_features = getEncodedImage(image_encoder, image).to(device=device)
            caption_features = getEncodedText(text_encoder, caption).to(device=device)

            loss = criterion(image_features, caption_features)

            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(image_encoder.parameters(), max_norm=0.1)

            optimizer.step()

        print(f'Epoch:{epoch + 1}, Loss:{loss.item():.4f}, Epoch time:{time.time()-epoch_start_time}')

    inferImageText(image_encoder, model.decoder, UCM_test_loader)

    print(f"Script execution time: {time.time()-script_start_time}")
