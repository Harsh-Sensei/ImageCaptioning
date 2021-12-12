import torch
import torch.nn as nn  # Neural networks like fully connected layers, CNNs, RNNs, LSTMs, GRUs
import torch.optim as optim  # Optimiations like SGD, Adam
import torch.nn.functional as F  # For activation functions like RELU, Tanh, sigmoid
import torch.utils.data as Dataloader  # For dataset management
import torchvision.datasets as datasets  # Standard datasets, has COCO
import torchvision.transforms as transforms  # transformations on dataset
import spacy

# Device setup for runtime
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

# Loading word embeddings
nlp = spacy.load("en_core_web_lg")

# hyperparameters
input_image_size = (224, 224, 3)
num_layers = 2
hidden_layer_size = 256
learning_rate = 0.001
batch_size = 64
num_epochs = 5

transform_train = transforms.Compose([
    transforms.Resize(256),  # smaller edge of image resized to 256
    transforms.RandomCrop(224),  # get 224x224 crop from random location
    transforms.RandomHorizontalFlip(),  # horizontally flip image with probability=0.5
    transforms.ToTensor(),  # convert the PIL Image to a tensor
    transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                         (0.229, 0.224, 0.225))])


# Input image dimensions: 224X224X3
class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()

        # Encoded features of image
        self.encoded_features = None

        # kernels and padding for VGG-16 architecture
        self.kernel = (3, 3)
        self.padding = (1, 1)

        # VGG16 layers
        self.conv1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=self.kernel, padding=self.padding)
        self.conv1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=self.kernel, padding=self.padding)

        self.conv2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=self.kernel, padding=self.padding)
        self.conv2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=self.kernel, padding=self.padding)

        self.conv3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=self.kernel, padding=self.padding)
        self.conv3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=self.kernel, padding=self.padding)
        self.conv3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=self.kernel, padding=self.padding)

        self.conv4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=self.kernel, padding=self.padding)
        self.conv4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=self.kernel, padding=self.padding)
        self.conv4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=self.kernel, padding=self.padding)

        self.conv5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=self.kernel, padding=self.padding)
        self.conv5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=self.kernel, padding=self.padding)
        self.conv5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=self.kernel, padding=self.padding)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(25088, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 300)

        # Setting up decoder LSTM network

    def forward(self, x):
        # Forward pass through VGG16 architecture
        x = F.relu(self.conv1_1(x))
        x = F.relu(self.conv1_2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.maxpool(x)
        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.maxpool(x)
        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.5)
        x = self.fc3(x)

        # Storing the encoded features
        self.encoded_features = x

        return x


class DecoderLSTM(nn.Module):

    def __init__(self, hidden_layer_size=256, num_hidden_layers=2, input_dimension=300, output_dimensions=300):
        super(DecoderLSTM, self).__init__()

        self.hidden_layer_size = hidden_layer_size
        self.num_hidden_layers = num_hidden_layers
        self.num_embedding_features = 300

        self.LSTM = nn.LSTM(input_dimension, hidden_layer_size, num_hidden_layers, batch_first=True)

    def init_hidden_state(self, sequence_length):
        h_ret = torch.zeros(self.num_hidden_layers, sequence_length, self.hidden_layer_size).to(device)
        c_ret = torch.zeros(self.num_hidden_layers, sequence_length, self.hidden_layer_size).to(device)

        return h_ret, c_ret

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.
        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)

        # Flatten image
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # # Embedding
        # embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores
        predictions = torch.zeros(batch_size, max(decode_lengths), self.num_embedding_features).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])

            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds

        return predictions, encoded_captions, decode_lengths, sort_ind


model = VGG16()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def convert_to_embeddings(data):
    return nlp(data)


for epoch in range(num_epochs):
    for batch_idx, (data, targets) in enumerate(train_loader):
        embedded_targets = convert_to_embeddings(targets)
