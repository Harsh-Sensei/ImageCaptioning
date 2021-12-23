# Sydney_captions = https://mega.nz/folder/pG4yTYYA#4c4buNFLibryZnlujsrwEQ
# UCM_captions = https://mega.nz/folder/wCpSzSoS#RXzIlrv--TDt3ENZdKN8JA

import PIL.Image as Image
import numpy
import os
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import json
import spacy

spacy_eng = spacy.load("en_core_web_lg")


class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UKN>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UKN>": 3}

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        freq = {}
        idx = 4

        for elem in sentence_list:
            for word in self.tokenize(elem):
                if word not in freq:
                    freq[word] = 1
                else:
                    freq[word] += 1

                if freq[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

        return

    def numericalize(self, text):
        tokenized_text = self.tokenize(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UKN>"]
            for token in tokenized_text
        ]


class UCM_Captions(Dataset):

    def __init__(self, transform=None, ret_type="image-image", freq_threshold=2):
        self.transform = transform
        self.ret_type = ret_type
        self.root_dir = "./dataset/UCM_Captions/UCM_captions/imgs"
        self.num_captions_per_img, self.meta_data = self.getMetaData()
        self.image_names = list(self.meta_data.keys())

        # Example of each elem in captions
        # [{'tokens': ['There', 'is', 'a', 'piece', 'of', 'farmland'], 'raw': 'There is a piece of farmland .',
        #  'imgid': 0, 'sentid': 0}, {'tokens':.........]

        self.captions = list(self.meta_data.values())
        self.captions_raw = self.getRawCaptions(self.captions)
        self.freq_threshold = freq_threshold
        self.vocab = Vocabulary(self.freq_threshold)
        self.vocab.build_vocabulary(self.captions_raw)

    def getMetaData(self):
        data_info = open(r"./dataset/UCM_Captions/UCM_captions/dataset.json")
        data_info = json.load(data_info)['images']
        # data_info is a list of dict and each dict contains filename, sentences, tokens, etc of each image

        captions_per_img = len(data_info[0]['sentids'])

        result = {}
        for elem in data_info:
            result[str(elem['filename'])] = elem['sentences']

        return captions_per_img, result

    def __len__(self):
        if self.ret_type == "image-image" or self.ret_type == "image-caption":
            return len(self.meta_data)
        else:
            return self.num_captions_per_img * len(self.meta_data)

    def __getitem__(self, index):

        if self.ret_type == "image-image":
            img_path = os.path.join(self.root_dir, self.image_names[index])
            image = Image.open(img_path)
            y_label = self.captions[index]
        elif self.ret_type == "image-caption":
            y_label = self.captions[index]
        else:
            y_label = self.captions[int(index) // int(self.num_captions_per_img)]
            y_label = y_label[int(index) % int(self.num_captions_per_img)]['raw']
            y_label = self.numericalize_caption(y_label)
            # print("y_label")
            # print(y_label)
            y_label = torch.tensor(y_label)



        if self.transform:
            image = self.transform(image)

        if self.ret_type == "image-image":
            return image, image
        elif self.ret_type == "image-caption":
            return image, y_label
        elif self.ret_type == "caption-caption":
            return y_label, y_label

    def numericalize_caption(self, caption):
        result = [self.vocab.stoi["<SOS>"]]
        result += self.vocab.numericalize(caption)
        result += [self.vocab.stoi["<EOS>"]]
        return result

    def getRawCaptions(self, captions):
        result = []
        for elem in captions:
            for cap in elem:
                result.append(cap['raw'])

        return result


class Sydney_Captions(Dataset):
    def __init__(self, transform=None, ret_type="image-image"):
        self.transform = transform
        self.ret_type = ret_type
        self.root_dir = "./dataset/Sydney_Captions/Sydney_captions/imgs"
        self.meta_data = self.getMetaData()
        self.image_names = list(self.meta_data.keys())
        self.captions = list(self.meta_data.values())

    def getMetaData(self):
        data_info = open(r"./dataset/Sydney_Captions/Sydney_captions/dataset.json")
        data_info = json.load(data_info)['images']

        result = {}
        for elem in data_info:
            result[str(elem['filename'])] = elem['sentences']
        return result

    def __len__(self):
        return len(self.meta_data)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.image_names[index])
        image = Image.open(img_path)
        y_label = self.captions[index]

        if self.transform:
            image = self.transform(image)

        if self.ret_type == "image-image":
            return (image, image)
        elif self.ret_type == "image-caption":
            return (image, y_label)
        elif self.ret_type == "caption-caption":
            return (y_label, y_label)


class AuxPadClass:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        items = [item[0] for item in batch]
        input = pad_sequence(items, batch_first=True, padding_value=self.pad_idx)
        return input, input


def getTextUCMDataLoader(batch_size=32):
    dataset = UCM_Captions(transform=None, ret_type="caption-caption")
    UCM_train_set, UCM_test_set = torch.utils.data.random_split(dataset,
                                                                [int(dataset.__len__() * 0.8), dataset.__len__() -
                                                                 int(dataset.__len__() * 0.8)])
    print(dataset.vocab.itos)
    pad_idx = dataset.vocab.stoi["<PAD>"]
    TrainLoader = DataLoader(UCM_train_set, batch_size=batch_size, collate_fn=AuxPadClass(pad_idx=pad_idx),
                             shuffle=True)
    TestLoader = DataLoader(UCM_test_set, batch_size=batch_size, collate_fn=AuxPadClass(pad_idx=pad_idx), shuffle=True)

    return TrainLoader, TestLoader, pad_idx, dataset.vocab.__len__()


if __name__ == "__main__":
    A = getTextUCMDataLoader()
