# Sydney_captions = https://mega.nz/folder/pG4yTYYA#4c4buNFLibryZnlujsrwEQ
# UCM_captions = https://mega.nz/folder/wCpSzSoS#RXzIlrv--TDt3ENZdKN8JA

import PIL.Image as Image
import numpy
import os
from torch.utils.data import Dataset
import json
import spacy

nlp = spacy.load()


class UCM_Captions(Dataset):

    def __init__(self, transform=None, ret_type="image-image"):
        self.transform = transform
        self.ret_type = ret_type
        self.root_dir = "./dataset/UCM_Captions/UCM_captions/imgs"
        self.meta_data = self.getMetaData()
        self.image_names = list(self.meta_data.keys())
        self.captions = list(self.meta_data.values())

    def getMetaData(self):
        data_info = open(r"./dataset/UCM_Captions/UCM_captions/dataset.json")
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


if __name__ == "__main__":
    sample = UCM_Captions()
    print(sample.captions)