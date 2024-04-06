import os
import shutil
import torch
import torchvision.models
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

def default_fn(file):
    img = Image.open(file).convert("RGB")
    return img

class ImageNet(Dataset):
    def __init__(self, data_folder, label_path='', transform=None, default_fn=default_fn):
        data = []
        classes = {}
        if label_path:
            with open(label_path, 'r') as fr:
                lines = fr.readlines()
                # lines = fr.read()
                labels = []
                for line in lines:
                    img_name, label = line.split()
                    data.append((os.path.join(data_folder, img_name), int(label)))
                    labels.append(int(label))
                # num_classes=len(set(labels))
                for i in range(len(set(labels))):
                    classes[i] = i
        else:
            dirlists = os.listdir(data_folder)
            try:
                dirlists.sort(key=lambda d: int(d))
            except:
                dirlists = sorted(dirlists)
            for i, dirfile in enumerate(dirlists):
                subdirname = os.path.join(data_folder, dirfile)
                classes[dirfile] = i
                for file in os.listdir(subdirname):
                    filepath = os.path.join(subdirname, file)
                    data.append((filepath, classes[dirfile]))
        self.data = data
        self.classes = classes
        self.num_classes = len(classes.values())
        self.transform = transform
        self.default_fn = default_fn

    def __len__(self):
        return len(self.data)

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        file, label = self.data[index]

        img = self.default_fn(file)

        if self.transform is not None:
            img = self.transform(img)

        return img, label