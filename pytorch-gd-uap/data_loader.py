import json
import os
import shutil
from typing import List, Any
import torchvision.models
from torch.utils.data import Dataset, Subset
from PIL import Image
import torchvision.datasets as dset
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
import torch
from glob import glob
import random


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
        else:  # Mixure dataset
            data_list = []
            for pth in data_folder:
                if "COCO" in pth or 'VOC' in pth:
                    file_list = glob(os.path.join(pth, "*.jpg"))
                    random.shuffle(file_list)
                    file_list = file_list[:5000]
                    data_list.extend(file_list)
                else:
                    current_list = []
                    for subfolder in os.listdir(pth):
                        if "SUN397" in pth:
                            file_list = glob(os.path.join(pth, subfolder, "*.jpg"))
                            current_list.extend(file_list)
                        else:
                            file_list = glob(os.path.join(pth, subfolder, "*.JPEG"))
                            current_list.extend(file_list)
                    random.shuffle(current_list)
                    current_list = current_list[:5000]
                    data_list.extend(current_list)
            data = data_list
        self.data = data
        self.label_path = label_path
        self.classes = classes
        self.num_classes = len(classes.values())
        self.transform = transform
        self.default_fn = default_fn

    def __len__(self):
        return len(self.data)

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        if self.label_path:
            file, label = self.data[index]
        else:
            file = self.data[index]
            label = 1
        img = self.default_fn(file)

        if self.transform is not None:
            img = self.transform(img)

        return img, label


class COCOImage(Dataset):
    def __init__(self, data_folder, transform=None, default_fn=default_fn):
        data = []
        classes = {}

        dirlists = os.listdir(data_folder)
        for file in dirlists:
            filepath = os.path.join(data_folder, file)
            data.append((filepath, 1))
        self.data = data
        self.classes = 1000
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

class SubTaskDataset(Dataset):
    def __init__(self, data_folder, label_path, transform=None, default_fn=default_fn):
        data = []
        with open(label_path, 'r') as fr:
            lines = fr.readlines()
            # lines = fr.read()
            for line in lines:
                img_name, label = line.split()
                data.append((os.path.join(data_folder, img_name), int(label)))
        self.data = data
        self.transform = transform
        self.default_fn = default_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file, label = self.data[index]

        img = self.default_fn(file)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

class ImageNetSplit(Dataset):
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
            for i, dirfile in enumerate(dirlists):
                subdirname = os.path.join(data_folder, dirfile)
                classes[dirfile] = i
                for file in os.listdir(subdirname):
                    filepath = os.path.join(subdirname, file)
                    data.append((filepath, i))
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
        return img, label, file


class COCODataset(Dataset):
    def __init__(self, data_folder, transform=None, default_fn=default_fn):
        data = []
        for file in os.listdir(data_folder):
            data.append((file, 1))
        self.data = data
        self.transform = transform
        self.default_fn = default_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file, label = self.data[index]
        img = self.default_fn(file)
        if self.transform is not None:
            img = self.transform(img)
        return img, label, file


class COCOVOCDataset(Dataset):
    def __init__(self, data_folder, transform=None, default_fn=default_fn):
        data = []
        for file in os.listdir(data_folder):
            file_path = os.path.join(data_folder, file)
            data.append((file_path, 1))
        self.data = data
        self.transform = transform
        self.default_fn = default_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        file, label = self.data[index]
        img = self.default_fn(file)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def get_dataset(data_name, system='linux', nb_images=2000, input_size=224, only_train=True):
    if data_name == 'imagenet':
        if "linux" in system:
            path = 'xxxx/dataset/ImageNet/'
        elif "win" in system:
            path = 'xxxx/ImageNet/'
        train_transform = transforms.Compose([
            transforms.Resize(256),
            # transforms.Resize(299), # inception_v3
            transforms.RandomCrop(input_size),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            # transforms.Resize(299), # inception_v3
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
        ])
        traindir = os.path.join(path, 'ImageNet10k')
        valdir = os.path.join(path, 'val')
        # dataset
        if only_train:
            train_dataset = ImageFolder(root=traindir, transform=train_transform)
            if nb_images < 10000:
                np.random.seed(1000) 
                sample_indices = np.random.permutation(range(train_dataset.__len__()))[:nb_images]
                train_dataset = Subset(train_dataset, sample_indices)
        else:
            train_dataset = ImageFolder(root=valdir, transform=test_transform)
    elif data_name == 'coco':
        if "linux" in system:
            path = 'xxxx/dataset/COCO/train2014/'
        elif "win" in system:
            path = "xxxx/COCO2014/train2014/"
        train_transform = transforms.Compose([
            transforms.Resize(int(input_size * 1.143)),
            transforms.RandomCrop(input_size),
            transforms.ToTensor(),
        ])
        train_dataset = COCOVOCDataset(path, train_transform)
        if nb_images < 50000:
            np.random.seed(1024)
            sample_indices = np.random.permutation(range(train_dataset.__len__()))[:nb_images]
            train_dataset = Subset(train_dataset, sample_indices)
    elif data_name == 'voc':
        if "linux" in system:
            path = 'xxxx/dataset/VOC200712/VOCdevkit/VOC2012/JPEGImages/'
        elif "win" in system:
            path = "xxxx/VOCPASCAL/VOCtrainval/VOCdevkit/VOC2007/JPEGImages/"
        train_transform = transforms.Compose([
            transforms.Resize(int(input_size * 1.143)),
            transforms.RandomCrop(input_size),
            transforms.ToTensor(),
        ])
        train_dataset = COCOVOCDataset(path, train_transform)
        if nb_images < 50000:
            np.random.seed(1024)
            sample_indices = np.random.permutation(range(train_dataset.__len__()))[:nb_images]
            train_dataset = Subset(train_dataset, sample_indices)
    elif data_name == 'sun397':
        if "linux" in system:
            path = 'xxxx/dataset/transfer/SUN397/'
        elif "win" in system:
            path = 'xxxx/ImageNet/'
        train_transform = transforms.Compose([
            transforms.Resize(256),
            # transforms.Resize(299), # inception_v3
            transforms.RandomCrop(input_size),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            # transforms.Resize(299), # inception_v3
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
        ])
        traindir = os.path.join(path, 'train')
        valdir = os.path.join(path, 'test')
        # dataset
        if only_train:
            train_dataset = ImageFolder(root=traindir, transform=train_transform)
            if nb_images < 50000:
                np.random.seed(1024)
                sample_indices = np.random.permutation(range(train_dataset.__len__()))[:nb_images]
                train_dataset = Subset(train_dataset, sample_indices)
        else:
            train_dataset = ImageFolder(root=valdir, transform=test_transform)
    elif data_name == 'mixed':
        if "linux" in system:
            path = [
                    'xxxx/dataset/transfer/SUN397/',
                    'xxxx/dataset/COCO/train2014/',
                    'xxxx/dataset/ImageNet/ImageNet10k/',
                    'xxxx/dataset/VOC200712/VOCdevkit/VOC2012/JPEGImages/'
                    ]
        elif "win" in system:
            path = [r'xxxx\ImageNet\train',
                    r'xxxx\COCO2014\train2014',
                    r'xxxx\VOCPASCAL\VOCtrainval\VOCdevkit\VOC2007\JPEGImages',
                    r'xxxx\AttackTransfer\SUN397\train'
                    ]
        train_transform = transforms.Compose([
            transforms.Resize(256),
            # transforms.Resize(299), # inception_v3
            transforms.RandomCrop(input_size),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Resize(256),
            # transforms.Resize(299), # inception_v3
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
        ])

        # dataset
        if only_train:
            train_dataset = ImageNet(data_folder=path, transform=train_transform)
            if nb_images < 50000:
                np.random.seed(1024)
                sample_indices = np.random.permutation(range(train_dataset.__len__()))[:nb_images]
                train_dataset = Subset(train_dataset, sample_indices)
        else:
            pass
    return train_dataset
