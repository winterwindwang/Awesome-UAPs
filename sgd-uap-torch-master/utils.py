'''
Functions for:
- Loading models, datasets
- Evaluating on datasets with or without UAP
'''

import multiprocessing
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torchvision

from models_cifar import *
from torch.utils import model_zoo
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset
from PIL import Image
import timm

CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]

IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]


class Normalizer(nn.Module):
    def __init__(self, mean, std):
        super(Normalizer, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        
    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)
    
    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)
    

def normalize_fn(tensor, mean, std):
    """
    Differentiable version of torchvision.functional.normalize
    - default assumes color channel is at dim = 1
    """
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)


'''
Load pre-trained ImageNet models

For models pre-trained on Stylized-ImageNet:
[ICLR 2019] ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness
Paper: https://openreview.net/forum?id=Bygh9j09KX
Code: https://github.com/rgeirhos/texture-vs-shape
'''    
def model_imgnet(model_name):
    '''
    model_name options:
    resnet50_SIN       trained on Stylized only
    resnet50_SIN-IN    trained on ImageNet + Stylized
    resnet50_SIN-2IN   trained on ImageNet + Stylized, then fine-tuned on ImageNet
    
    or load torchvision.models pre-trained on ImageNet: https://pytorch.org/docs/stable/torchvision/models.html
    '''
    
    if model_name[:12] == 'resnet50_SIN':
        model_urls = {
            'resnet50_SIN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar',
            'resnet50_SIN-IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar',
            'resnet50_SIN-2IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar',
        }
        model = torchvision.models.resnet50(pretrained=False)
        model = nn.DataParallel(model).cuda()
        checkpoint = model_zoo.load_url(model_urls[model_name])
        model.load_state_dict(checkpoint['state_dict'])
        
    # Load pre-trained ImageNet models from torchvision
    else:
        try:
            model = eval("torchvision.models.{}(pretrained=True)".format(model_name))
            model = nn.DataParallel(model).cuda()
        except:
            if "mlp" in model_name:
                model = timm.create_model('mixer_b16_224', pretrained=True)
            elif "vit" in model_name:
                model = timm.create_model('vit_base_patch8_224', pretrained=True)
            elif "vit_tiny" in model_name:
                model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
            elif "vit_small" in model_name:
                model = timm.create_model('vit_small_patch32_224', pretrained=True)
            else:
                pass
            model = nn.DataParallel(model).cuda()

    

    # Normalization wrapper, so that we don't have to normalize adversarial perturbations
    normalize = Normalizer(mean = IMGNET_MEAN, std = IMGNET_STD)
    model = nn.Sequential(normalize, model)
    model = model.cuda()
    print("Model loading complete.")
    
    return model


# Load pre-trained CIFAR-10 models
def model_cifar(model_name, ckpt_path):
    '''
    CIFAR-10 model implementations from:
    https://github.com/kuangliu/pytorch-cifar
    '''
    if model_name == 'resnet18':
        model = ResNet18()
    elif model_name == 'vgg16':
        model = VGG('VGG16')
        
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    
    # Load saved weights and stats
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    
    # Normalization wrapper, so that we don't have to normalize adversarial perturbations
    normalize = Normalizer(mean = CIFAR_MEAN, std = CIFAR_STD)
    model = nn.Sequential(normalize, model)
    model = model.cuda()

    return model, best_acc


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



# dataloader for ImageNet
def loader_imgnet(dir_data, nb_images = 50000, batch_size = 100, img_size = 224, train=True):
    traindir = os.path.join(dir_data, 'ImageNet10k')
    valdir = os.path.join(dir_data, 'val')
    if train:
        train_transform = transforms.Compose([
            transforms.Resize(256),
            # transforms.Resize(299), # inception_v3
            transforms.RandomCrop(img_size),
            transforms.ToTensor(),
        ])
        train_data = ImageFolder(root=traindir, transform=train_transform)
        np.random.seed(1024)
        sample_indices = np.random.permutation(range(train_data.__len__()))[:nb_images]
        train_data = Subset(train_data, sample_indices)
        dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                            # num_workers=4,
                                            pin_memory=True)
    else:
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ])
        val_dataset = ImageFolder(root=valdir, transform=val_transform)
        dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            # num_workers=4
        )
    return dataloader



# dataloader for CIFAR-10
def loader_cifar(dir_data, train = False, batch_size = 250):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if train:
        trainset = torchvision.datasets.CIFAR10(root = dir_data, train = True, download = True, transform = transform_test)
        dataloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = True, num_workers = max(1, multiprocessing.cpu_count() - 1))
    else:
        testset = torchvision.datasets.CIFAR10(root = dir_data, train = False, download = True, transform = transform_test)
        dataloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = True, num_workers = max(1, multiprocessing.cpu_count() - 1))
    return dataloader


# Evaluate model on data with or without UAP
# Assumes data range is bounded by [0, 1]
def evaluate(model, loader, uap = None, n = 5):
    '''
    OUTPUT
    top         top n predicted labels (default n = 5)
    top_probs   top n probabilities (default n = 5)
    top1acc     array of true/false if true label is in top 1 prediction
    top5acc     array of true/false if true label is in top 5 prediction
    outputs     output labels
    labels      true labels
    '''
    probs, labels = [], []
    model.eval()
    
    if uap is not None:
        # _, (x_val, y_val) = next(enumerate(loader))
        batch_size = 50 # len(x_val)
        uap = uap.unsqueeze(0).repeat([batch_size, 1, 1, 1])
    
    with torch.set_grad_enabled(False):
        for i, (x_val, y_val) in enumerate(loader):
            
            if uap is None:
                out = torch.nn.functional.softmax(model(x_val.cuda()), dim = 1)
            else:
                perturbed = torch.clamp((x_val + uap).cuda(), 0, 1) # clamp to [0, 1]
                out = torch.nn.functional.softmax(model(perturbed), dim = 1)
                
            probs.append(out.cpu().numpy())
            labels.append(y_val)
            
    # Convert batches to single numpy arrays    
    probs = np.stack([p for l in probs for p in l])
    labels = np.array([t for l in labels for t in l])
    
    # Extract top 5 predictions for each example
    top = np.argpartition(-probs, n, axis = 1)[:,:n]
    top_probs = probs[np.arange(probs.shape[0])[:, None], top].astype(np.float16)
    top1acc = top[range(len(top)), np.argmax(top_probs, axis = 1)] == labels
    top5acc = [labels[i] in row for i, row in enumerate(top)]
    outputs = top[range(len(top)), np.argmax(top_probs, axis = 1)]
        
    return top, top_probs, top1acc, top5acc, outputs, labels
