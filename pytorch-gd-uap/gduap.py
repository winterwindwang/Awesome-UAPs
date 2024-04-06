import os

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import json
from torch.utils.data import DataLoader, Subset
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import timm

# TORCH_HUB_DIR = '/vulcanscratch/psando/TorchHub'
#
# IMAGENET_VAL_DIR = '/vulcanscratch/psando'
# VOC_VAL_DIR = '/vulcanscratch/psando/VOC'
# CIFAR_VAL_DIR = '/fs/vulcan-datasets/CIFAR'

debug = False

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

def get_model(model_name, device):
    # torch.hub.set_dir(TORCH_HUB_DIR)
    if "resnet50" in model_name:
        model = models.resnet50(pretrained=True)
        eps_step = 0.001
    elif "resnet101" in model_name:
        model = models.resnet101(pretrained=True)
        eps_step = 0.001
    elif "resnet152" in model_name:
        model = models.resnet152(pretrained=True)
        eps_step = 0.001
    elif "resnext50" in model_name:
        model = models.resnext50_32x4d(pretrained=True)
        eps_step = 0.005
    elif "wideresnet" in model_name:
        model = models.wide_resnet50_2(pretrained=True)
        eps_step = 0.01
    elif "vgg16" in model_name:
        model = models.vgg16(pretrained=True)
        eps_step = 0.003
    elif "vgg19" in model_name:
        model = models.vgg19(pretrained=True)
        eps_step = 0.003
    elif "densenet121" in model_name:
        model = models.densenet121(pretrained=True)
        eps_step = 0.01
    elif "densenet161" in model_name:
        model = models.densenet161(pretrained=True)
        eps_step = 0.003
    elif "inception_v3" in model_name:
        model = models.inception_v3(pretrained=True)
        eps_step = 0.003
    elif "googlenet" in model_name:
        model = models.googlenet(pretrained=True)
        eps_step = 0.005
    elif "alexnet" in model_name:
        model = models.alexnet(pretrained=True)
        eps_step = 0.001
    elif "mnasnet10" in model_name:
        model = models.mnasnet1_0(pretrained=True)
        eps_step = 0.001
    elif "efficientnetb0" in model_name:
        model = models.efficientnet_b0(pretrained=True)
        eps_step = 0.001
    elif "mlp" in model_name:
        model = timm.create_model('mixer_b16_224', pretrained=True)
    elif "vit" in model_name:
        model = timm.create_model('vit_base_patch8_224', pretrained=True)
    elif "vit_tiny" in model_name:
        model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    elif "vit_small" in model_name:
        model = timm.create_model('vit_small_patch32_224', pretrained=True)
    else:
        pass

    # Normalization wrapper, so that we don't have to normalize adversarial perturbations
    normalize = Normalizer(mean=IMGNET_MEAN, std=IMGNET_STD)
    model = nn.Sequential(normalize, model)
    model = model.to(device)
    print("Model loading complete.")
    return model


def normalize(x):
    """
    Normalizes a batch of images with size (batch_size, 3, height, width)
    by mean and std dev expected by PyTorch models
    """
    mean = torch.Tensor([0.485, 0.456, 0.406])
    std = torch.Tensor([0.229, 0.224, 0.225])
    return (x - mean.type_as(x)[None,:,None,None]) / std.type_as(x)[None,:,None,None]

def get_conv_layers(model):
    return [module for module in model.modules() if type(module) == nn.Conv2d]

def l2_layer_loss(model, delta):
    loss = torch.tensor(0.)
    activations = []
    remove_handles = []

    def activation_recorder_hook(self, input, output):
        activations.append(output)
        return None

    for conv_layer in get_conv_layers(model):
        handle = conv_layer.register_forward_hook(activation_recorder_hook)
        remove_handles.append(handle)

    model.eval()
    model.zero_grad()
    model(delta)

    # unregister hook so activation tensors have no references
    for handle in remove_handles:
        handle.remove()

    loss = -sum(list(map(lambda activation: torch.log(torch.sum(torch.square(activation)) / 2), activations)))
    return loss

def get_index_to_label_map(dataset_name):
    if dataset_name == 'imagenet':
        with open('imagenet_class_index.json', 'r') as read_file:
            class_idx = json.load(read_file)
            index_to_label = [class_idx[str(k)][1] for k in range(len(class_idx))]
            return index_to_label

def get_data_loader(dir_data, nb_images=2000, batch_size=64, img_size = 224, train=True):
    """
    Returns a DataLoader with validation images for dataset_name
    """
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
        if nb_images < 50000:
            np.random.seed(0)
            sample_indices = np.random.permutation(train_data.__len__())[:nb_images]
            train_data = Subset(train_data, sample_indices)

        dataloader = torch.utils.data.DataLoader(train_data,
                                                 batch_size=batch_size,
                                                 shuffle=True,
                                                 # num_workers=4,
                                                 pin_memory=True)
    else:
        val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ])
        val_dataset = ImageFolder(root=valdir, transform=val_transform)
        if nb_images < 50000:
            np.random.seed(0)
            sample_indices = np.random.permutation(val_dataset.__len__())[:nb_images]
            val_dataset = Subset(val_dataset, sample_indices)
        dataloader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=True,
            # num_workers=4
        )
    return dataloader


@torch.no_grad()
def get_fooling_rate(model, delta, data_loader, device, disable_tqdm=False):
    correct = 0
    ori_correct = 0
    fool_num = 0
    n = 0
    # phar = tqdm(val_loader, desc="train loader")
    for i, (images, labels) in enumerate(data_loader):
        images = images.to(device)
        labels = labels.to(device)
        ori_output = model(images)
        ori_pred = torch.argmax(ori_output, dim=1)

        pred_ori_idx = labels == ori_pred
        ori_correct += pred_ori_idx.sum().item()

        adv_img = torch.clamp((images + delta.repeat([images.size(0), 1, 1, 1]).to(device)), 0, 1)
        output = model(adv_img)
        pred = torch.argmax(output, dim=1)

        pred_pert_idx = labels == pred

        correct += (pred_pert_idx ^ pred_ori_idx).sum().item()

        fool_num += (ori_pred != pred).sum().item()

        n += images.size(0)
    print("Total:{}, success pred: {}, success attack: {}, fool number: {}".format(n, ori_correct, correct, fool_num))
    return np.round(100 * (ori_correct / n), 2), np.round(100 * (correct / n), 2), np.round(
        100 * (fool_num / n), 2)

def get_baseline_fooling_rate(model, device, disable_tqdm=False):
    """
    Baseline fooling rate is always evaluated on ILSVRC 2012 dataset
    """
    xi_min = -10/255
    xi_max = 10/255
    delta = (xi_min - xi_max) * torch.rand((1, 3, 224, 224), device=device) + xi_max
    delta.requires_grad = True
    data_loader = get_data_loader('imagenet')
    fr = get_fooling_rate(model, delta, data_loader, device, disable_tqdm=disable_tqdm)
    return fr

def get_rate_of_saturation(delta, xi):
    """
    Returns the proportion of pixels in delta
    that have reached the max-norm limit xi
    """
    return np.sum(np.equal(np.abs(delta), xi)) / np.size(delta)

def gd_universal_adversarial_perturbation(model, model_name, train_type, batch_size, device, data_dir, patience_interval, id, disable_tqdm=False,dataset_name='imagenet'):
    """
    Returns a universal adversarial perturbation tensor
    """

    max_iter = 10000
    size = 224

    sat_threshold = 0.00001
    sat_prev = 0
    sat = 0
    sat_change = 0
    sat_min = 0.5
    sat_should_rescale = False

    iter_since_last_fooling = 0
    iter_since_last_best = 0
    best_fooling_rate = 0

    xi_min = -10/255
    xi_max = 10/255
    delta = (xi_min - xi_max) * torch.rand((1, 3, size, size), device=device) + xi_max
    delta.requires_grad = True

    print(f"Initial norm: {torch.norm(delta, p=np.inf)}")

    optimizer = optim.Adam([delta], lr=0.1)
    nb_images = 64
    train_data_loader = get_data_loader(data_dir, nb_images=nb_images,  batch_size=batch_size)
    data_iter = iter(train_data_loader)

    for i in tqdm(range(max_iter), disable=disable_tqdm):
        try:
            images, labels = next(data_iter)
        except:
            data_iter = iter(train_data_loader)
            images, labels = next(data_iter)
        images = images.to(device)

        images = torch.clamp((images + delta.repeat([images.shape[0], 1, 1, 1])), 0, 1)

        iter_since_last_fooling += 1
        optimizer.zero_grad()
        loss = l2_layer_loss(model, images)
        loss.backward()
        optimizer.step()

        if i % 1000 == 0:
            print(f"Iter {i}, Loss: {loss.item()}")
            if debug:
                print(f"Norm before clip: {torch.norm(delta, p=np.inf)}")

        # clip delta after each step
        with torch.no_grad():
            delta.clamp_(xi_min, xi_max)

        # compute rate of saturation on a clamped delta
        sat_prev = np.copy(sat)
        sat = get_rate_of_saturation(delta.cpu().detach().numpy(), xi_max)
        sat_change = np.abs(sat - sat_prev)

        if sat_change < sat_threshold and sat > sat_min:
            if debug:
                print(f"Saturated delta in iter {i} with {sat} > {sat_min}\nChange in saturation: {sat_change} < {sat_threshold}\n")
            sat_should_rescale = True

        # fooling rate is measured every 200 iterations if saturation threshold is crossed
        # otherwise, fooling rate is measured every 400 iterations
        if iter_since_last_fooling > 400 or (sat_should_rescale and iter_since_last_fooling > 200):
            iter_since_last_fooling = 0
            print("Getting latest fooling rate...")
            acc, asr, current_fooling_rate = get_fooling_rate(model, delta, train_data_loader, device, disable_tqdm=disable_tqdm)
            print(f"Latest fooling rate: {current_fooling_rate}")

            if current_fooling_rate > best_fooling_rate:
                print(f"Best fooling rate thus far: {current_fooling_rate}")
                best_fooling_rate = current_fooling_rate
                filename = f"perturbations/{id}_{model_name}_{train_type}_iter={i}_val={dataset_name}_fr={int(best_fooling_rate * 1000)}_sample_number_{nb_images}.pth"
                # np.save(filename, delta.cpu().detach().numpy())
                torch.save(delta.data, filename)
            else:
                iter_since_last_best += 1

            # if the best fooling rate has not been overcome after patience_interval iterations
            # then training is considered complete
            if iter_since_last_best == patience_interval:
                break

        if sat_should_rescale:
            with torch.no_grad():
                delta.data = delta.data / 2
            sat_should_rescale = False

    print(f"Training complete.\nLast delta saved at: {filename}\nLast delta Iter: {i}, Loss: {loss}, Fooling rate: {best_fooling_rate}")
    return delta
