import argparse

import torchvision.models as models
import torchvision.transforms as transforms
from loader import ImagetNet
from Normalize import Normalize
from torchvision.datasets import ImageFolder
import warnings
from universal_pert import universal_perturbation
warnings.filterwarnings("ignore")
import numpy as np
from torch.utils.data import DataLoader
import torch
import os
from torch.utils.data import Subset
import timm


@torch.no_grad()
def evaluate_pert(model, val_loader, uap):
    correct = 0
    ori_correct = 0
    fool_num = 0
    n = 0
    # phar = tqdm(val_loader, desc="train loader")
    for i, (images, labels) in enumerate(val_loader):
        images = images.to(device)
        labels = labels.to(device)
        ori_output = model(images)
        ori_pred = torch.argmax(ori_output, dim=1)

        pred_ori_idx = labels == ori_pred
        ori_correct += pred_ori_idx.sum().item()

        adv_img = torch.clamp((images + uap.repeat([images.size(0), 1, 1, 1]).to(device)), 0, 1)
        output = model(adv_img)
        pred = torch.argmax(output, dim=1)

        pred_pert_idx = labels == pred

        correct += (pred_pert_idx ^ pred_ori_idx).sum().item()

        fool_num += (ori_pred != pred).sum().item()

        n += images.size(0)
    print("total:{}, success pred: {}, success attack: {}".format(n, ori_correct, correct))
    return np.round(100 * (ori_correct / n), 2), np.round(100 * (correct / n), 2), np.round(
        100 * (fool_num / n), 2)


def get_args():
    parser = argparse.ArgumentParser(description='Parameters loader')
    parser.add_argument('--model_name', type=str, default='squeezenet', help='')
    parser.add_argument('--data_dir', type=str, default='xx/ImageNet', help='')
    parser.add_argument('--save_dir', type=str, default='checkpoints_uaps/', help='')
    parser.add_argument('--nb_images', type=int, default=10000, help='iteration sample number')
    parser.add_argument('--norm', type=str, default='inf', help='')
    parser.add_argument('--epsilon', type=float, default=10/255, help='epsilon, limit the perturbation to [-10, 10] respect to [0, 255]')
    parser.add_argument('--batch_size', type=int, default=25, help='')
    parser.add_argument('--num_works', type=int, default=8, help='')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cuda'
    args = get_args()
    if "resnet50" in args.model_name:
        model = models.resnet50(pretrained=True)
        args.eps_step = 0.001
    elif "resnet101" in args.model_name:
        model = models.resnet101(pretrained=True)
        args.eps_step = 0.0005
    elif "resnet152" in args.model_name:
        model = models.resnet152(pretrained=True)
        args.eps_step = 0.001
    elif "resnext50" in args.model_name:
        model = models.resnext50_32x4d(pretrained=True)
        args.eps_step = 0.005
    elif "wideresnet" in args.model_name:
        model = models.wide_resnet50_2(pretrained=True)
        args.eps_step = 0.01
    elif "vgg16" in args.model_name:
        model = models.vgg16(pretrained=True)
        args.eps_step = 0.003
    elif "vgg19" in args.model_name:
        model = models.vgg19(pretrained=True)
        args.eps_step = 0.003
    elif "densenet121" in args.model_name:
        model = models.densenet121(pretrained=True)
        args.eps_step = 0.005
    elif "densenet161" in args.model_name:
        model = models.densenet161(pretrained=True)
        args.eps_step = 0.003
    elif "inception_v3" in args.model_name:
        model = models.inception_v3(pretrained=True)
        args.eps_step = 0.003
    elif "googlenet" in args.model_name:
        model = models.googlenet(pretrained=True)
        args.eps_step = 0.005
    elif "alexnet" in args.model_name:
        model = models.alexnet(pretrained=True)
        args.eps_step = 0.001
    elif "mnasnet10" in args.model_name:
        model = models.mnasnet1_0(pretrained=True)
        args.eps_step = 0.001
    elif "efficientnetb0" in args.model_name:
        model = models.efficientnet_b0(pretrained=True)
        args.eps_step = 0.001
    elif "mlp" in args.model_name:
        model = timm.create_model('mixer_b16_224', pretrained=True)
        args.eps_step = 0.001
    elif "vit" in args.model_name:
        model = timm.create_model('vit_base_patch8_224', pretrained=True)
    elif "vit_tiny" in args.model_name:
        args.eps_step = 0.001
        model = timm.create_model('vit_tiny_patch16_224', pretrained=True)
    elif "vit_small" in args.model_name:
        args.eps_step = 0.001
        model = timm.create_model('vit_small_patch32_224', pretrained=True)
    elif "squeezenet" in args.model_name:
            model = models.squeezenet1_0(pretrained=True)
    else:
        pass


    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    model = torch.nn.Sequential(Normalize(mean, std), model)
    model.eval()
    model = model.to(device)
    input_size = 224

    transform = transforms.Compose([
        transforms.Resize((330, 330)),
        transforms.CenterCrop(299),
        transforms.ToTensor()])

    print('loader data')
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
    sys_str = "linux"
    if "linux" in sys_str:
        path = 'xxx/dataset/ImageNet/'
    elif "win" in sys_str:
        path = 'xxx/ImageNet/'

    traindir = os.path.join(path, 'ImageNet10k')
    valdir = os.path.join(path, 'val')
    # dataset
    train_data = ImageFolder(root=traindir, transform=train_transform)
    nb_images = 2000
    np.random.seed(1024)
    sample_indices = np.random.permutation(range(train_data.__len__()))[:nb_images]
    train_data = Subset(train_data, sample_indices)
    test_data = ImageFolder(root=valdir, transform=test_transform)

    validation_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size,
                                                    num_workers=args.num_works)
    print(args)
    print(f'Computation: {args.model_name}')
    uap = universal_perturbation(args, train_data, test_data, model, xi=args.epsilon)

    uap = torch.tensor(uap[0])
    acc, asr, fooling_rate = evaluate_pert(model, validation_loader, uap.to(device))
    # print(v)
    torch.save(torch.tensor(uap),
               f'{args.save_dir}/Revised_uap_{args.model_name}_eps10_{nb_images}_fr{fooling_rate}.pth')
    print(f"model: {args.model_name}")
    print(f"validtion: acc: {acc}, fooling rate:{fooling_rate}, asr: {asr}")