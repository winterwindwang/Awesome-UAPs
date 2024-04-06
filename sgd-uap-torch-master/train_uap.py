import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import os, copy
import sys
import torch

sys.path.append(os.path.realpath('..'))

from attacks import uap_sgd
from utils import loader_imgnet, model_imgnet, evaluate
import torchvision.models as models
import random


@torch.no_grad()
def evaluate_pert(model, val_loader, uap, log=None):
    correct = 0
    ori_correct = 0
    fool_num = 0
    n = 0
    for i, (images, labels) in enumerate(val_loader):
        images = images.to(device)
        labels = labels.to(device)
        ori_output = model(images)
        ori_pred = torch.argmax(ori_output, dim=1)

        pred_ori_idx = labels == ori_pred
        ori_correct += pred_ori_idx.sum()

        adv_img = torch.clamp((images + uap.repeat([images.size(0), 1, 1, 1]).to(device)), 0, 1)
        output = model(adv_img)
        pred = torch.argmax(output, dim=1)

        pred_pert_idx = labels == pred

        correct += (pred_pert_idx ^ pred_ori_idx).sum()

        fool_num += (ori_pred != pred).sum()

        n += images.size(0)
    clean_acc, perturbed_acc, fooling_ratio = np.round(100 * (ori_correct.item() / n), 2), np.round(100 * (correct.item() / n), 2), np.round(
        100 * (fool_num.item() / n), 2)
    print("Total:{}, success pred: {}, success attack: {}, fool number: {}".format(n, ori_correct, correct, fool_num))
    if log:
        print_log('\n\t#######################', log)
        print_log('\tClean model accuracy: {:.3f}'.format(clean_acc), log)
        print_log('\tPerturbed model accuracy: {:.3f}'.format(perturbed_acc), log)
        print_log('\tFooling Ratio: {:.3f}'.format(fooling_ratio), log)

    return clean_acc, perturbed_acc, fooling_ratio


def parse_arguments():
    parser = argparse.ArgumentParser(description='Trains a UAP')
    # pretrained
    parser.add_argument('--dataset', default='imagenet', choices=['cifar10', 'cifar100', 'imagenet', 'coco', 'voc', 'places365'],
                        help='Used dataset to generate UAP (default: imagenet)')
    parser.add_argument('--pretrained_dataset', default='imagenet', choices=['cifar10', 'cifar100', 'imagenet'],
                        help='Used dataset to train the initial model (default: imagenet)')
    parser.add_argument('--pretrained_arch', default='vit_tiny', choices=['vgg16_cifar', 'vgg19_cifar', 'resnet20', 'resnet56',
                                                'alexnet', 'googlenet', 'vgg16', 'vgg19',
                                                'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                                                'inception_v3','densenet121','densenet161',"resnext50_32x4d",
                                                "efficientnet_b0", "wide_resnet50_2", "mnasnet1_0","mlp","vit_tiny"],
                        help='Used model architecture: (default: vgg16)')
    parser.add_argument('--pretrained_seed', type=int, default=123,
                        help='Seed used in the generation process (default: 123)')
    parser.add_argument('--nb_epoch', type=int, default=50,
                        help='')
                        
    # Parameters regarding UAP
    parser.add_argument('--epsilon', type=float, default=10/255,
                        help='Norm restriction of UAP (default: 10/255)')
    parser.add_argument('--beta', type=float, default=12,
                        help='')
    parser.add_argument('--step_decay', type=float, default=0.7,
                        help='')
    parser.add_argument('--num_iterations', type=int, default=2000,
                        help='Number of iterations (default: 2000)')
    parser.add_argument('--postfix', default='Revised_sample_number',
                        help='Postfix to attach to result folder')
    # Optimization options

    parser.add_argument('--batch_size', type=int, default=25,
                        help='Batch size (default: 32)')
    parser.add_argument('--learning_rate', type=float, default=0.005,
                        help='Learning Rate (default: 0.001)')
    parser.add_argument('--print_freq', default=200, type=int, metavar='N',
                        help='print frequency (default: 200)')
    parser.add_argument('--ngpu', type=int, default=1,
                        help='Number of used GPUs (0 = CPU) (default: 1)')
    parser.add_argument('--workers', type=int, default=0,
                        help='Number of data loading workers (default: 6)')
    args = parser.parse_args()

    args.use_cuda = args.ngpu>0 and torch.cuda.is_available()

    if args.pretrained_seed is None:
        args.pretrained_seed = random.randint(1, 10000)
    return args


def get_result_path(dataset_name, network_arch, postfix=''):
    if not os.path.isdir(RESULT_PATH):
        os.makedirs(RESULT_PATH)
    ISOTIMEFORMAT='%Y-%m-%d-%H-%M-%S'
    t_string = '{}'.format(time.strftime( ISOTIMEFORMAT, time.gmtime(time.time()) ))
    result_path = os.path.join(RESULT_PATH, "{}_{}_{}_{}".format(t_string, dataset_name, network_arch, postfix))
    os.makedirs(result_path)
    return result_path


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()    


def save_checkpoint(state, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)


def main():
    args = parse_arguments()
    # get the result path to store the results
    result_path = get_result_path(dataset_name=args.dataset,
                                network_arch=args.pretrained_arch,
                                postfix=args.postfix)
    
    # Init logger
    log_file_name = os.path.join(result_path, 'log.txt')
    print("Log file: {}".format(log_file_name))
    log = open(log_file_name, 'w')
    print_log('save path : {}'.format(result_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    for key, value in state.items():
        print_log("{} : {}".format(key, value), log)
    print_log("Random Seed: {}".format(args.pretrained_seed), log)
    print_log("Python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("Torch  version : {}".format(torch.__version__), log)
    print_log("Cudnn  version : {}".format(torch.backends.cudnn.version()), log)   

    nb_images = 2000
    loader = loader_imgnet(dir_data, nb_images=nb_images, batch_size=args.batch_size) # adjust batch size as appropriate
    

    # load model
    model = model_imgnet(args.pretrained_arch)
    model.eval()
    print_log("=> Creating model '{}'".format(args.pretrained_arch), log)

    uap = uap_sgd(model, loader, args.nb_epoch, args.epsilon, args.beta, args.step_decay)


    test_loader = loader_imgnet(dir_data, batch_size=args.batch_size, train=False)  # adjust batch size as appropriate
    acc, asr, fooling_rate = evaluate_pert(model, test_loader, uap, log=log)
    print(f"valdtion: acc: {acc}, fooling rate:{fooling_rate}, asr: {asr}")
    save_checkpoint({
      'arch'        : args.pretrained_arch,
      'uap'  : uap,
      'args'        : copy.deepcopy(args),
      'fooling rate'        : fooling_rate,
    }, result_path, f'checkpoint_{fooling_rate*100}.pth.tar')
    torch.save(result_path, f'{args.pretrained_arch}_checkpoint_{fooling_rate*100}_{iter}.pth')
    log.close()


if __name__ == "__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cuda'
    sys_str = "linux"
    if "linux" in sys_str:
        dir_data = 'xxx/dataset/ImageNet/'
        RESULT_PATH = "xxxx/sgd-uap-torch-master/results/"
    elif "win" in sys_str:
        dir_data = 'xxx/ImageNet/'
        RESULT_PATH = "xxx/sgd-uap-torch-master/results/"
    main()

