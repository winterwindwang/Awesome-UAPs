import argparse
import torch
from gduap import gd_universal_adversarial_perturbation, get_data_loader, get_fooling_rate, get_baseline_fooling_rate, get_model
from analyze import get_tf_uap_fooling_rate

def validate_arguments(args):
    models = ["vgg16", "vgg19", "resnet50", "resnet101",
     "resnet152", "resnext50", "wideresnet",  "efficientnetb0",  "densenet121",
      "densenet161", "alexnet", "googlenet", "mnasnet10","mlp", "vit_tiny"]

    if not (args.model in models):
        print ('Argument Error: invalid network')
        exit(-1)

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='vit_tiny',
                        help='The network eg. vgg16, resnet50')
    parser.add_argument('--prior_type', default='imagenet_data',
                        help='Which kind of prior to use')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='The batch size to use for training and testing')
    parser.add_argument('--patience_interval', type=int, default=5,
                        help='The number of iterations to wait to verify convergence')
    parser.add_argument('--val_dataset_name', default='xxxx/ImageNet/',
                        help='The dataset to be used as validation')
    parser.add_argument('--final_dataset_name', default='xxxx/ImageNet/',
                        help='The dataset to be used for final evaluation')
    parser.add_argument('--id',
                        help='An identification number (e.g. SLURM Job ID) that will prefix saved files')
    parser.add_argument('--baseline', action='store_true',
                        help='Obtain a fooling rate for a baseline random perturbation')
    parser.add_argument('--tf_uap', default=None,
                        help='Obtain a fooling rate for a input TensorFlow UAP')
    args = parser.parse_args()
    validate_arguments(args)
    args.val_dataset_name = path

    device = torch.device(f"cuda:{1}" if torch.cuda.is_available() else "cpu")
    model = get_model(args.model, device)

    if args.baseline:
        print("Obtaining baseline fooling rate...")
        baseline_fooling_rate = get_baseline_fooling_rate(model, device, disable_tqdm=True)
        print(f"Baseline fooling rate for {args.model}: {baseline_fooling_rate}")
        return

    if args.tf_uap:
        print(f"Obtaining fooling rate for TensorFlow UAP called {args.tf_uap} using {args.model}...")
        tf_fooling_rate = get_tf_uap_fooling_rate(args.tf_uap, model, device)
        print(f"Fooling rate for {args.tf_uap}: {tf_fooling_rate}")
        return

    print(args)
    # create a universal adversarial perturbation
    uap = gd_universal_adversarial_perturbation(model, args.model, args.prior_type, args.batch_size, device, args.val_dataset_name, args.patience_interval, args.id, disable_tqdm=True)

    # perform a final evaluation
    # final_data_loader = get_data_loader(args.final_dataset_name)
    test_data_loader = get_data_loader(args.val_dataset_name, batch_size=args.batch_size, train=False)
    acc, asr, fooling_rate = get_fooling_rate(model, uap.to(device), test_data_loader, device)
    print(f"validtion: acc: {acc}, fooling rate:{fooling_rate}, asr: {asr}")


if __name__ == '__main__':
    path = 'xxx/dataset/ImageNet/'
    main()
