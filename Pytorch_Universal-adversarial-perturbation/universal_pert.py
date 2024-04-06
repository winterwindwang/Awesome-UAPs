import numpy as np
from deepfool import deepfool
from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm
from torch.autograd import Variable
from torch.utils.data import Subset

def proj_lp(v, xi, p):

    # Project on the lp ball centered at 0 and of radius xi

    # SUPPORTS only p = 2 and p = Inf for now
    if p == 2:
        v = v * min(1, xi/np.linalg.norm(v.flatten(1)))
    elif p == np.inf:
        v = np.sign(v) * np.minimum(abs(v), xi)
    else:
         raise ValueError('Values of p different from 2 and Inf are currently not supported...')

    return v


def universal_perturbation(args, dataset,
                           valset,
                           f,
                           delta=0.2,
                           max_iter_uni = np.inf,
                           xi=10/255.0,
                           p=np.inf,
                           num_classes=10,
                           overshoot=0.02,
                           max_iter_df=10):
    """
    :param dataset: Images of size MxHxWxC (M: number of images)
    :param f: feedforward function (input: images, output: values of activation BEFORE softmax).
    :param grads: gradient functions with respect to input (as many gradients as classes).
    :param delta: controls the desired fooling rate (default = 80% fooling rate)
    :param max_iter_uni: optional other termination criterion (maximum number of iteration, default = np.inf)
    :param xi: controls the l_p magnitude of the perturbation (default = 10)
    :param p: norm to be used (FOR NOW, ONLY p = 2, and p = np.inf ARE ACCEPTED!) (default = np.inf)
    :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
    :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
    :param max_iter_df: maximum number of iterations for deepfool (default = 10)
    :return: the universal perturbation.
    """

    # sampler=train_sampler, drop_last=True)
    nb_images = 2000
    np.random.seed(1024)
    sample_indices = np.random.permutation(range(valset.__len__()))[:nb_images]
    valset = Subset(valset, sample_indices)

    

    valset_loader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size,
                                                    num_workers=args.num_works)


    print('p =', p, xi)
    v = 0
    fooling_rate = 0.0
    best_fooling = 0.0
    num_images = nb_images   # The length of testing data

    if args.nb_images <= 2000: 
        delta = 0.2
    else:
        delta = 0.1
    idx = 0
    while best_fooling < 1-delta and idx < 20:
        # Shuffle the dataset
        data_loader = DataLoader(dataset, batch_size = 1, shuffle = True, pin_memory=True)

        # Go through the data set and compute the perturbation increments sequentially
        k = 0
        f.cuda()
        for cur_img, _ in tqdm(data_loader):
            k += 1
            cur_img = cur_img.cuda()
            per = Variable(cur_img + torch.tensor(v).cuda(), requires_grad = True)
            if int(f(cur_img).argmax()) == int(f(per).argmax()):
                # Compute adversarial perturbation
                f.zero_grad()
                dr, iter = deepfool(per,
                                   f,
                                   num_classes = num_classes,
                                   overshoot = overshoot,
                                   max_iter = max_iter_df)
                # print('dr = ', abs(dr).max())

                # Make sure it converged...
                if iter < max_iter_df-1:
                    v = v + dr
                    v = proj_lp(v, xi, p)

        # Perturb the dataset with computed perturbation
        # dataset_perturbed = dataset + v
        est_labels_orig = torch.zeros((num_images)).cuda()
        est_labels_pert = torch.zeros((num_images)).cuda()
        idx += 1
        # Compute the estimated labels in batches
        ii = 0
        with torch.no_grad():
            for img_batch, _ in tqdm(valset_loader):
                m = (ii * args.batch_size)
                M = min((ii + 1) * args.batch_size, num_images)
                img_batch = img_batch.cuda()
                per_img_batch = (img_batch + torch.tensor(v).cuda()).cuda()
                ii += 1
                # print(img_batch.shape)
                # print(m, M)
                est_labels_orig[m:M] = torch.argmax(f(img_batch), dim=1)
                est_labels_pert[m:M] = torch.argmax(f(per_img_batch), dim=1)

            # Compute the fooling rate
            fooling_rate = torch.sum(est_labels_pert != est_labels_orig).float() / num_images
            print("Fool num", torch.sum(est_labels_pert != est_labels_orig).float().item())
            print('FOOLING RATE = ', fooling_rate.item())
            if fooling_rate > best_fooling:
                torch.save(torch.tensor(v), f'{args.save_dir}/Revised_best_uap_{args.model_name}-eps10-fr{fooling_rate}.pth')
                best_fooling = fooling_rate
            print('Best Fooling Rate = ', best_fooling.item())
            # pertbation_name = f'{args.save_dir}/Test-{np.round(abs(v).max(),2)}-{np.round(fooling_rate.item()*100, 2)}.npy'
            # np.save(pertbation_name, v)

    return v