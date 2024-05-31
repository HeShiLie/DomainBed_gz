import argparse
import collections
import json
import os
import random
import sys
import time
import uuid

import numpy as np
import PIL
import torch
import torchvision
import torch.utils.data

from domainbed import datasets
from domainbed import hparams_registry
from domainbed import algorithms
from domainbed.lib import misc
from domainbed.lib.fast_data_loader import InfiniteDataLoader, FastDataLoader

def compute_mu_sigma(algorithm, loader, device):
    '''
    save the output of shape [N,n_mu_sigma*2] and reshape as [n_mu_sigma*2,N]
    '''
    algorithm.eval()

    saving_list = []
    for x, y in loader:
        mu_sigma = algorithm.predict(x.to(device))
        saving_list.append(mu_sigma)

    saving_tensor = torch.cat(saving_list, dim=0)
    saving_tensor = saving_tensor.transpose(0, 1)

    return saving_tensor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='testing the idea')
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--dataset', type=str, default="RotatedMNIST")
    parser.add_argument('--algorithm', type=str, default="ERM")
    # parser.add_argument('--task', type=str, default="domain_generalization",
    #     choices=["domain_generalization", "domain_adaptation"])
    parser.add_argument('--algorithm_dict', type=str, default=None)
    parser.add_argument('--hparams', type=str,
        help='JSON-serialized hparams dict')
    parser.add_argument('--hparams_seed', type=int, default=0,
        help='Seed for random hparams (0 means "default hparams")')
    parser.add_argument('--trial_seed', type=int, default=0,
        help='Trial number (used for seeding split_dataset and '
        'random_hparams).')
    parser.add_argument('--seed', type=int, default=0,
        help='Seed for everything else')
    # parser.add_argument('--steps', type=int, default=None,
    #     help='Number of steps. Default is dataset-dependent.')
    # parser.add_argument('--checkpoint_freq', type=int, default=None,
    #     help='Checkpoint every N steps. Default is dataset-dependent.')
    # parser.add_argument('--test_envs', type=int, nargs='+', default=[0])
    parser.add_argument('--output_dir', type=str, default="train_output")
    parser.add_argument('--holdout_fraction', type=float, default=0.2)
    # parser.add_argument('--uda_holdout_fraction', type=float, default=0,
    #     help="For domain adaptation, % of test to use unlabeled for training.")
    # parser.add_argument('--skip_model_save', action='store_true')
    # parser.add_argument('--save_model_every_checkpoint', action='store_true')
    args = parser.parse_args()

    # If we ever want to implement checkpointing, just persist these values
    # every once in a while, and then load them from disk here.
    # start_step = 0
    algorithm_dict = parser.algorithm_dict

    os.makedirs(args.output_dir, exist_ok=True)
    sys.stdout = misc.Tee(os.path.join(args.output_dir, 'out.txt'))
    sys.stderr = misc.Tee(os.path.join(args.output_dir, 'err.txt'))

    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))

    print('Args:')
    for k, v in sorted(vars(args).items()):
        print('\t{}: {}'.format(k, v))

    # load the algorithm
    #   load the hparams
    if args.hparams_seed == 0:
        hparams = hparams_registry.default_hparams(args.algorithm, args.dataset)
    else:
        hparams = hparams_registry.random_hparams(args.algorithm, args.dataset,
            misc.seed_hash(args.hparams_seed, args.trial_seed))
    if args.hparams:
        hparams.update(json.loads(args.hparams))

    print('HParams:')
    for k, v in sorted(hparams.items()):
        print('\t{}: {}'.format(k, v))
    # setting the seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load the dataset
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if args.dataset in vars(datasets):
        dataset = vars(datasets)[args.dataset](args.data_dir,
            args.test_envs, hparams)
    else:
        raise NotImplementedError
    #   creat loaders list contains domain_nums loaders for each domain
    eval_loaders = [FastDataLoader(
        dataset=env,
        batch_size=64,
        num_workers=dataset.N_WORKERS)
        for env, _ in dataset] 

    # out put the mu and sigma
    #   load the model
    algorithm_class = algorithms.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(dataset.input_shape, dataset.num_classes,
        len(dataset) - len(args.test_envs), hparams)

    if algorithm_dict is not None:
        algorithm.load_state_dict(algorithm_dict)  

    algorithm.to(device)
            
    # save the mu and sigma
    saving_tensor = compute_mu_sigma(algorithm, eval_loaders[0], device)
    torch.save(saving_tensor, os.path.join(args.output_dir, 'mu_sigma.pt'))