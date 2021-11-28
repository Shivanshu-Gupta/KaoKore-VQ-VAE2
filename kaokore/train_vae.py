# taken from https://github.com/nadavbh12/VQ-VAE
from collections import defaultdict
from functools import partial
import os, sys
import time
import logging
import argparse
import comet_ml
from comet_ml import comet
import torch
import numpy as np
import torch.backends.cudnn as cudnn

from tqdm import tqdm
from pathlib import Path
from torch import optim
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

from utils import setup_logging_from_args
from data import Kaokore
from models.vae import VAE, VQ_VAE, CVAE, VQ_CVAE

model_classes = {
    'vae': VAE,
    'vqvae': VQ_VAE,
    'cvae': CVAE,
    'vqcvae': VQ_CVAE
}

dataset_models = {
    'custom': ['vqcvae'],
    'imagenet': ['vqcvae'],
    'kaokore': ['cvae', 'vqcvae'],
    'cifar10': ['cvae', 'vqcvae'],
    'mnist': ['vae', 'vqcvae']
}
datasets_classes = {
    'custom': datasets.ImageFolder,
    'imagenet': datasets.ImageFolder,
    'kaokore': Kaokore,
    'cifar10': datasets.CIFAR10,
    'mnist': datasets.MNIST
}
kaokore_category = 'gender'
dataset_train_args = {
    'custom': {},
    'imagenet': {},
    'kaokore': {'category': kaokore_category, 'split': 'train'},
    'cifar10': {'train': True, 'download': True},
    'mnist': {'train': True, 'download': True},
}
dataset_test_args = {
    'custom': {},
    'imagenet': {},
    'kaokore': {'category': kaokore_category, 'split': 'test'},
    'cifar10': {'train': False, 'download': True},
    'mnist': {'train': False, 'download': True},
}
dataset_n_channels = {
    'custom': 3,
    'imagenet': 3,
    'kaokore': 3,
    'cifar10': 3,
    'mnist': 1,
}

def get_transform(resize=False, crop=False, normalize=False):
    transform_list = []
    if resize: transform_list.append(transforms.Resize(256))
    if crop: transform_list.append(transforms.CenterCrop(256))
    transform_list.append(transforms.ToTensor())
    if normalize: transform_list.append(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)))
    if len(transform_list) == 1: return transform_list[0]
    else: return transforms.Compose(transform_list)

dataset_transforms = {
    'custom': get_transform(resize=True, crop=True, normalize=True),
    'imagenet': get_transform(resize=True, crop=True, normalize=True),
    'kaokore': get_transform(resize=False, crop=False, normalize=True),
    'cifar10': get_transform(resize=False, crop=False, normalize=True),
    'mnist': get_transform(resize=False, crop=False, normalize=False),
}

default_hyperparams = {
    'custom': {'lr': 2e-4, 'k': 512, 'hidden': 128},
    'imagenet': {'lr': 2e-4, 'k': 512, 'hidden': 128},
    'kaokore': {'lr': 2e-4, 'k': 10, 'hidden': 256},
    'cifar10': {'lr': 2e-4, 'k': 10, 'hidden': 256},
    'mnist': {'lr': 1e-4, 'k': 10, 'hidden': 64}
}

input_sizes = {
    'kaokore': 256,
    'mnist': 32
}

def main(args: argparse.Namespace):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    dataset_dir_name = args.dataset if args.dataset != 'custom' else args.dataset_dir_name

    args.lr = args.lr or default_hyperparams[args.dataset]['lr']
    args.k = args.k or default_hyperparams[args.dataset]['k']
    args.hidden = args.hidden or default_hyperparams[args.dataset]['hidden']
    args.num_channels = dataset_n_channels[args.dataset]

    save_path = setup_logging_from_args(args)
    writer = SummaryWriter(save_path)

    experiment = None
    if args.comet:
        print("COMET_ML_API_KEY: ", os.getenv("COMET_ML_API_KEY"))
        experiment = comet_ml.Experiment(api_key=os.getenv("COMET_ML_API_KEY"),
                                        project_name='cs274e')
        if args.dataset != 'kaokore':
            experiment.set_name(f'{args.dataset}/{args.model}[channels={args.hidden}, k={args.k}]')
        else:
            experiment.set_name(f'{args.dataset}[{kaokore_category}]/{args.model}[channels={args.hidden}, k={args.k}]')
        experiment.add_tags([args.dataset, args.model,
                            f'lr={args.lr}',
                            f'batch={args.batch_size}'])
        if args.dataset == 'kaokore': experiment.add_tag(kaokore_category)
        experiment.log_parameters(args.__dict__)


    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        args.gpus = [int(i) for i in args.gpus.split(',')]
        torch.cuda.set_device(args.gpus[0])
        cudnn.benchmark = True
        torch.cuda.manual_seed(args.seed)

    model = model_classes[args.model](args.hidden, k=args.k, num_channels=args.num_channels,
                                      input_size=input_sizes[args.dataset])
    if args.cuda: model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 10 if args.dataset == 'imagenet' else 30, 0.5,)

    def _get_dataloader(dataset_dir, dataset, train):
        kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
        dataset_args = dataset_train_args if train else dataset_test_args
        print(dataset_args[dataset])
        if dataset in ['imagenet', 'custom']:
            dataset_dir = dataset_dir / 'train' if train else 'test'
        loader = torch.utils.data.DataLoader(
            datasets_classes[dataset](
                str(dataset_dir),
                transform=dataset_transforms[dataset],
                **dataset_args[dataset]),
            batch_size=args.batch_size, shuffle=train, **kwargs)
        return loader

    if args.dataset == 'kaokore': dataset_dir = Path(args.data_dir)
    else: dataset_dir = Path(args.data_dir) / dataset_dir_name
    train_loader = _get_dataloader(dataset_dir, args.dataset, True)
    test_loader = _get_dataloader(dataset_dir, args.dataset, False)

    for epoch in range(1, args.epochs + 1):
        print()
        print('Epoch', epoch)
        if args.comet:
            with experiment.train():
                train_losses = train(epoch, model, train_loader, optimizer, args.cuda,
                                    args.log_interval, save_path, args, writer, experiment)
                experiment.log_metrics(train_losses, epoch=epoch)
            with experiment.test():
                test_losses = test_net(epoch, model, test_loader, args.cuda, save_path, args, writer, experiment)
                experiment.log_metrics(test_losses, epoch=epoch)
        else:
            train_losses = train(epoch, model, train_loader, optimizer, args.cuda,
                                args.log_interval, save_path, args, writer, experiment)
            test_losses = test_net(epoch, model, test_loader, args.cuda, save_path, args, writer, experiment)


        for k in train_losses.keys():
            # name = k.replace('_train', '')
            # train_name = k
            # test_name = k.replace('train', 'test')

            writer.add_scalars(k, {
                'train': train_losses[k],
                'test': test_losses[k],
                })

        scheduler.step()

def print_atom_hist(argmin):
    argmin = argmin.detach().cpu().numpy()
    unique, counts = np.unique(argmin, return_counts=True)
    print('historgrams')
    print(counts)
    print(unique)


def train(epoch, model, train_loader, optimizer, cuda, log_interval, save_path, args, writer, experiment: comet_ml.Experiment=None):
    model.train()
    losses = defaultdict(lambda: 0)
    epoch_losses = defaultdict(lambda: 0)
    batch_idx, data = None, None
    bar = tqdm(train_loader)
    for batch_idx, (data, _) in enumerate(bar):
        if cuda:
            data = data.cuda()
        optimizer.zero_grad()
        outputs = model(data)
        loss = model.loss(data, *outputs)
        loss.backward()
        optimizer.step()
        latest_losses = model.latest_losses()
        for key in latest_losses:
            losses[key] += float(latest_losses[key])
            epoch_losses[key] += float(latest_losses[key])

        if batch_idx % log_interval == 0:
            for key in latest_losses:
                losses[key] /= log_interval
            loss_string = ' '.join(['{}: {:.5f}'.format(k, v) for k, v in losses.items()])
            bar.set_description(f'train: {loss_string}')
            # print('z_e norm: {:.2f}'.format(float(torch.mean(torch.norm(outputs[1][0].contiguous().view(256,-1),2,0)))))
            # print('z_q norm: {:.2f}'.format(float(torch.mean(torch.norm(outputs[2][0].contiguous().view(256,-1),2,0)))))
            losses = defaultdict(lambda: 0)

        if batch_idx == (len(train_loader) - 1):
            # save_reconstructed_images(data, epoch, outputs[0], save_path, 'reconstruction_train')
            write_images(data, outputs[0], save_path, writer, experiment, f'train-{epoch}')

        if args.dataset in ['imagenet', 'custom'] and batch_idx * len(data) > args.max_epoch_samples:
            break

    for key in epoch_losses:
        epoch_losses[key] /= (len(train_loader.dataset) / train_loader.batch_size)
    loss_string = '\t'.join(['{}: {:.6f}'.format(k, v) for k, v in epoch_losses.items()])
    print('Epoch {} train: {}'.format(epoch, dict(epoch_losses)))
    if args.model == 'vqcvae':
        writer.add_histogram('dict_frequency', outputs[3], bins=range(args.k + 1))
        print_atom_hist(outputs[3])
    return epoch_losses


def test_net(epoch, model, test_loader, cuda, save_path, args, writer, experiment: comet_ml.Experiment=None):
    model.eval()
    losses = defaultdict(lambda: 0)
    i, data = None, None
    with torch.no_grad():
        bar = tqdm(test_loader)
        for i, (data, _) in enumerate(bar):
            if cuda:
                data = data.cuda()
            outputs = model(data)
            model.loss(data, *outputs)
            latest_losses = model.latest_losses()
            for key in latest_losses:
                losses[key] += float(latest_losses[key])
            if i == 0:
                write_images(data, outputs[0], save_path, writer, experiment, f'test-{epoch}')
                # save_reconstructed_images(data, epoch, outputs[0], save_path, 'reconstruction_test')
                save_checkpoint(model, epoch, save_path)
            if args.dataset == 'imagenet' and i * len(data) > 1000:
                break

    for key in losses:
        if args.dataset not in ['imagenet', 'custom']:
            losses[key] /= (len(test_loader.dataset) / test_loader.batch_size)
        else:
            losses[key] /= (i * len(data))
    loss_string = ' '.join(['{}: {:.6f}'.format(k, v) for k, v in losses.items()])
    print('Epoch {} test: {}'.format(epoch, dict(losses)))
    return losses


def write_images(data, outputs, save_path, writer, experiment: comet_ml.Experiment, name, sample_size=8):
    original = data[:sample_size].mul(0.5).add(0.5)
    reconstructed = outputs[:sample_size].mul(0.5).add(0.5)
    image_grid = make_grid(torch.cat([original, reconstructed], 0))
    # original_grid = make_grid(original[:6])
    # writer.add_image(f'original/{name}', original_grid)
    # reconstructed_grid = make_grid(reconstructed[:6])
    # writer.add_image(f'reconstructed/{name}', reconstructed_grid)
    writer.add_image(name, image_grid)
    save_image(image_grid.cpu(), save_path / f'{name}.png',
               nrow=sample_size, normalize=True)

    if experiment:
        comet_image = partial(experiment.log_image, image_channels='first', overwrite=True)
        # comet_image(image_data=original_grid.to('cpu'), name=f'original/{name}')
        # comet_image(image_data=reconstructed_grid.to('cpu'), name=f'reconstructed/{name}')
        comet_image(image_data=image_grid.cpu(), name=name)


def save_reconstructed_images(data, epoch, outputs, save_path: Path, name):
    size = data.size()
    n = min(data.size(0), 8)
    batch_size = data.size(0)
    comparison = torch.cat([data[:n],
                            outputs.view(batch_size, size[1], size[2], size[3])[:n]])
    save_image(comparison.cpu(),
               save_path / f'{name}_{epoch}.png',
               nrow=n, normalize=True)


def save_checkpoint(model, epoch, save_path: Path):
    checkpoint_path = save_path / 'checkpoints' / f'model_{epoch}.pth'
    os.makedirs(checkpoint_path.parent, exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)

def get_argparser():
    parser = argparse.ArgumentParser(description='Variational AutoEncoders')

    model_parser = parser.add_argument_group('Model Parameters')
    model_parser.add_argument('--model', default='vae', choices=model_classes.keys(),
                              help='autoencoder variant to use')
    model_parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                              help='input batch size for training (default: 128)')
    model_parser.add_argument('--hidden', type=int, metavar='N',
                              help='number of hidden channels')
    model_parser.add_argument('-k', '--dict-size', type=int, dest='k', metavar='K',
                              help='number of atoms in dictionary')
    model_parser.add_argument('--lr', type=float, default=None,
                              help='learning rate')
    model_parser.add_argument('--vq_coef', type=float, default=None,
                              help='vq coefficient in loss')
    model_parser.add_argument('--commit_coef', type=float, default=None,
                              help='commitment coefficient in loss')
    model_parser.add_argument('--kl_coef', type=float, default=None,
                              help='kl-divergence coefficient in loss')

    training_parser = parser.add_argument_group('Training Parameters')
    training_parser.add_argument('--dataset', default='cifar10',
                                 choices=['mnist', 'cifar10', 'kaokore', 'imagenet', 'custom'],
                                 help='dataset to use')
    training_parser.add_argument('--dataset_dir_name', default='',
                                 help='name of the dir containing the dataset if dataset == custom')
    training_parser.add_argument('--data-dir', default='/media/ssd/Datasets',
                                 help='directory containing the dataset')
    training_parser.add_argument('--epochs', type=int, default=20, metavar='N',
                                 help='number of epochs to train (default: 10)')
    training_parser.add_argument('--max-epoch-samples', type=int, default=50000,
                                 help='max num of samples per epoch')
    training_parser.add_argument('--no-cuda', action='store_true', default=False,
                                 help='enables CUDA training')
    training_parser.add_argument('--seed', type=int, default=1, metavar='S',
                                 help='random seed (default: 1)')
    training_parser.add_argument('--gpus', default='0',
                                 help='gpus used for training - e.g 0,1,3')
    training_parser.add_argument('--comet', action='store_true',
                                 help='whether to use comet_ml for logging')

    logging_parser = parser.add_argument_group('Logging Parameters')
    logging_parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                                help='how many batches to wait before logging training status')
    logging_parser.add_argument('--results-dir', metavar='RESULTS_DIR', default='./results',
                                help='results dir')
    logging_parser.add_argument('--save-name', default='',
                                help='saved folder')
    logging_parser.add_argument('--data-format', default='json',
                                help='in which format to save the data')
    return parser

if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args(sys.argv[1:])
    main(args)
