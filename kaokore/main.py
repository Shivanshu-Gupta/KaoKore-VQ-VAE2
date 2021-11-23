import os
import time
import logging
import argparse

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image, make_grid


parser = argparse.ArgumentParser()

parser.add_argument('--dataset_dir', default='kaokore-dataset', help='name of the dir containing the dataset')
parser.add_argument('--cuda', action='store_true', help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--batch_size', type=int, default=32, help='batch size (default: 32)')
parser.add_argument('--log_interval', type=int, default=10, help='batch ogging interval (default: 10)')
parser.add_argument('--num_epochs', type=int, default=50, help='number of training epochs (default: 50)')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate (default: 1e-4)')
parser.add_argument('--results_dir', metavar='RESULTS_DIR', default='./results', help='results dir')

args = parser.parse_args()


def train(epoch, model, loader, optimizer, cuda, log_interval, results_dir):
    model.train()
    losses = torch.zeros(len(loader))
    start_time = time.time()
    for batch_idx, (data, _) in enumerate(loader):
        if cuda:
            data = data.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(data, output)
        loss.backward()
        optimizer.step()
        losses[batch_idx] = loss.item()

        if batch_idx % log_interval == 0 and len(losses) >= log_interval:
            logging.info('Epoch: {} ({}/{}), last_time: {:3.2f}, last_loss: {loss}'.format(
                epoch, 
                batch_idx * len(data), 
                len(loader) * len(data),
                time=time.time() - start_time,
                loss=torch.mean(losses[-log_interval:])
            ))
            start_time = time.time()

    logging.info('Epoch: {}\n\tTraining loss: {}'.format(epoch, torch.mean(losses)))
    return losses


def test(epoch, model, loader, cuda, results_dir):
    model.eval()
    losses = torch.zeros(len(loader))
    with torch.no_grad():
        for batch_idx, (data, _) in enumerate(loader):
            if cuda:
                data = data.cuda()
            output = model(data)
            loss = loss_function(data, output)
            losses[batch_idx] = loss.item()
            if batch_idx == 0:
                save_visualizations(data, output, epoch, results_dir)  # TODO

    torch.save(
        model.state_dict(), 
        os.path.join(results_dir, 'checkpoints', 'model_{}.pth'.format(epoch))
    )
    logging.info('\tTesting losses: {}'.format(torch.mean(losses)))
    return losses


def loss_function(data, output):
    return None  # TODO


if __name__ == "__main__":
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    writer = SummaryWriter(args.results_dir)
    os.makedirs(os.path.join(args.results_dir, 'checkpoints'), exist_ok=True)

    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)

    model = VQVAE()  # TODO
    if args.cuda:
        model.cuda()

    optimizer = Adam(model.parameters(), lr=args.lr)

    dataset_train, dataset_test = get_data_splits(args.dataset_dir)  # TODO
    loader_train = DataLoader(dataset_train, batch_size=args.batch_size, 
                              shuffle=True, num_workers=8, pin_memory=args.cuda)
    loader_test = DataLoader(dataset_test, batch_size=args.batch_size, 
                             shuffle=False, num_workers=8, pin_memory=args.cuda)

    for epoch in range(args.num_epochs):
        losses_train = train(epoch, model, loader_train, optimizer, args.cuda, 
                           args.log_interval, args.results_dir)
        losses_test = test(epoch, model, loader_test, args.cuda, 
                         args.results_dir)

        writer.add_scalars("loss", {
            'train': torch.mean(losses_train),
            'test': torch.mean(losses_test),
        })