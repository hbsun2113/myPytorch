from __future__ import division, print_function

import argparse

import torch
import torch.nn.functional as F
from torch import distributed, nn
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms


class AverageMeter(object):

    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, value, number):
        self.sum += value * number
        self.count += number

    @property
    def average(self):
        return self.sum / self.count


class AccuracyMeter(object):

    def __init__(self):
        self.correct = 0
        self.count = 0

    def update(self, output, label):
        predictions = output.data.argmax(dim=1)
        correct = predictions.eq(label.data).sum().item()

        self.correct += correct
        self.count += output.size(0)

    @property
    def accuracy(self):
        return self.correct / self.count


class Trainer(object):

    def __init__(self, net, optimizer, train_loader, test_loader, device):
        self.net = net
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    def train(self):
        train_loss = AverageMeter()
        train_acc = AccuracyMeter()

        self.net.train()

        for data, label in self.train_loader:
            data = data.to(self.device)
            label = label.to(self.device)

            output = self.net(data)
            loss = F.cross_entropy(output, label)

            self.optimizer.zero_grad()
            loss.backward()
            # average the gradients
            self.average_gradients()
            self.optimizer.step()

            train_loss.update(loss.item(), data.size(0))
            train_acc.update(output, label)

        return train_loss.average, train_acc.accuracy

    def evaluate(self):
        test_loss = AverageMeter()
        test_acc = AccuracyMeter()

        self.net.eval()

        with torch.no_grad():
            for data, label in self.test_loader:
                data = data.to(self.device)
                label = label.to(self.device)

                output = self.net(data)
                loss = F.cross_entropy(output, label)

                test_loss.update(loss.item(), data.size(0))
                test_acc.update(output, label)

        return test_loss.average, test_acc.accuracy

    def average_gradients(self):
        world_size = distributed.get_world_size()

        for p in self.net.parameters():
            distributed.all_reduce(p.grad.data, op=distributed.reduce_op.SUM)
            p.grad.data /= float(world_size)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(784, 10)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


def get_dataloader(root, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.13066047740239478,), (0.3081078087569972,))
    ])

    train_set = datasets.MNIST(
        root, train=True, transform=transform, download=True)
    sampler = DistributedSampler(train_set)

    train_loader = data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler)

    test_loader = data.DataLoader(
        datasets.MNIST(root, train=False, transform=transform, download=True),
        batch_size=batch_size,
        shuffle=False)

    return train_loader, test_loader


def solve(args):
    device = torch.device('cuda' if args.cuda else 'cpu')

    net = Net().to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

    train_loader, test_loader = get_dataloader(args.root, args.batch_size)

    trainer = Trainer(net, optimizer, train_loader, test_loader, device)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = trainer.train()
        test_loss, test_acc = trainer.evaluate()

        print(
            'Epoch: {}/{},'.format(epoch, args.epochs),
            'train loss: {:.6f}, train acc: {:.6f}, test loss: {:.6f}, test acc: {:.6f}.'.
            format(train_loss, train_acc, test_loss, test_acc))


def init_process(args):
    distributed.init_process_group(
        backend=args.backend,
        init_method=args.init_method,
        rank=args.rank,
        world_size=args.world_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--backend',
        type=str,
        default='gloo',
        help='Name of the backend to use.')
    parser.add_argument(
        '--init-method',
        '-i',
        type=str,
        default='tcp://127.0.0.1:23456',
        help='URL specifying how to initialize the package.')
    parser.add_argument(
        '--rank', '-r', type=int, help='Rank of the current process.')
    parser.add_argument(
        '--world-size',
        '-s',
        type=int,
        help='Number of processes participating in the job.')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-3)
    parser.add_argument('--root', type=str, default='data')
    parser.add_argument('--batch-size', type=int, default=128)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available() and not args.no_cuda
    print(args)

    init_process(args)
    solve(args)


if __name__ == '__main__':
    main()
