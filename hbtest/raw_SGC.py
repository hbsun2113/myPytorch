# coding=UTF-8
# hbsun-distributed-mnist
# hbsun

import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from math import ceil
from random import Random
from torch.multiprocessing import Process
from torch.autograd import Variable
from torchvision import datasets, transforms


class Net(nn.Module):
    """ Network architecture. """

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


""" Dataset partitioning helper """


class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class GlobalMomentum(object):
    def __init__(self, alpha, model):  # alpha是动量参数
        print("rank=", rank, " global_momentum.alpha=", alpha, " type=", type(alpha))
        self.alpha = alpha
        self.momentum = []
        for param in model.parameters():
            self.momentum.append(torch.zeros(param.size()))

    def __call__(self, model):
        for index, param in enumerate(model.parameters()):
            # print ('rank=',rank," index=",index)
            current_momentum = self.alpha * self.momentum[index] + param.grad.data
            self.momentum[index] = current_momentum
            param.grad.data = current_momentum


class Filter(object):
    def __init__(self, threshold, model):
        print("rank=", rank, " filter.threshold=", threshold, " type=", type(threshold))
        self.threshold = threshold
        self.residual = []
        self.logrecord = 0
        for param in model.parameters():
            self.residual.append(torch.zeros(param.size()))

    def __call__(self, model):
        if self.threshold == 0.0:
            return  # 这行语句不知道会不会起作用
        for index, param in enumerate(model.parameters()):
            current_residual = self.residual[index] + param.grad.data
            abs_residual = abs(current_residual)
            one_residual = abs_residual.view(-1, 1)  # 展开成一维数组
            sorted_residual = sorted(one_residual)
            assert (sorted_residual[0] <= sorted_residual[len(sorted_residual) - 1])
            filter_value = sorted_residual[int(len(sorted_residual) * self.threshold)]
            #             if(self.logrecord%3==0 and index%5==0):
            #               print ("logrecord=",self.logrecord," param.index=",index," threshold=",self.threshold," filter_value=",filter_value)
            mask = abs_residual > filter_value
            param.grad = torch.where(mask, current_residual, torch.tensor([0.]))
            self.residual[index] = torch.where(~mask, current_residual, torch.tensor([0.]))
        self.logrecord += 1


class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])


""" Partitioning MNIST """


def partition_dataset():
    dataset = datasets.MNIST('./data', train=True, download=True,
                             transform=transforms.Compose([
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.1307,), (0.3081,))
                             ]))
    size = dist.get_world_size()
    bsz = 128 / float(size)
    bsz = int(bsz)
    partition_sizes = [1.0 / size for _ in range(size)]
    partition = DataPartitioner(dataset, partition_sizes)
    partition = partition.use(dist.get_rank())
    train_set = torch.utils.data.DataLoader(partition,
                                            batch_size=bsz,
                                            shuffle=True)
    return train_set, bsz


""" Gradient averaging. """


def average_gradients(model):
    size = float(dist.get_world_size())
    if (size == 1):  # 这两行代码是我hbsun加上去的。如果只有一个节点，自然是不需要reduce和avg了。
        return
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
        param.grad.data /= size


""" Distributed Synchronous SGD Example """


def run(rank, size):
    torch.manual_seed(1234)
    train_set, bsz = partition_dataset()
    model = Net()
    optimizer = optim.SGD(model.parameters(),
                          lr=0.01, momentum=0.5)

    num_batches = ceil(len(train_set.dataset) / float(bsz))
    calculate_global_momentum = GlobalMomentum(optimizer.__dict__['defaults']['momentum'], model)
    filter_gradient = Filter(0.9, model)
    for epoch in range(10):
        epoch_loss = 0.0
        for data, target in train_set:
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.item()
            loss.backward()
            filter_gradient(model)
            average_gradients(model)
            # calculate_global_momentum(model)
            optimizer.step()
        print('Rank ', dist.get_rank(), ', epoch ',
              epoch, ': ', epoch_loss / num_batches)


def init_processes(rank, size, fn, backend='Gloo'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)


if __name__ == "__main__":
    size = 2
    processes = []
    for rank in range(size):
        p = Process(target=init_processes, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("\nfinish")
