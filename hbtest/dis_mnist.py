# coding=UTF-8
# hbsun-distributed-mnist
# hbsun


from __future__ import print_function
import torch.nn.functional as F

# imagenet
import os
import random
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Parameter():
    def __init__(self):
        self.batch_size = 64
        self.test_batch_size = 1000
        self.epochs = 10
        self.lr = 0.01
        self.momentum = 0.5
        self.no_cuda = False
        self.seed = 1
        self.log_interval = 10
        self.save_model = False
        self.rank = -1
        self.world_size = 1
        self.backend = 'gloo'
        self.no_cuda = False
        self.use_cuda = not self.no_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, input):
        x = F.relu(self.conv1(input))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Trainer:
    def __init__(self, args, model, train_loader, test_loader, optimizer, ):
        self.args = args
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        print('len(train_loader.dataset)=', len(train_loader.dataset), ' len(test_loader.dataset)=',
              len(test_loader.dataset))  # 总数量？
        print('len(train_loader)=', len(train_loader), ' len(test_loader)=', len(test_loader))  # 总批次个数？

    def adjust_learning_rate(self, epoch):  # 这是imagenet的策略，暂不用于mnist
        """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
        lr = self.args.lr * (0.1 ** (epoch // 30))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def train(self, epoch, train_sampler):
        print("Go into training")
        data_time = AverageMeter()  # 加载数据的时间
        forward_time = AverageMeter()  # 前向计算的时间
        backward_time = AverageMeter()  # 后向计算的时间
        update_time = AverageMeter()  # 模型更新的时间
        batch_time = AverageMeter()  # 整体的train_batch时间

        losses = AverageMeter()
        acces = AverageMeter()

        # switch to train mode
        self.model.train()  # 设置状态为训练状态，应该是考虑到了dropout和BN。

        train_sampler.set_epoch(epoch)

        # adjust_learning_rate(epoch)

        end = time.time()
        for batch_id, (data, target) in enumerate(self.train_loader):

            # measure data loading time
            data_time.update(time.time() - end)

            data, target = data.to(self.args.device, non_blocking=True), target.to(self.args.device,
                                                                                   non_blocking=True)  # https://discuss.pytorch.org/t/pytorch-imagenet-github-example-training-input-variable-not-cuda/710 这行代码到底加不加呢？

            # compute output(include the loss)
            start_forward = time.time()
            output = self.model(data)
            loss = F.nll_loss(output, target)  # 这个是标量，为均值
            forward_time.update(time.time() - start_forward)  # 我暂时认为前向计算的时间包含计算loss
            losses.update(loss.item(), target.size(0))

            # zero the history of gradient
            self.optimizer.zero_grad()

            # backward
            start_backward = time.time()
            loss.backward()
            backward_time.update(time.time() - start_backward)

            # update the model
            start_update = time.time()
            self.optimizer.step()
            update_time.update(time.time() - start_update)

            # measure elapsed time
            batch_time.update(time.time() - end)

            # compute acc #这个是我需要知道的额外信息，应该是可以不被包含在batch的时间里的。
            pred = 0
            correct = 0
            with torch.no_grad():
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct = pred.eq(target.view_as(pred)).sum().item()
                acces.update(correct, target.size(0))

            if batch_id % self.args.log_interval == 0:
                print('Train---Rank:{0} '  # 0
                      'Epoch: [{1}] [{2}/{3}]\t'  # 1      
                      'Overprogress [{4}/{5}]\t'  # 2
                      'TrainBatchTime {batch_time.val:.3f} ({batch_time.avg:.3f})\t'  # 3
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'  # 4
                      'Forward {forward_time.val:.3f} ({forward_time.avg:.3f})\t'  # 5
                      'Backward {backward_time.val:.3f} ({backward_time.avg:.3f})\t'  # 6
                      'Update {update_time.val:.3f} ({update_time.avg:.3f})\t'  # 7
                      'Acc {acc.val:.4f} ({acc.avg:.4f})\t'  # 8
                      'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(  # 9
                    dist.get_rank(),  # 0
                    epoch, batch_id, len(self.train_loader),  # 1
                    batch_id * len(data), len(self.train_loader.dataset),  # 2
                    batch_time=batch_time,  # 3
                    data_time=data_time,  # 4
                    forward_time=forward_time,  # 5
                    backward_time=backward_time,  # 6
                    update_time=update_time,  # 7
                    acc=acces,  # 8
                    loss=losses))  # 9

            end = time.time()

    def valid(self, epoch):
        batch_time = AverageMeter()  # 整体的test_batch时间
        data_time = AverageMeter()  # 加载数据的时间
        losses = AverageMeter()
        acces = AverageMeter()

        # switch to evaluate mode
        self.model.eval()

        with torch.no_grad():
            end = time.time()
            for batch_id, (data, target) in enumerate(self.test_loader):
                # measure data loading time
                data_time.update(time.time() - end)

                data, target = data.to(self.args.device, non_blocking=True), target.to(self.args.device,
                                                                                       non_blocking=True)  # https://discuss.pytorch.org/t/pytorch-imagenet-github-example-training-input-variable-not-cuda/710 这行代码到底加不加呢？

                output = self.model(data)
                loss = F.nll_loss(output, target,
                                  reduction='sum').item()  # sum up batch loss,default is mean  #https://pytorch.org/docs/stable/nn.html
                losses.update(loss, target.size(0))
                pred = output.max(1, keepdim=True)[1]
                correct = pred.eq(target.view_as(pred)).sum().item()
                acces.update(correct, target.size(0))

                batch_time.update(time.time() - end)

                if batch_id % self.args.log_interval == 0 or batch_id == len(self.test_loader) - 1:
                    print('Test---Rank:{0} '  # 0
                          'Epoch: [{1}] [{2}/{3}]\t'  # 1      
                          'Overprogress [{4}/{5}]\t'  # 2
                          'TrainBatchTime {batch_time.val:.3f} ({batch_time.avg:.3f})\t'  # 3
                          'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'  # 4
                          'Acc {acc.val:.4f} ({acc.avg:.4f})\t'  # 5
                          'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(  # 6
                        dist.get_rank(),  # 0
                        epoch, batch_id, len(self.test_loader),  # 1
                        batch_id * len(data), len(self.test_loader.dataset),  # 2
                        batch_time=batch_time,  # 3
                        data_time=data_time,  # 4
                        acc=acces,  # 5
                        loss=losses))  # 6

                end = time.time()


def process(args):
    print("Go into process:")
    # define model
    model = Net()
    #   model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model)  # distributed tag
    if args.device != torch.device('cpu'):
        print(args.device)
        print("how to convert model?", type(model.cuda()))
        model.cuda()  # 是不是这么写？不确定

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss()
    if args.device != torch.device('cpu'):
        criterion.cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum)

    cudnn.benchmark = True  # 模型的输入维度不变的话，设置这个会使cudn加速

    kwargs = {'num_workers': args.world_size, 'pin_memory': True}  # if args.use_cuda else {}
    data_transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_dataset = datasets.MNIST('../data', train=True, download=True, transform=data_transform)

    if args.world_size >= 1:  # 这个判断条件全面吗？
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)  # distributed tag
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size,
                                               shuffle=(train_sampler is None), sampler=train_sampler,
                                               **kwargs)  # train_sampler里包含了shuffle了。 batch_size是需要手动改变吗？

    test_dataset = datasets.MNIST('../data', train=False,
                                  transform=data_transform)  # 想知道它和train的大小关系？？test和train的测试集应该是分开的。
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True,
                                              **kwargs)  # test有必要shuffle吗？ 另外，test也是在多台机器上吗？ 如何分布式test?

    trainer = Trainer(args, model, train_loader, test_loader, optimizer)

    for epoch in range(1, args.epochs + 1):
        trainer.train(epoch, train_sampler)
        trainer.valid(epoch)


def main_worker(index, args, flag):
    args.rank = index
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29502'
    print(args.backend, args.rank, args.world_size)
    dist.init_process_group(backend=args.backend, rank=args.rank, world_size=args.world_size)
    process(args)


def main():
    # Training settings
    args = Parameter()
    print('args=', args)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '  # 不理解，为什么会降低速度；为什么会有不确定行为？
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    mp.spawn(main_worker, nprocs=args.world_size, args=(args, 1))

    return

    processes = []
    for rank in range(args.world_size):
        args.rank = rank
        p = Process(target=main_worker, args=(args, 1))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == '__main__':
    main()
