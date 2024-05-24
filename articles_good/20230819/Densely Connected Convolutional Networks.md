
作者：禅与计算机程序设计艺术                    

# 1.简介
  

DenseNets是一种新的CNN模型，其特点是通过密集连接的方式融合了多个卷积层而不会出现参数冗余，从而达到提升模型性能的效果。本文将会介绍 DenseNet 的相关知识、结构、优点和实现方法。
## 1.1 什么是 DenseNet？
DenseNet (Densely Connected Convolutional Networks) 是一种基于深度学习的神经网络。它在传统的 CNN 基础上，增加了“密集连接”（dense connectivity）的机制，使得网络能够学习到更强大的特征表示。在标准的 CNN 中，每一层都直接与其上一层相连，但是这样会导致网络难以训练，因此 DenseNet 提出了一种新型的连接方式——密集连接（dense connectivity）。这种连接方式保证了每一层的输入都会与前面所有层的输出进行连接，这有效地扩充了网络的感受野。
DenseNet 的基本思想是将每一个层都与后续所有层进行密集连接。通过这种方式，可以令网络学到更加紧凑的、高级的特征表示，同时又不必担心参数冗余的问题。DenseNet 的这种连接方式带来的好处主要有两方面。一方面，由于每一层都接收了来自所有先前层的输入，因此可以充分利用这些先验信息来产生有效的特征表示；另一方面，通过使用跳跃链接（skip connections），DenseNet 可以显著减少梯度消失或爆炸的风险，并帮助优化网络的收敛速度。

## 1.2 为什么要用 DenseNet？
DenseNet 具有以下几个显著优点：

1. 能力强大：DenseNet 在图像分类任务中获得了最好的结果。在 CIFAR10 和 ImageNet 数据集上的最新结果表明，它的准确率超过其他 CNN 模型。DenseNet 甚至可以在更复杂的数据集上获得更好的效果。
2. 参数减少：DenseNet 使用了稀疏连接，因此模型的大小比传统的 CNN 模型小很多。
3. 层次性增强：因为每个层都与后续所有层连接，所以 DenseNet 的模型具有更强的层次性，这进一步增强了特征提取的能力。
4. 特征重用：DenseNet 通过增加额外的连接，因此同一层的不同位置可以共享相同的权值。这有利于网络学习到更深层的抽象特征，并且可以避免过拟合现象的发生。
5. 适应多尺寸输入：虽然 DenseNet 只限于处理固定尺寸的图像，但它还是可以处理变化较小的尺度。在测试阶段，可以在不同的输入尺度下，对相同的 DenseNet 模型进行微调，来获得最佳的结果。

总之，DenseNet 的出现使得卷积神经网络（CNN）在图像识别领域取得了新的突破，并成为深度学习界的一股清流。

## 1.3 应用案例
* Object Detection: 如图形目标检测、文字定位等领域。
* Semantic Segmentation: 如街道场景分割、道路交通标志识别等领域。
* Natural Language Processing: 如文本情感分析、机器翻译等领域。
* GAN: 如人脸生成模型、图像超像素等领域。

# 2.基本概念术语说明
DenseNet 中的一些重要概念包括：

1. 稠密连接：在 DenseNet 中，每一个层都与后续所有层进行密集连接，因此称之为“稠密连接”。
2. 瓶颈层：在 DenseNet 中，深度为 k 的层被称为 k+1 块瓶颈层。
3. 分支结构：在 DenseNet 中，某些层存在分支结构。当该层没有残差连接时，分支结构可以起到辅助作用，提升模型的鲁棒性。
4. 残差连接：在 DenseNet 中，每一个残差块中的两个堆叠层之间均加入残差连接。残差连接采用的是“串联”的方式，即从前面的层传递的信息会串联到后面的层。
5. 混合精度训练：在 DenseNet 中，可以通过混合精度模式来加速训练过程，提升计算效率。

# 3.核心算法原理和具体操作步骤以及数学公式讲解
## 3.1 网络结构
DenseNet 使用密集连接的方法来构建网络，而非普通的分层连接。在卷积层之间的连接线路上，第一层有 n 个节点（feature map channel 数量），第二层有 m 个节点，第三层有 p 个节点，依此类推。然而，对于 Conv(l)(x)，它只与 Conv(l−1)(x) 和 x 有关，不能直接与任意层的输出相连，这就造成了 DenseNet 的稠密连接特性。如下图所示：

其中，特征图大小为 $H_{in}$×$W_{in}$ ，深度为 $L$ 。则全连接层的输入维度为 $(\sum_{i=1}^{L} H_i \times W_i \times C_{in}) \times m + C_{in}$, 输出维度为 $\lfloor (C_{in}+m)/m \rfloor$. 这里 $C_{in}$ 表示第一个卷积层的通道数。假设 $H_k=\frac{H_{in}}{2^{k}}$ ， $W_k=\frac{W_{in}}{2^{k}}$ 。其中，$\lfloor \cdot \rfloor$ 函数用于向下取整。根据 DenseNet 的设计思想，每一层的输入包含了之前所有层的输出信息，因此可以充分利用这些先验信息来产生有效的特征表示。
## 3.2 优化算法
DenseNet 使用了动量优化算法 Momentum 来更新网络参数。为了提升模型训练速度，DenseNet 把学习率初始化为较小的值（如 0.02），然后逐步增大到较大的学习率（如 0.1）。Momentum 算法在更新过程中使用了指数加权移动平均（EWMA）的滑动窗口，使得前期的梯度信号得到更多的保留。
## 3.3 混合精度训练
通过混合精度训练，可以提升模型的训练速度和资源占用率。当 GPU 的浮点运算能力较弱时，可以使用混合精度训练来提升模型的计算效率。混合精度训练使用半精度数据（FP16）来计算，可以降低内存和计算时间，同时保持模型精度，从而加速训练过程。使用混合精度训练时，一般需要在 optimizer 中设置参数 fp16_optimizer_params，用于设置半精度 optimizer。
```python
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
fp16_optimizer = apex.optimizers.FusedAdam(model.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=1e-08, weight_decay=weight_decay)

... # other training code

if args.use_fp16:
    model, optimizer = amp.initialize(model, optimizer, opt_level='O2')
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
else:
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.parallel.DataParallel(model)
```
# 4.具体代码实例和解释说明
## 4.1 PyTorch 实现
### 4.1.1 安装 Apex
如果使用混合精度训练，需要安装 Apex。Apex 是 NVIDIA 提供的一个用于混合精度训练的库。该项目目前支持 PyTorch 和 Tensorflow2.X。
```bash
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext"./
```
### 4.1.2 DenseNet 模块定义
本节定义了 DenseNet 模块。DenseNet 的主要结构由若干个 dense blocks 组成，每个 block 由多个卷积层、归一化层和激活函数组成。
```python
import torch
import torch.nn as nn


class BottleneckBlock(nn.Module):

    def __init__(self, in_channels, growth_rate, bn_size=4):
        super().__init__()

        inner_channels = bn_size * growth_rate // 2

        self.conv1 = nn.Conv2d(in_channels, inner_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(inner_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(inner_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(growth_rate)
        self.relu2 = nn.ReLU(inplace=True)


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        return out


class TransitionBlock(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.pool(out)
        out = self.bn(out)
        out = self.relu(out)

        return out


class DenseBlock(nn.ModuleList):

    def __init__(self, num_layers, in_channels, growth_rate, bn_size=4):
        super().__init__()

        for i in range(num_layers):
            layer = BottleneckBlock(in_channels + i * growth_rate, growth_rate, bn_size)
            self.append(layer)

    def forward(self, x):
        features = [x]

        for layer in self:
            new_features = layer(torch.cat(features, dim=1))
            features.append(new_features)

        return torch.cat(features, dim=1)



class DenseNet(nn.Module):

    def __init__(self, depth=26, growth_rate=12, reduction=0.5, bottleneck_width=4, num_classes=10):
        super().__init__()

        self.depth = depth
        self.growth_rate = growth_rate
        self.reduction = reduction
        self.bottleneck_width = bottleneck_width

        nblocks = (depth - 4) // 6
        if bottleneck_width == 4:
            nbasic_block = 1
        else:
            nbasic_block = bottleneck_width

        num_planes = 2 * growth_rate + 2

        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)
        self.dense1 = DenseBlock(nblocks[0], num_planes, growth_rate, bn_size=nbasic_block)
        self.trans1 = TransitionBlock(num_planes, int(math.floor(num_planes * reduction)))

        num_planes += int(math.floor(num_planes * reduction))
        self.dense2 = DenseBlock(nblocks[1], num_planes, growth_rate, bn_size=nbasic_block)
        self.trans2 = TransitionBlock(num_planes, int(math.floor(num_planes * reduction)))

        num_planes += int(math.floor(num_planes * reduction))
        self.dense3 = DenseBlock(nblocks[2], num_planes, growth_rate, bn_size=nbasic_block)
        self.trans3 = TransitionBlock(num_planes, int(math.floor(num_planes * reduction)))

        self.dense4 = DenseBlock(nblocks[3], num_planes, growth_rate, bn_size=nbasic_block)

        self.bn = nn.BatchNorm2d(int(math.floor(num_planes * reduction)))
        self.fc = nn.Linear(int(math.floor(num_planes * reduction)), num_classes)

        self._initialize_weights()

    def _initialize_weights(self):
        for name, param in self.named_parameters():
            if 'conv' in name and 'weight' in name:
                nn.init.kaiming_normal_(param)

            elif 'bn' in name and 'weight' in name:
                nn.init.constant_(param, 1)

            elif 'bn' in name and 'bias' in name:
                nn.init.constant_(param, 0)

            elif 'linear' in name and 'weight' in name:
                nn.init.kaiming_uniform_(param)

            elif 'linear' in name and 'bias' in name:
                nn.init.constant_(param, 0)

    def forward(self, x):
        out = self.conv1(x)

        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.trans3(self.dense3(out))

        out = self.dense4(out)

        out = F.avg_pool2d(F.relu(self.bn(out)), 4).view(out.size(0), -1)
        out = self.fc(out)

        return out
```
### 4.1.3 训练脚本
本节定义了 DenseNet 的训练脚本。主要包括：

1. 加载数据集
2. 创建模型对象
3. 设置优化器和损失函数
4. 配置学习率和训练轮数
5. 配置混合精度训练（可选）
6. 执行训练过程

```python
import argparse
import math
from datetime import datetime

import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data.distributed
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.tensorboard
import torchvision.models as models
import os
import sys
import apex.amp as amp


parser = argparse.ArgumentParser(description='PyTorch DenseNet Training', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', type=str, default='cifar10', help='dataset choice', choices=['cifar10'])
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size per process (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_false',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch N processes per node, which has N GPUs. This is the fastest way to use PyTorch for either single node or multi node data parallel training')
parser.add_argument('--fp16', action='store_true', help='Whether to use 16-bit float precision instead of 32-bit')
parser.add_argument('--opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision optimization level')
parser.add_argument('--loss-scale', type=float, default=1.,
                    help='Loss scaling factor. If positive, gradients are scaled by the loss scale value. Only used when fp16 set to True.\n'
                         '0 (default value): dynamic loss scaling.\n'
                         'Positive power of 2: static loss scaling value.\n')


best_acc1 = 0


def main():
    global best_acc1
    
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. This will turn on the CUDNN deterministic setting, which can slow down your training considerably! You may see unexpected behavior when restarting from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

        
def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:
        print("=> creating model '{}'".format(args.arch))
        model = DenseNet(depth=args.depth, growth_rate=args.growth_rate,
                         reduction=args.reduction, bottleneck_width=args.bottleneck_width, num_classes=args.num_classes)
        
    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
            
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.opt_level, loss_scale=args.loss_scale)

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()
        
    
    cudnn.benchmark = True

    # Data loading code
    normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447],
                                     std=[0.247, 0.243, 0.262])

    
    train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
                                    transforms.RandomCrop(32, padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    normalize,]))
    
    
    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        normalize,])),
        batch_size=128, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, args)
        return

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                 .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    tb_writer = SummaryWriter()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        acc1, acc5, loss = train(train_loader, model, criterion, optimizer, epoch, args, tb_writer)

        # evaluate on validation set
        acc1, acc5 = validate(val_loader, model, criterion, args, tb_writer)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed
                and args.rank % ngpus_per_node == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
               'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer' : optimizer.state_dict(),
            }, is_best, filename=f'ckpt_{epoch}.pth.tar')


    
def train(train_loader, model, criterion, optimizer, epoch, args, tb_writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader),
                              [batch_time, data_time, losses, top1, top5],
                              prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    # tensorboard log
    tb_writer.add_scalar('Train/Loss', losses.avg, epoch)
    tb_writer.add_scalar('Train/Top1 Acc', top1.avg, epoch)
    tb_writer.add_scalar('Train/Top5 Acc', top5.avg, epoch)

    return top1.avg, top5.avg, losses.avg
    
    
def validate(val_loader, model, criterion, args, tb_writer):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader),
                              [batch_time, losses, top1, top5],
                              prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
             .format(top1=top1, top5=top5))

    # tensorboard log
    tb_writer.add_scalar('Val/Loss', losses.avg, args.current_epoch)
    tb_writer.add_scalar('Val/Top1 Acc', top1.avg, args.current_epoch)
    tb_writer.add_scalar('Val/Top5 Acc', top5.avg, args.current_epoch)

    return top1.avg, top5.avg
    
    
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename,'model_best.pth.tar')