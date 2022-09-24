# GhostNet

## GhostNet描述

### 概述

GhostNet由华为诺亚方舟实验室在2020年提出，此网络提供了一个全新的Ghost模块，旨在通过廉价操作生成更多的特征图。基于一组原始的特征图，作者应用一系列线性变换，以很小的代价生成许多能从原始特征发掘所需信息的“幻影”特征图（Ghost feature maps）。该Ghost模块即插即用，通过堆叠Ghost模块得出Ghost bottleneck，进而搭建轻量级神经网络——GhostNet。该架构可以在同样精度下，速度和计算量均少于SOTA算法。

如下为MindSpore使用ImageNet2012数据集对GhostNet进行训练的示例。

### 论文

1. [论文](https://arxiv.org/pdf/1911.11907.pdf): Kai Han, Yunhe Wang, Qi Tian."GhostNet: More Features From Cheap Operations"

## 模型架构

GhostNet的总体网络架构如下：[链接](https://arxiv.org/pdf/1911.11907.pdf)

## 训练过程

```shell
python train.py --model ghostnet --data_url ./dataset/imagenet
```

```text
epoch: 1 step: 1251, loss is 5.001419
epoch time: 457012.100 ms, per step time: 365.317 ms
epoch: 2 step: 1251, loss is 4.275552
epoch time: 280175.784 ms, per step time: 223.961 ms
epoch: 3 step: 1251, loss is 4.0788813
epoch time: 280134.943 ms, per step time: 223.929 ms
epoch: 4 step: 1251, loss is 4.0310946
epoch time: 280161.342 ms, per step time: 223.950 ms
epoch: 5 step: 1251, loss is 3.7326777
epoch time: 280178.602 ms, per step time: 223.964 ms
...
```

## 评估过程

```shell
python validate.py --model ghostnet --data_url ./dataset/imagenet --checkpoint_path=[CHECKPOINT_PATH]
```

```text
{'top_5_accuracy': 0.9162371134020618, 'top_1_accuracy': 0.739368556701031}
```
