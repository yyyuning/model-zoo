<!--- SPDX-License-Identifier: Apache-2.0 -->

# ResNet50-vd

## Description

ResNet 系列模型是在 2015 年提出的，一举在 ILSVRC2015 比赛中取得冠军，top5 错误率为 3.57%。该网络创新性的提出了残差结构，通过堆叠多个残差结构从而构建了 ResNet 网络。实验表明使用残差块可以有效地提升收敛速度和精度。

斯坦福大学的 Joyce Xu 将 ResNet 称为「真正重新定义了我们看待神经网络的方式」的三大架构之一。由于 ResNet 卓越的性能，越来越多的来自学术界和工业界学者和工程师对其结构进行了改进，比较出名的有 Wide-ResNet, ResNet-vc,ResNet-vd, Res2Net 等，其中 ResNet-vc 与 ResNet-vd 的参数量和计算量与 ResNet 几乎一致，所以在此我们将其与 ResNet 统一归为 ResNet 系列。

本次发布 ResNet 系列的模型包括 ResNet50，ResNet50_vd，ResNet50_vd_ssld，ResNet200_vd 等 14 个预训练模型。在训练层面上，ResNet 的模型采用了训练 ImageNet 的标准训练流程，而其余改进版模型采用了更多的训练策略，如 learning rate 的下降方式采用了 cosine decay，引入了 label smoothing 的标签正则方式，在数据预处理加入了 mixup 的操作，迭代总轮数从 120 个 epoch 增加到 200 个 epoch。

其中，ResNet50_vd_v2 与 ResNet50_vd_ssld 采用了知识蒸馏，保证模型结构不变的情况下，进一步提升了模型的精度，具体地，ResNet50_vd_v2 的 teacher 模型是 ResNet152_vd（top1 准确率 80.59%），数据选用的是 ImageNet-1k 的训练集，ResNet50_vd_ssld 的 teacher 模型是 ResNeXt101_32x16d_wsl（top1 准确率 84.2%），数据选用结合了 ImageNet-1k 的训练集和 ImageNet-22k 挖掘的 400 万数据。知识蒸馏的具体方法正在持续更新中。

## Model

|Model          |Download                       |ONNX version   |Opset version  |Top-1 accuracy (%) |Top-5 accuracy (%) |
|---------------|:------------------------------|:--------------|:--------------|:------------------|:------------------|
|ResNet50-vd    |[98 MB](https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz)|v8             |7              |79.12              |94.44              |

## Dataset

[ImageNet (ILSVRC2015)](http://www.image-net.org/challenges/LSVRC/2015/).

## References

* **ResNetvd**
* [model zoo](https://github.com/PaddlePaddle/PaddleClas/blob/release/2.4/docs/zh_CN/models/ResNet_and_vd.md)
* [onnx/models](https://bj.bcebos.com/paddlehub/fastdeploy/ResNet50_vd_infer.tgz)

## License

Apache 2.0
