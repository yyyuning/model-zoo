<!--- SPDX-License-Identifier: Apache-2.0 -->

# SqueezeNet

## Description
SqueezeNet is a small CNN which achieves AlexNet level accuracy on ImageNet with 50x fewer parameters. SqueezeNet requires less communication across servers during distributed training, less bandwidth to export a new model from the cloud to an autonomous car and more feasible to deploy on FPGAs and other hardware with limited memory.

## Model

|Model        |Download  |Download (with sample test data)| ONNX version |Opset version|Top-1 accuracy (%)|Top-5 accuracy (%)|
|-------------|:--------------|:--------------|:--------------|:--------------|:--------------|:--------------|
|SqueezeNet 1.0| [5 MB](model/squeezenet1.0-12.onnx)  |  [5 MB](model/squeezenet1.0-12.tar.gz) |  1.9 | 12|56.85|79.87|

>
> **Note** 
> The performance depends on the test hardware. Performance data here is collected with Intel® Xeon® Platinum 8280 Processor, 1s 4c per instance, CentOS Linux 8.3, data batch size is 1.

## Dataset
[ImageNet (ILSVRC2012)](http://www.image-net.org/challenges/LSVRC/2012/). 

## References
* **SqueezeNet1.0** Model from the papera [SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size](https://arxiv.org/abs/1602.07360)
* [MXNet](http://mxnet.incubator.apache.org), [Gluon model zoo](https://mxnet.incubator.apache.org/api/python/gluon/model_zoo.html), [GluonCV](https://gluon-cv.mxnet.io)

* [Intel® Neural Compressor](https://github.com/intel/neural-compressor)
* https://github.com/onnx/models

## Contributors
* [abhinavs95](https://github.com/abhinavs95) (Amazon AI)
* [ankkhedia](https://github.com/ankkhedia) (Amazon AI)
* [mengniwang95](https://github.com/mengniwang95) (Intel)
* [airMeng](https://github.com/airMeng) (Intel)
* [ftian1](https://github.com/ftian1) (Intel)
* [hshen14](https://github.com/hshen14) (Intel)

## License
Apache 2.0