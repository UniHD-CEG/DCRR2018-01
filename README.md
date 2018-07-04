# ECML-PKDD 2018 Reproducible Research
Source code for reproducibility of our experiments for the ECML-PKDD 2018 paper "Towards Efficient Forward Propagation on Resource-Constrained Systems"

## Dependencies

* Python 2 or 3
* TensorFlow
* TensorPack
* Python bindings for OpenCV

## Usage
Configuration: ternary weights and 8-bit activations
ConvNet on SVHN

```shell
python svhn-digit.py --gpu 0
```

ResNet on Cifar10

```shell
python cifar10-resnet.py --n 3 --gpu 0,1
```

AlexNet on ImageNet

```shell
python imagenet-alexnet.py --data PATH --gpu 0,1
```
