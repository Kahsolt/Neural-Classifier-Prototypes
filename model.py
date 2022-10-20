#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/30 

import torchvision.models as M

MODELS = [
  'alexnet', 

#  'vgg11',
#  'vgg13',
#  'vgg16',
#  'vgg19',
#  'vgg11_bn',
#  'vgg13_bn',
#  'vgg16_bn',
  'vgg19_bn',

  'convnext_tiny',
#  'convnext_small',
#  'convnext_base',
#  'convnext_large',
  
  'densenet121',
#  'densenet161',
#  'densenet169',
#  'densenet201',

#  'efficientnet_b0',
#  'efficientnet_b1',
#  'efficientnet_b2',
#  'efficientnet_b3',
#  'efficientnet_b4',
#  'efficientnet_b5',
#  'efficientnet_b6',
#  'efficientnet_b7',
  'efficientnet_v2_s',
#  'efficientnet_v2_m',
#  'efficientnet_v2_l',

#  'googlenet',

  'inception_v3',

#  'mnasnet0_5',
#  'mnasnet0_75',
#  'mnasnet1_0',
  'mnasnet1_3',

#  'mobilenet_v2',
#  'mobilenet_v3_small',
  'mobilenet_v3_large',

  'regnet_y_400mf',
#  'regnet_y_800mf',
#  'regnet_y_1_6gf',
#  'regnet_y_3_2gf',
#  'regnet_y_8gf',
#  'regnet_y_16gf',
#  'regnet_y_32gf',
#  'regnet_y_128gf',
#  'regnet_x_400mf',
#  'regnet_x_800mf',
#  'regnet_x_1_6gf',
#  'regnet_x_3_2gf',
#  'regnet_x_8gf',
#  'regnet_x_16gf',
#  'regnet_x_32gf',

  'resnet18',
  'resnet34',
  'resnet50',
#  'resnet101',
#  'resnet152',
  'resnext50_32x4d',
#  'resnext101_32x8d',
#  'resnext101_64x4d',
  'wide_resnet50_2',
#  'wide_resnet101_2',

#  'shufflenet_v2_x0_5',
#  'shufflenet_v2_x1_0',
  'shufflenet_v2_x1_5',
#  'shufflenet_v2_x2_0',

#  'squeezenet1_0',
  'squeezenet1_1',

  'vit_b_16',
#  'vit_b_32',
#  'vit_l_16',
#  'vit_l_32',
#  'vit_h_14',

  'swin_t',
#  'swin_s',
#  'swin_b',
]


def get_model(name):
  if hasattr(M, name):
    if   name == 'swin_t': model = M.swin_t(weights=M.Swin_T_Weights.IMAGENET1K_V1)
    elif name == 'swin_s': model = M.swin_s(weights=M.Swin_S_Weights.IMAGENET1K_V1)
    elif name == 'swin_b': model = M.swin_b(weights=M.Swin_B_Weights.IMAGENET1K_V1)
    else:
      model = getattr(M, name)(pretrained=True)
  else:
    raise ValueError(f'[get_model] unknown model {name}')
  return model


def has_model(name):
  return hasattr(M, name)
