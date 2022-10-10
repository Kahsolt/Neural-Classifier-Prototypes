#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/30 

import os
import json
from PIL import Image
from traceback import print_exc

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as D
import torchvision.transforms as T
import torchvision.transforms.functional as TF

DATASETS = [
  'mnist', 
  'svhn', 
  'cifar10', 
  'cifar100', 
  'tiny-imagenet', 
  'imagenet-1k'
]


class TinyImageNet(Dataset):

  def __init__(self, root: str, split='train', transform=T.ToTensor()):
    self.base_path = os.path.join(root, split)
    self.transform = transform

    with open(os.path.join(root, 'synsets.txt'), encoding='utf-8') as fh:
      class_names = fh.read().strip().split('\n')
      assert len(class_names) == 1000
    class_name_to_label = {cname: i for i, cname in enumerate(class_names)}

    metadata = [ ]
    for class_name in os.listdir(self.base_path):
      dp = os.path.join(self.base_path, class_name, 'images' if split=='train' else '')
      for fn in os.listdir(dp):
        fp = os.path.join(dp, fn)
        metadata.append((fp, class_name_to_label[class_name]))

    self.metadata = metadata

  def __len__(self):
    return len(self.metadata)

  def __getitem__(self, idx):
    fp, tgt = self.metadata[idx]
    img = Image.open(fp)
    img = img.convert('RGB')
    if self.transform is not None:
      img = self.transform(img)

    return img, tgt


class ImageNet_1k(Dataset):

  def __init__(self, root: str, transform=T.ToTensor()):
    self.base_path = os.path.join(root, 'val')
    self.transform = transform

    fns = [fn for fn in os.listdir(self.base_path)]
    fps = [os.path.join(self.base_path, fn) for fn in fns]
    with open(os.path.join(root, 'image_name_to_class_id_and_name.json'), encoding='utf-8') as fh:
      mapping = json.load(fh)
    tgts = [mapping[fn]['class_id'] for fn in fns]

    self.metadata = [x for x in zip(fps, tgts)]

  def __len__(self):
    return len(self.metadata)

  def __getitem__(self, idx):
    fp, tgt = self.metadata[idx]
    img = Image.open(fp)
    img = img.convert('RGB')
    if self.transform is not None:
      img = self.transform(img)

    return img, tgt


def chk_dataset_compatible(dataset1:str, dataset2:str):
  if dataset1 == dataset2: return True
  if {dataset1, dataset2} == {'imagenet-1k', 'imagenet'}: return True
  return False


def normalize(X: torch.Tensor, dataset:str='') -> torch.Tensor:
  ''' NOTE: to insure attack validity, normalization is delayed until put into model '''

  if chk_dataset_compatible(dataset, 'imagenet'):
    mean = (0.485, 0.456, 0.406)
    std  = (0.229, 0.224, 0.225)
    X = TF.normalize(X, mean, std)       # [B, C, H, W]

  return X


def get_dataloader(name, data_path, split='train', shuffle=False):
  transform_gray2rgb = T.Compose([
    T.ToTensor(),
    T.Lambda(lambda x: x.expand(3, 1, 1)),
  ])
  transform = T.Compose([
    T.ToTensor(),
  ])
  datasets = {
    # 28 * 28
    'mnist'        : lambda: D.MNIST   (root=data_path, split=split=='train', transform=transform_gray2rgb, download=True),
    # 32 * 32
    'svhn'         : lambda: D.SVHN    (root=data_path, split=split         , transform=transform, download=True),
    'cifar10'      : lambda: D.CIFAR10 (root=data_path, train=split=='train', transform=transform, download=True),
    'cifar100'     : lambda: D.CIFAR100(root=data_path, train=split=='train', transform=transform, download=True),
    # 64 * 64
    'tiny-imagenet': lambda: TinyImageNet(root=os.path.join(data_path, 'tiny-imagenet-200'), split='train' if split=='train' else 'val', transform=transform),
    # 224 * 224
    'imagenet-1k'  : lambda: ImageNet_1k(root=os.path.join(data_path, 'imagenet-1k'), transform=transform),
  }
  try:
    dataset = datasets[name]()
    return DataLoader(dataset, batch_size=1, shuffle=shuffle, drop_last=True, pin_memory=True, num_workers=0)
  except Exception as e:
    print_exc()
