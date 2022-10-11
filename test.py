#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/27 

import os
from argparse import ArgumentParser

from model import *
from data import *
from util import *


def test(args):
  ''' Model '''
  model = get_model(args.model).to(device)
  model.eval()

  ''' Data '''
  dataloader = get_dataloader(args.atk_dataset, args.data_path, split=args.split)
  
  # Test Clean
  if not args.ncp:
    if chk_dataset_compatible(args.train_dataset, args.atk_dataset):
      acc = test_acc(model, dataloader)
      print(f'   acc(clean): {acc:.3%}')
    else:
      print('atk_dataset is not compatible with train_dataset')

  # Test Attack
  if args.ncp:
    NX = torch.load(args.ncp)[0]
    if args.sel == 'dx':            # 只取差分增量，[N_CLASS, C, H, W]
      ncps = NX['dx']               # 扰动强度是被attacker保证的 (-eps, eps)
    elif args.sel == 'nx':          # 取类原型，放缩强度到eps
      ncps = NX['x'] + NX['dx']
      ncps_n = ncps - ncps.mean([1, 2, 3], keepdim=True)
      mag = max(ncps_n.max(), -ncps_n.min())
      ncps = ncps_n * (args.eps / mag)
    else: raise ValueError
    ncps = ncps.to(device)
    del NX

    n_classes = len(ncps)
    for i, ncp in enumerate(ncps):   # 使用每个类的特征原型去攻击整个数据集
      print(f'[{i}/{n_classes}] induce whole dataset to class {i}')

      # Try testing attack success rate
      if True:
        target = torch.IntTensor([i]).to(device)
        asr = test_asr(model, dataloader, target, ncp, resizer=args.resizer)
        print(f'   asr: {asr:.3%}')
      
      # Try testing remnet accuracy (:= 1 - misclf rate) after adding ncp
      if chk_dataset_compatible(args.train_dataset, args.atk_dataset):
        acc = test_acc(model, dataloader, ncp, resizer=args.resizer)
        print(f'   acc: {acc:.3%}')

      # Try testing predction changing rate after adding ncp
      if True:
        pcr = test_pcr(model, dataloader, ncp, resizer=args.resizer)
        print(f'   pcr: {pcr:.3%}')


def test_asr(model, dataloader, target, ncp=None, resizer='tile') -> float:
  ''' Attack Success Rate '''

  if ncp is not None:
    X, _ = iter(dataloader).next()
    DX = ncp_expand_batch(ncp, X.shape, resizer=resizer)
    target = target.repeat(X.shape[0])

  total, attacked = 0, 0
  with torch.no_grad():
    for X, _ in dataloader:
      X = X.to(device)
      if ncp is not None:
        X = (X + DX).clip(0.0, 1.0)
      X = normalize(X, args.atk_dataset)

      pred = model(X).argmax(dim=-1)

      total += len(pred)
      attacked += (pred == target).sum()
  
  return (attacked / total).item()


def test_acc(model, dataloader, ncp=None, resizer='tile') -> float:
  ''' Accuracy '''

  if ncp is not None:
    X, _ = iter(dataloader).next()
    DX = ncp_expand_batch(ncp, X.shape, resizer=resizer)

  total, correct = 0, 0
  with torch.no_grad():
    for X, Y in dataloader:
      X, Y = X.to(device), Y.to(device)
      if ncp is not None:
        X = (X + DX).clip(0.0, 1.0)
      X = normalize(X, args.atk_dataset)

      pred = model(X).argmax(dim=-1)

      total += len(pred)
      correct += (pred == Y).sum()
  
  return (correct / total).item()


def test_pcr(model, dataloader, ncp, resizer='tile') -> float:
  ''' Prediction Change Rate '''

  if True:
    X, _ = iter(dataloader).next()
    DX = ncp_expand_batch(ncp, X.shape, resizer=resizer)
  
  total, changed = 0, 0
  with torch.no_grad():
    for X, _ in dataloader:
      X = X.to(device)
      AX = (X + DX).clip(0.0, 1.0)
      X  = normalize(X,  args.atk_dataset)
      AX = normalize(AX, args.atk_dataset)

      pred1 = model(X ).argmax(dim=-1)
      pred2 = model(AX).argmax(dim=-1)

      total += len(pred1)
      changed += (pred1 != pred2).sum()
  
  return (changed / total).item()


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-M', '--model', default='resnet18', choices=MODELS, help='model to attack')
  parser.add_argument('-D', '--atk_dataset', default='imagenet-1k', choices=DATASETS, help='dataset to attack (atk_dataset can be different from train_dataset')
  parser.add_argument('--split', default='test', choices=['test', 'train'], help='split name for atk_dataset')
  parser.add_argument('--sel', default='dx', choices=['nx', 'dx'], help='use cls-prototype(nx) or perturbation(dx)')
  parser.add_argument('--eps', default=0.1, type=float, help='eps for nx')
  
  parser.add_argument('--ncp', help='path to ncp.npy file')
  parser.add_argument('--resizer', default='tile', choices=['tile', 'interpolate'], help='resize ncp when shape mismatch')
  
  parser.add_argument('-B', '--batch_size', type=int, default=100)
  parser.add_argument('--data_path', default='data', help='folder path to downloaded dataset')
  parser.add_argument('--log_path', default='log', help='folder path to local trained model weights and logs')
  args = parser.parse_args()

  print(f'>> testing on dataset "{args.atk_dataset}" of split "{args.split}"')
  if args.ncp:
    print(f'>> using ncp "{os.path.basename(args.ncp)}"')
    print(f'>> using resizer "{args.resizer}" in case of shape mismatch')
    print(f'>> using sel "{args.sel}"')
    print(f'>> using eps "{args.eps}"')

  print('[Ckpt] use pretrained weights from torchvision/torchhub')
  args.train_dataset = 'imagenet'   # NOTE: currently all models in MODELS are pretrained on `imagenet`

  test(args)
