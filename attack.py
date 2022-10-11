#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/27 

import os
import logging
from argparse import ArgumentParser

import torch
import torch.nn.functional as F

from alienater import Alienater, ATK_METHODS
from model import *
from data import *
from util import *


def attack(args):
  ''' Dirs '''
  log_dp = os.path.join(args.log_path, args.exp_name)
  os.makedirs(args.data_path, exist_ok=True)
  os.makedirs(log_dp, exist_ok=True)

  ''' Logger '''
  logger = logging.getLogger('ncp')
  logging.basicConfig(level=logging.INFO)
  logger.setLevel(logging.INFO)
  handler = logging.FileHandler(os.path.join(log_dp, "attack.log"))
  handler.setLevel(logging.INFO)
  formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%m/%d/%Y %I:%M:%S")
  handler.setFormatter(formatter)
  logger.addHandler(handler)

  ''' Data'''
  dataloader = get_dataloader(args.atk_dataset, args.data_path, split='test', shuffle=args.shuffle)

  ''' Model '''
  model = get_model(args.model).to(device)
  model.eval()

  ''' Info '''
  n_samples = args.limit or len(dataloader)

  logger.info(f'[Attack]')
  logger.info(f'   name          = {args.exp_name}')
  logger.info(f'   model         = {args.model}')
  logger.info(f'   train_dataset = {args.train_dataset}')
  logger.info(f'   atk_dataset   = {args.atk_dataset}')
  logger.info(f'     n_examples  = {n_samples}')
  logger.info(f'   atk_method    = {args.method}')
  logger.info(f'   atk_eps       = {args.eps}')
  logger.info(f'   atk_alpha     = {args.alpha}')
  logger.info(f'   atk_steps     = {args.steps}')

  ''' Attack '''
  normalizer = lambda X: normalize(X, args.atk_dataset)     # [B=1, C, H, W], delayed normalize to attackers
  atk = Alienater(model, method=args.method, eps=args.eps, alpha=args.alpha, steps=args.steps,
                  normalizer=normalizer, mode=args.mode)
  with torch.no_grad():
    X, _ = iter(dataloader).next()
    X = X.to(device)                          # [B=1, C, H, W]
    y_hat = model(X)                          # [B=1, N_CLASS]
    N_CLASSES = y_hat.shape[-1]               # N_CLASS
    assert N_CLASSES % args.batch_size == 0
  
  NXs = [ ]
  for i, (x, _) in enumerate(dataloader):
    if args.limit and i > args.limit: break

    X = x.to(device)
    logits = model(X)
    prob = F.softmax(logits, dim=-1).squeeze()
    x_pred = logits.argmax().cpu().item()

    print(f'[{i+1}/{n_samples}] pred: {x_pred}, prob: {prob[x_pred]:%}')

    X_repeat = X.repeat([args.batch_size, 1, 1, 1])      # [B, C=3, H, W]
    dxs, losses, grads, preds = [], [], [], []
    for b in range(N_CLASSES // args.batch_size):
      cls_s = b * args.batch_size
      cls_e = (b + 1) * args.batch_size
      print(f'>> alienating for class {cls_s} to {cls_e - 1}')
      Y_tgt = torch.LongTensor([i for i in range(cls_s, cls_e)])
      Y_tgt = Y_tgt.to(device)

      dx, loss, grad, pred = atk(X_repeat, Y_tgt)

      dxs   .append(dx  .detach().cpu())
      losses.append(loss.detach().cpu())
      grads .append(grad.detach().cpu())
      preds .append(pred.detach().cpu())

    NX = {
      'x':       x,
      'x_pred':  x_pred,
      'dx':      torch.cat(dxs,    axis=0),
      'nx_pred': torch.cat(preds,  axis=0),
      'loss':    torch.cat(losses, axis=0),
      'grad':    torch.cat(grads,  axis=0),
    }
    analyze_NX(NX)
    NXs.append(NX)
  
  torch.save(NXs, args.save_fp)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-M', '--model', default='resnet18', choices=MODELS, help='victim model with pretrained weight')
  parser.add_argument('-D', '--atk_dataset', default='imagenet-1k', choices=DATASETS, help='victim dataset')
  parser.add_argument('--mode', default='min', choices=['min', 'max'], help='find nearest point with min/max grad')
  parser.add_argument('--shuffle', action='store_true', help='shuffle dataset')
  parser.add_argument('--limit', default=16, help='limit n_sameples from dataset')

  parser.add_argument('--method', default='pgd', choices=ATK_METHODS, help='base attack method')
  parser.add_argument('--eps', type=float, default=0.03, help='total pertubation limit')
  parser.add_argument('--alpha', type=float, default=0.001, help='stepwise pertubation limit ~= learning rate')
  parser.add_argument('--steps', type=int, default=100, help='n_iters on one single picture towards a single target')

  parser.add_argument('-B', '--batch_size', type=int, default=100, help='process n_attacks on one picture simultaneously, must be divisible by model n_classes')
  parser.add_argument('--overwrite', action='store_true', help='force overwrite')
  parser.add_argument('--data_path', default='data', help='folder path to downloaded dataset')
  parser.add_argument('--log_path', default='log', help='folder path to local trained model weights and logs')
  args = parser.parse_args()

  if args.eps <= 0.0:
    raise ValueError('--eps should > 0')
  if args.alpha > args.eps:
    raise ValueError('--alpha should be smaller than --eps')

  print('[Ckpt] use pretrained weights from torchvision/torchhub')
  args.train_dataset = 'imagenet'     # NOTE: currently all `torchvision.models` are pretrained on `imagenet`
  args.exp_name = exp_name(args.model, args.train_dataset, args.mode, args.atk_dataset, args.method, args.eps, args.alpha)
  args.save_fp = os.path.join(args.log_path, args.exp_name + '.pkl')
  if os.path.exists(args.save_fp) and not args.overwrite:
    print('safely ignore due to exists:', args.save_fp)
  else:
    attack(args)
