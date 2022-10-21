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
  os.makedirs(args.data_path, exist_ok=True)
  os.makedirs(args.img_path, exist_ok=True)
  os.makedirs(args.log_path, exist_ok=True)

  ''' Logger '''
  logger = logging.getLogger('ncp')
  logging.basicConfig(level=logging.INFO)
  logger.setLevel(logging.INFO)
  handler = logging.FileHandler(os.path.join(args.log_path, f"{args.exp_name}.log"))
  handler.setLevel(logging.INFO)
  formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%m/%d/%Y %I:%M:%S")
  handler.setFormatter(formatter)
  logger.addHandler(handler)

  ''' Model '''
  model = get_model(args.model).to(device)
  model.eval()

  ''' Info '''
  logger.info(f'[Attack]')
  logger.info(f'   name          = {args.exp_name}')
  logger.info(f'   model         = {args.model}')
  logger.info(f'   train_dataset = {args.train_dataset}')
  logger.info(f'   mode_cr       = {args.mode_cr}')
  logger.info(f'   atk_method    = {args.method}')
  logger.info(f'   atk_eps       = {args.eps}')
  logger.info(f'   atk_alpha     = {args.alpha}')
  logger.info(f'   atk_steps     = {args.steps}')

  ''' Attack '''
  atk = Alienater(model, method=args.method, eps=args.eps, alpha=args.alpha, steps=args.steps, mode=args.mode)
  
  ''' Data '''
  X_shape = torch.Size([1, 3, args.height, args.width])
  if args.mode_cr == 'const':
    x = torch.ones(X_shape)
    for c in range(3):
      x[:, c, :, :] = x[:, c, :, :] * args.const[c]
    x = x / 255.0
  elif args.mode_cr == 'rand':
    if args.rand == 'uniform':
      x = torch.empty(X_shape).uniform_(args.low, args.high)
    elif args.rand == 'normal':
      x = torch.empty(X_shape).normal_(args.mu, args.sigma)

  X = x.to(device)                            # [B=1, C, H, W]
  with torch.no_grad():
    y_hat = model(X)                          # [B=1, N_CLASS]
    N_CLASSES = y_hat.shape[-1]               # N_CLASS
    assert N_CLASSES % args.batch_size == 0
  
  ''' Evolve '''
  logits = model(X)
  prob = F.softmax(logits, dim=-1).squeeze()
  x_pred = logits.argmax().cpu().item()

  logger.info(f'>> pred: {x_pred}, prob: {prob[x_pred]:%}')

  X_expand = X.expand(args.batch_size, -1, -1, -1)      # [B, C=3, H, W]
  dxs, losses, grads, preds = [], [], [], []
  for b in range(N_CLASSES // args.batch_size):
    cls_s = b * args.batch_size
    cls_e = (b + 1) * args.batch_size
    logger.info(f'>> alienating for class {cls_s} to {cls_e - 1}')
    Y_tgt = torch.LongTensor([i for i in range(cls_s, cls_e)])
    Y_tgt = Y_tgt.to(device)

    dx, loss, grad, pred = atk(X_expand, Y_tgt)

    dxs   .append(dx  .detach().cpu())
    losses.append(loss.detach().cpu())
    grads .append(grad.detach().cpu())
    preds .append(pred.detach().cpu())

    fp = os.path.join(args.img_path, f'{args.exp_name};cls={cls_s}~{cls_e - 1}.png')
    save_images(dx, fp)

  NX = {
    'x':       x,                           # [1, C, H, W]
    'x_pred':  x_pred,                      # []
    'dx':      torch.cat(dxs,    axis=0),   # [N_CLASS, C, H, W]
    'nx_pred': torch.cat(preds,  axis=0),   # [N_CLASS, C, H, W]
    'loss':    torch.cat(losses, axis=0),   # [N_CLASS, C, H, W]
    'grad':    torch.cat(grads,  axis=0),   # [N_CLASS, C, H, W]
  }
  analyze_NX(NX)

  torch.save([NX], args.save_fp)


if __name__ == '__main__':
  RANDOMS = ['uniform', 'normal']

  parser = ArgumentParser()
  parser.add_argument('-M', '--model', default='resnet18', choices=MODELS, help='victim model with pretrained weight')
  parser.add_argument('--mode', default='min', choices=['min', 'max'], help='find nearest point with min/max grad')

  parser.add_argument('-C', '--const', default=0, type=int, help='const single pixel or comma separated RGB values like 11,45,14 (values in 0 ~ 255)')
  parser.add_argument('-R', '--rand', default=None, choices=RANDOMS, help='victim model with pretrained weight')
  parser.add_argument('--low',   default=0.0, type=float, help='rand param for uniform')
  parser.add_argument('--high',  default=1.0, type=float, help='rand param for uniform')
  parser.add_argument('--mu',    default=0.5, type=float, help='rand param for normal')
  parser.add_argument('--sigma', default=0.1, type=float, help='rand param for normal')
  parser.add_argument('-H', '--height', default=224, type=int, help='image height')
  parser.add_argument('-W', '--width',  default=224, type=int, help='image width')

  parser.add_argument('--method', default='pgd', choices=ATK_METHODS, help='base attack method')
  parser.add_argument('--eps', type=float, default=0.1, help='total pertubation limit')
  parser.add_argument('--alpha', type=float, default=0.001, help='stepwise pertubation limit ~= learning rate')
  parser.add_argument('--steps', type=int, default=10000, help='n_iters on one single image towards a single target')

  parser.add_argument('-B', '--batch_size', type=int, default=100, help='process n_attacks on one image simultaneously, must be divisible by model n_classes')
  parser.add_argument('--overwrite', action='store_true', help='force overwrite')
  parser.add_argument('--data_path', default='data', help='folder path to downloaded dataset')
  parser.add_argument('--log_path', default='log', help='folder path to local trained model weights and logs')
  parser.add_argument('--img_path', default='img', help='folder path to image display')
  args = parser.parse_args()

  if args.eps <= 0.0:
    raise ValueError('--eps should > 0')
  if args.alpha > args.eps:
    raise ValueError('--alpha should be smaller than --eps')
  
  print('[Ckpt] use pretrained weights from torchvision/torchhub')
  args.train_dataset = 'imagenet'     # NOTE: currently all `torchvision.models` are pretrained on `imagenet`

  # let rand override const
  args.mode_cr = 'const'
  if isinstance(args.const, int):
    args.const = [args.const, args.const, args.const]
  else:
    args.const = [int(x) for x in args.const.split(',')]
  assert 0 <= min(args.const) and max(args.const) <= 255
  args.exp_name = exp_cr_name(args.model, args.train_dataset, args.mode, args.method, const=args.const)

  if args.rand is not None:
    args.mode_cr = 'rand'
    if args.rand == 'uniform':
      args.exp_name = exp_cr_name(args.model, args.train_dataset, args.mode, args.method, rand=args.rand, low=args.low, high=args.high)
    elif args.rand == 'normal':
      args.exp_name = exp_cr_name(args.model, args.train_dataset, args.mode, args.method, rand=args.rand, mu=args.mu, sigma=args.sigma)
    else: raise ValueError

  args.save_fp = os.path.join(args.log_path, args.exp_name + '.pkl')
  if os.path.exists(args.save_fp) and not args.overwrite:
    print('safely ignore due to exists:', args.save_fp)
  else:
    attack(args)
