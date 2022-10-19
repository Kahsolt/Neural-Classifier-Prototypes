#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/10/18 

import gc
import os
from argparse import ArgumentParser

import torch
from torchattacks import PGD

from model import *
from data import *
from util import *

device = 'cpu'


def attack_transfer(args):
  ''' Dirs '''
  os.makedirs(args.data_path, exist_ok=True)

  ''' Model '''
  model = get_model(args.model).to(device)
  model.eval()

  transfer_model_names = [m for m in MODELS if m != args.model]
  transfer_models = [ ]
  for m in transfer_model_names:
    tm = get_model(m).to(device)
    tm.eval()
    transfer_models.append(tm) 
  
  print('[transfer attack]')
  print(f'   from : {args.model}')
  print(f'   to   : {transfer_model_names}')

  ''' Data '''
  dataloader = get_dataloader(args.atk_dataset, args.data_path, split='test', shuffle=True)

  x, y = iter(dataloader).next()
  X = x.to(device)                            # [B=1, C, H, W]
  with torch.no_grad():
    y_hat = model(X)                          # [B=1, N_CLASS]
    N_CLASSES = y_hat.shape[-1]               # N_CLASS
    assert N_CLASSES % args.batch_size == 0
  
  ''' Attack '''
  atk = PGD(model, eps=0.03, alpha=0.001, steps=40)
  atk.set_mode_targeted_by_function(lambda x, y: y)

  ''' Test '''
  n_samples = len(dataloader.dataset)
  for i, (X, Y) in enumerate(dataloader):
    with torch.no_grad():
      logits = model(normalize(X, dataset=args.atk_dataset))
      pred = logits.argmax(dim=-1).squeeze().item()

      print(f'[{i}/{n_samples}] original label truth: {Y.squeeze().item()}, pred {pred}')

    X_repeat = X.repeat([args.batch_size, 1, 1, 1])      # [B, C=3, H, W]
    for b in range(N_CLASSES // args.batch_size):
      cls_s = b * args.batch_size
      cls_e = (b + 1) * args.batch_size
      Y_tgt = torch.LongTensor([i for i in range(cls_s, cls_e)]).to(device)

      AX = atk(X_repeat, Y_tgt)

      with torch.no_grad():
        logits = model(normalize(AX, dataset=args.atk_dataset))
        X_pred = logits.argmax(dim=-1)

        T_pred_X = []
        for t_model in transfer_models:
          logits = t_model(normalize(X_repeat, dataset=args.atk_dataset))
          T_pred_X.append(logits.argmax(dim=-1))  # [B]
        T_pred_X = torch.stack(T_pred_X, dim=0)     # [K=model_cnt, B]

        T_pred_AX = []
        for t_model in transfer_models:
          logits = t_model(normalize(AX, dataset=args.atk_dataset))
          T_pred_AX.append(logits.argmax(dim=-1))  # [B]
        T_pred_AX = torch.stack(T_pred_AX, dim=0)     # [K=model_cnt, B]
      
      K = T_pred_AX.shape[0]
      for i in range(len(AX)):
        T_asr = (T_pred_AX[:, i] == Y_tgt[i].item()).sum()
        T_acr = (T_pred_AX[:, i] != T_pred_X[:, i]).sum()
        print(f'[{i}/{n_samples}] induce {pred} => {Y_tgt[i].item()}, X_pred: {X_pred[i].item()}, ' + 
              f'T_asr: {T_asr}/{K}={T_asr/K:.2%}, T_acr: {T_acr}/{K}={T_acr/K:.2%}')

      gc.collect()
      if device == 'cuda': torch.cuda.empty_cache()

    print('=' * 42)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-M', '--model', default='resnet18', choices=MODELS, help='victim model with pretrained weight')
  parser.add_argument('--mode', default='min', choices=['min', 'max'], help='find nearest point with min/max grad')
  parser.add_argument('--atk_dataset', default='imagenet-1k', choices=DATASETS, help='victim dataset')

  parser.add_argument('-B', '--batch_size', type=int, default=100, help='process n_attacks on one image simultaneously, must be divisible by model n_classes')
  parser.add_argument('--overwrite', action='store_true', help='force overwrite')
  parser.add_argument('--data_path', default='data', help='folder path to downloaded dataset')
  parser.add_argument('--log_path', default='log', help='folder path to local trained model weights and logs')
  parser.add_argument('--img_path', default='img', help='folder path to image display')
  args = parser.parse_args()

  print('[Ckpt] use pretrained weights from torchvision/torchhub')
  args.train_dataset = 'imagenet'     # NOTE: currently all `torchvision.models` are pretrained on `imagenet`

  attack_transfer(args)
