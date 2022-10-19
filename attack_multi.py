#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/10/18 

import gc
import os
import logging
from argparse import ArgumentParser

import torch

from model import *
from data import *
from util import *

cpu = 'cpu'
device = 'cpu'

#MODELS = [
#  'resnet18',
#  'resnet50',
#  'densenet121',
#  'inception_v3',
#  'efficientnet_v2_s',
#  'swin_t',
#]
STOP_RATIO = 0.7


log_dp = 'log'
os.makedirs(log_dp, exist_ok=True)


''' Logger '''
logger = logging.getLogger('ncp')
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)
handler = logging.FileHandler(os.path.join(log_dp, "attack_multi.log"))
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%m/%d/%Y %I:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)


def pgd_multi(models, images, labels, eps=0.03, alpha=0.001, steps=100, epoch=20, **kwargs):
  images = images.clone().detach().to(device)
  labels = labels.clone().detach().to(device)

  adv_images = images.clone().detach()
  adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
  adv_images = torch.clamp(adv_images, min=0, max=1).detach()

  normalizer = kwargs.get('normalizer', lambda _: _)
  for e in range(epoch):
    logger.info(f'[pgd_multi] epoch [{e}/{epoch}]')
    deltas = [ ]
    for k, model in enumerate(models):
      model.to(device)
      for i in range(steps):
        adv_images.requires_grad = True
        adv_images_norm = normalizer(adv_images)
        logits = model(adv_images_norm)

        pred = logits.argmax(dim=-1)
        mask_alienated = (pred == labels)      # [B=100]
        alienate_rate = mask_alienated.sum() / labels.numel()

        if alienate_rate > STOP_RATIO:
          logger.info(f'   [{i}/{steps}] attack on {MODELS[k]} aliente_rate: {alienate_rate:.3%}, early stop')
          break

        # Calculate targeted loss
        loss_each = F.cross_entropy(logits, labels, reduce=False)
        # Update adversarial images
        grad_each = torch.autograd.grad(loss_each, adv_images, grad_outputs=loss_each)[0]

        mask_expand = (~mask_alienated).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        adv_images = adv_images.detach() - alpha * grad_each.tanh() * mask_expand

        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        del grad_each, logits

        if i % 10 == 0:
          logger.info(f'   [{i}/{steps}] attack on {MODELS[k]} aliente_rate: {alienate_rate:.3%}, loss_avg: {loss_each.mean():.6}')

      deltas.append(delta)
      model.to(cpu)

      gc.collect()
      if device == 'cuda': torch.cuda.ipc_collect()

    mean_delta = torch.stack(deltas, axis=0).mean(axis=0)
    adv_images = torch.clamp(images + mean_delta, min=0, max=1).detach()
    logger.info(f'   Linf: {mean_delta.abs().max().item()}, L2: {(mean_delta.view(1, -1).norm(p=2, dim=-1) / mean_delta.numel()).item()}')

  return adv_images.to(cpu)


def attack_multi(args):
  ''' Dirs '''
  os.makedirs(args.data_path, exist_ok=True)

  ''' Model '''
  models = [ ]
  for m in MODELS:
    model = get_model(m)
    model.eval()
    models.append(model) 
  
  ''' Data '''
  dataloader = get_dataloader(args.atk_dataset, args.data_path, split='test', shuffle=True)

  X, Y = iter(dataloader).next()              # [B=1, C, H, W]
  with torch.no_grad():
    y_hat = model(X)                          # [B=1, N_CLASS]
    N_CLASSES = y_hat.shape[-1]               # N_CLASS
    assert N_CLASSES % args.batch_size == 0
  
  ''' Test '''
  n_samples = len(dataloader.dataset)
  for i, (X, Y) in enumerate(dataloader):
    with torch.no_grad():
      logits = model(normalize(X, dataset=args.atk_dataset))
      pred = logits.argmax(dim=-1).squeeze().item()

      logger.info(f'[{i}/{n_samples}] original label truth: {Y.squeeze().item()}, pred {pred}')

    X_repeat = X.repeat([args.batch_size, 1, 1, 1])      # [B, C=3, H, W]
    for b in range(N_CLASSES // args.batch_size):
      cls_s = b * args.batch_size
      cls_e = (b + 1) * args.batch_size
      Y_tgt = torch.LongTensor([i for i in range(cls_s, cls_e)])

      normalizer = lambda x: normalize(x, dataset=args.atk_dataset)
      AX = pgd_multi(models, X_repeat, Y_tgt, normalizer=normalizer)

      with torch.no_grad():
        pred_AX = []
        for model in models:
          logits = model(normalize(AX, dataset=args.atk_dataset))
          pred_AX.append(logits.argmax(dim=-1))     # [B]
        pred_AX = torch.stack(pred_AX, dim=0)     # [K=model_cnt, B]
      
      K = pred_AX.shape[0]
      for i in range(len(AX)):
        T_asr = (pred_AX[:, i] == Y_tgt[i].item()).sum()
        logger.info(f'[{i}/{n_samples}] induce {pred} => {Y_tgt[i].item()}, T_asr: {T_asr}/{K}={T_asr / K:.2%}')

    logger.info('=' * 42)


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

  attack_multi(args)
