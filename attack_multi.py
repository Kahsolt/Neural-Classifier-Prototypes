#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/10/18 

import os
import gc
import logging
import psutil
import warnings
from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F

from model import get_model
from data import get_dataloader, normalize, DATASETS


MODELS = [
  'alexnet', 

#  'vgg11',
#  'vgg13',
#  'vgg16',
  'vgg19',
#  'vgg11_bn',
#  'vgg13_bn',
#  'vgg16_bn',
  'vgg19_bn',

#  'convnext_tiny',
#  'convnext_small',
  'convnext_base',
#  'convnext_large',
  
  'densenet121',
#  'densenet161',
#  'densenet169',
  'densenet201',

#  'efficientnet_b0',
#  'efficientnet_b1',
#  'efficientnet_b2',
#  'efficientnet_b3',
#  'efficientnet_b4',
#  'efficientnet_b5',
#  'efficientnet_b6',
#  'efficientnet_b7',
#  'efficientnet_v2_s',
#  'efficientnet_v2_m',   # request image resize
#  'efficientnet_v2_l',

#  'googlenet',           # request image resize

#  'inception_v3',        # request image resize

#  'mnasnet0_5',
#  'mnasnet0_75',
#  'mnasnet1_0',
  'mnasnet1_3',

#  'mobilenet_v2',
#  'mobilenet_v3_small',
  'mobilenet_v3_large',

#  'regnet_y_400mf',
#  'regnet_y_800mf',
  'regnet_y_1_6gf',
#  'regnet_y_3_2gf',
#  'regnet_y_8gf',
#  'regnet_y_16gf',
#  'regnet_y_32gf',
#  'regnet_y_128gf',

#  'regnet_x_400mf',
#  'regnet_x_800mf',
  'regnet_x_1_6gf',
#  'regnet_x_3_2gf',
#  'regnet_x_8gf',
#  'regnet_x_16gf',
#  'regnet_x_32gf',

  'resnet18',
#  'resnet34',
  'resnet50',
  'resnet101',
#  'resnet152',
  'resnext50_32x4d',
#  'resnext101_32x8d',
#  'resnext101_64x4d',
  'wide_resnet50_2',
#  'wide_resnet101_2',

#  'shufflenet_v2_x0_5',
#  'shufflenet_v2_x1_0',
#  'shufflenet_v2_x1_5',
  'shufflenet_v2_x2_0',

#  'squeezenet1_0',
  'squeezenet1_1',

  'vit_b_16',
#  'vit_b_32',
  'vit_l_16',
#  'vit_l_32',
#  'vit_h_14',

#  'swin_t',
#  'swin_s',
  'swin_b',
]

''' Mem tracker '''
device = 'cuda'      # NOTE: use cuda will cause memory leak until broken, don't know why :(
proc = psutil.Process(os.getpid())
def show_mem_stats(where:str=''):
  stats = proc.memory_full_info()

  print(f'[Mem] {where}')
  print(f'   ' + 
        (f'uss={stats.uss//2**20}MB, '                     if hasattr(stats, 'uss') else '') + 
        (f'rss={stats.rss//2**20}MB, '                     if hasattr(stats, 'rss') else '') + 
        (f'vms={stats.vms//2**20}MB, '                     if hasattr(stats, 'vms') else '') + 
        (f'peak_wset={stats.peak_wset//2**20}MB, '         if hasattr(stats, 'peak_wset') else '') + 
        (f'n_page_faults={stats.num_page_faults // 1000}k' if hasattr(stats, 'num_page_faults') else ''))
  if device == 'cuda':
    print(f'   ' + 
          f'vram_alloc={torch.cuda.memory_allocated(device=device)//2**20}MB, ' + 
          f'vram_cache={torch.cuda.memory_reserved(device=device)//2**20}MB')


def gc_all():
  if device == 'cuda':
    for _ in range(5):
      torch.cuda.empty_cache()
      torch.cuda.ipc_collect()
  print('gc.collect:', gc.collect())


''' Gloabls '''
AVAILABLE_MODELS = [ ]
for name in MODELS:
  try:
    m = get_model(name) ; del m
    AVAILABLE_MODELS.append(name)
  except Exception as e:
    print(f'ignore model {name} due to {e}')
gc_all()

N_CLASSES = 1000
STOP_RATE = (len(AVAILABLE_MODELS) - 1) / len(AVAILABLE_MODELS)
print('final STOP_RATE:', STOP_RATE)

log_dp = 'log'
os.makedirs(log_dp, exist_ok=True)


''' Logger '''
logger = logging.getLogger('ncp_multi')
logging.basicConfig(level=logging.INFO)
logger.setLevel(logging.INFO)
handler = logging.FileHandler(os.path.join(log_dp, "attack_multi.log"))
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%m/%d/%Y %I:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)


def pgd_multi(images, labels, args, **kwargs):
  eps   = args.eps
  alpha = args.alpha
  steps = args.steps
  epoch = args.epoch

  with torch.no_grad():
    images = images.to(device)
    labels = labels.to(device)

    adv_images = images.clone().detach()
    adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
    adv_images = torch.clamp(adv_images, min=0, max=1)

  normalizer = kwargs.get('normalizer', lambda _: _)
  for e in range(epoch):
    logger.info(f'[pgd_multi] epoch [{e}/{epoch}]')
    deltas = [ ]
    for k, model_name in enumerate(AVAILABLE_MODELS):
      model = get_model(model_name).to(device)

      alienate_rates = []
      for i in range(steps):
        adv_images.requires_grad = True
        adv_images_norm = normalizer(adv_images)
        logits = model(adv_images_norm)

        pred = logits.argmax(dim=-1)
        mask_alienated = (pred == labels)      # [B=100]
        alienate_rate = mask_alienated.sum() / labels.numel()

        stop_ratio = (50 / (epoch - 1) * e + 50) / 100
        if alienate_rate > stop_ratio:
          logger.info(f'   [{i}/{steps}] attack on {MODELS[k]} aliente_rate: {alienate_rate:.3%}, early stop')
          break

        # Calculate targeted loss
        loss_each = F.cross_entropy(logits, labels, reduce=False)
        with torch.no_grad():
          loss_mean = loss_each.mean().item()
        grad_each = torch.autograd.grad(loss_each, adv_images, grad_outputs=loss_each)[0]
        # Force clear grads and loss, otherwise the VRAM will booom!!
        model.zero_grad() ; del loss_each
        
        # Update adversarial images
        with torch.no_grad():
          mask_expand = (~mask_alienated).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
          adv_images = adv_images - alpha * grad_each.tanh() * mask_expand

          delta = torch.clamp(adv_images - images, min=-eps, max=eps)
          adv_images = torch.clamp(images + delta, min=0, max=1)

        if i % 10 == 0:
          logger.info(f'   [{i}/{steps}] attack on {MODELS[k]} aliente_rate: {alienate_rate:.3%}, loss_avg: {loss_mean:.6}')
          
      alienate_rates.append(alienate_rate)
      deltas.append(delta.to('cpu'))

      del model ; gc_all()
      show_mem_stats()

    with torch.no_grad():
      images = images.to('cpu')
      mean_delta = torch.stack(deltas, axis=0).mean(axis=0)
      adv_images = torch.clamp(images + mean_delta, min=0, max=1)
      logger.info(f'   Linf: {mean_delta.abs().max().item()}, L2: {(mean_delta.view(1, -1).norm(p=2, dim=-1) / mean_delta.numel()).item()}')

      mean_alienate_rate = sum(alienate_rates) / len(alienate_rates)
      if mean_alienate_rate > STOP_RATE:
        logger.info(f'   >> mean_alienate_rate = {mean_alienate_rate}, early stop')
        break

      del deltas, mean_delta, alienate_rates

  with torch.no_grad():
    adv_images.to(device)
    pred_AX = []
    for model_name in AVAILABLE_MODELS:
      model = get_model(model_name).to(device)

      logits = model(normalizer(adv_images))
      pred_AX.append(logits.argmax(dim=-1))     # [B]

      del model ; gc_all()
    pred_AX = torch.stack(pred_AX, dim=0)     # [K=model_cnt, B]

  del images, labels, adv_images ; gc_all()

  return pred_AX


def attack_multi(args):
  ''' Dirs '''
  os.makedirs(args.data_path, exist_ok=True)
  
  ''' Data '''
  dataloader = get_dataloader(args.atk_dataset, args.data_path, split='test', shuffle=True)
  show_mem_stats('after dataloader')

  ''' Test '''
  N = len(dataloader.dataset)
  for X, Y in dataloader:
    X = X.to(device)
    X_expand = X.expand(args.batch_size, -1, -1, -1)      # [B, C=3, H, W]
    for b in range(N_CLASSES // args.batch_size):
      cls_s = b * args.batch_size
      cls_e = (b + 1) * args.batch_size
      Y_tgt = torch.LongTensor([i for i in range(cls_s, cls_e)]).to(device)

      normalizer = lambda x: normalize(x, dataset=args.atk_dataset)
      pred_AX = pgd_multi(X_expand, Y_tgt, args, normalizer=normalizer)

      K = len(AVAILABLE_MODELS)
      for i in range(len(pred_AX)):
        T_asr = (pred_AX[:, i] == Y_tgt[i].item()).sum()
        logger.info(f'[{i}/{N}] induce {Y.squeeze().item()} => {Y_tgt[i].item()}, T_asr: {T_asr}/{K}={T_asr / K:.2%}')

      del Y_tgt, pred_AX ; gc_all()

    del X, Y, X_expand ; gc_all()

    logger.info('=' * 42)


if __name__ == '__main__':
  parser = ArgumentParser()
  parser.add_argument('-M', '--model', default='resnet18', choices=MODELS, help='victim model with pretrained weight')
  parser.add_argument('--atk_dataset', default='imagenet-1k', choices=DATASETS, help='victim dataset')

  parser.add_argument('--eps',   default=0.03,  type=float)
  parser.add_argument('--alpha', default=0.001, type=float)
  parser.add_argument('--steps', default=100,   type=int)
  parser.add_argument('--epoch', default=20,    type=int)

  parser.add_argument('-B', '--batch_size', type=int, default=20, help='process n_attacks on one image simultaneously, must be divisible by model n_classes')
  parser.add_argument('--overwrite', action='store_true', help='force overwrite')
  parser.add_argument('--data_path', default='data', help='folder path to downloaded dataset')
  parser.add_argument('--log_path', default='log', help='folder path to local trained model weights and logs')
  parser.add_argument('--img_path', default='img', help='folder path to image display')
  args = parser.parse_args()

  print('[Ckpt] use pretrained weights from torchvision/torchhub')
  args.train_dataset = 'imagenet'     # NOTE: currently all `torchvision.models` are pretrained on `imagenet`

  attack_multi(args)
