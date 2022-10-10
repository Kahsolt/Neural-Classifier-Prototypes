#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/27 

from random import randrange

import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == 'cuda':
  torch.backends.cudnn.enabled = True
  torch.backends.cudnn.benchmark = True

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


def analyze_NX(nx: dict):
  device = nx['x'].device

  n_classes = nx['nx_pred'].shape[-1]
  Y_tgt = torch.LongTensor([i for i in range(n_classes)]).to(device)
  mask_alienated = (nx['nx_pred'] == Y_tgt)
  alienate_rate = mask_alienated.sum() / n_classes

  L0   = torch.stack([(dx.abs() > 0).to(torch.float32).mean()       for dx in nx['dx']])
  L1   = torch.stack([dx.abs().mean()                               for dx in nx['dx']])
  L2   = torch.stack([dx.view(1, -1).norm(p=2, dim=-1) / dx.numel() for dx in nx['dx']])
  Linf = torch.stack([dx.abs().max()                                for dx in nx['dx']])
  loss = nx['loss']
  grad = nx['grad']

  print(f'L0  : max={L0  .max()} min={L0  .min()}')
  print(f'L1  : max={L1  .max()} min={L1  .min()}')
  print(f'L2  : max={L2  .max()} min={L2  .min()}')
  print(f'Linf: max={Linf.max()} min={Linf.min()}')
  print(f'loss: max={loss.max()} min={loss.min()}')
  print(f'grad: max={grad.max()} min={grad.min()}')


def plt_images(AX):
  AX = make_grid(AX, nrow=10)
  AX = AX.cpu().numpy()
  plt.imshow(AX.transpose([1, 2, 0]))
  plt.show()


def save_images(AX, fp):
  AX = make_grid(AX, nrow=10)
  AX = AX.cpu().numpy()
  plt.imshow(AX.transpose([1, 2, 0]))
  plt.savefig(fp, dpi=400)
  

# 'resnet18_imagenet-min-svhn_pgd_e3e-2_a1e-3'
def exp_name(model='resnet18', train_dataset='imagenet', mode='min', atk_dataset='svhn', method='pgd', eps=0.03, alpha=0.001) -> str:
  return f'{model}_{train_dataset}-{mode}-{atk_dataset}_{method}_e{float_to_str(eps)}_a{float_to_str(alpha)}'

# 'resnet18_imagenet-min-pgd_c(r,g,b)'
# 'resnet18_imagenet-min-pgd_rU(low,high)'
# 'resnet18_imagenet-min-pgd_rN(mu,sigma)'
def exp_cr_name(model='resnet18', train_dataset='imagenet', mode='min', method='pgd',
               const=None, rand=None, low=None, high=None, mu=None, sigma=None) -> str:
  prefix = f'{model}_{train_dataset}-{mode}-{method}'

  if const:
    return f'{prefix}_c({",".join([str(c) for c in const])})'
  else:
    if rand == 'uniform':
      return f'{prefix}_rU({float_to_str(low)},{float_to_str(high)})'
    elif rand == 'normal':
      return f'{prefix}_rN({float_to_str(mu)},{float_to_str(sigma)})'

  raise ValueError('bad experimental setting')


def float_to_str(x:str, n_prec:int=3) -> str:
  # integer
  if int(x) == x: return str(int(x))
  
  # float
  sci = f'{x:e}'
  frac, exp = sci.split('e')
  
  frac_r = round(float(frac), n_prec)
  frac_s = f'{frac_r}'
  if frac_s.endswith('.0'):   # remove tailing '.0'
    frac_s = frac_s[:-2]
  exp_i = int(exp)
  
  if exp_i != 0:
    # '3e-5', '-1.2e+3'
    return f'{frac_s}e{exp_i}'
  else:
    # '3.4', '-1.2'
    return f'{frac_s}'
