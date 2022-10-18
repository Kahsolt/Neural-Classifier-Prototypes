#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/10/18 

import gc
import os
from argparse import ArgumentParser
from random import randrange

import torch
import torch.nn.functional as F
from torchattacks import PGD

from model import *
from data import *
from util import *

#device = 'cpu'

#IMG_EXT = '.png'
#IMG_EXT = '.svg'
IMG_EXT = '.jpg'


def calc_grad(model, X, Y):
  images = X.clone()   # avoid inplace grad overwrite
  labels = Y.clone()

  images.requires_grad = True
  images_norm = normalize(images, args.atk_dataset)
  logits = model(images_norm)

  # Calculate loss (targeted); should be minimized element-wisely; [B=100]
  loss_each = F.cross_entropy(logits, labels, reduce=False)
  # Calculate 1st & 2dn order gradient; [B=100, C, H, W]
  grad_each  = torch.autograd.grad(outputs=loss_each, inputs=images, grad_outputs=loss_each, create_graph=True)[0]
  grad2_each = torch.autograd.grad(outputs=grad_each, inputs=images, grad_outputs=grad_each, create_graph=False)[0]

  return loss_each, grad_each, grad2_each


def show_grad(args):
  ''' Dirs '''
  os.makedirs(args.data_path, exist_ok=True)

  ''' Model '''
  model = get_model(args.model).to(device)
  model.eval()

  ''' Data '''
  y = None
  if args.D:
    dataloader = get_dataloader(args.atk_dataset, args.data_path, split='test', shuffle=True)
    nth = randrange(len(dataloader.dataset))
    for _ in range(nth):
      x, y = iter(dataloader).next()
    
    save_dn = f'{args.atk_dataset}_{nth}'
  else:
    X_shape = torch.Size([1, 3, args.height, args.width])
    if args.C:
      if args.const.isdigit():
        rgb = [int(args.const)] * 3
      else:
        rgb = [int(x) for x in args.const.split(',')]
      assert 0 <= min(rgb) and max(rgb) <= 255, f'bad rgb: {rgb}'

      x = torch.ones(X_shape)
      for c in range(3):
        x[:, c, :, :] = x[:, c, :, :] * rgb[c]
      x = x / 255.0

      save_dn = f'C[{rgb[0]},{rgb[1]},{rgb[2]}]'
    elif args.R:
      if args.rand == 'uniform':
        x = torch.empty(X_shape).uniform_(args.low, args.high)
        save_dn = f'rU[{float_to_str(args.low)},{float_to_str(args.high)}]'
      elif args.rand == 'normal':
        x = torch.empty(X_shape).normal_(args.mu, args.sigma)
        save_dn = f'rN[{float_to_str(args.mu)},{float_to_str(args.sigma)}]'
      else: raise
  
  os.makedirs(os.path.join(args.img_path, save_dn), exist_ok=True)
  print('save_dn:', save_dn)

  X = x.to(device)                            # [B=1, C, H, W]
  with torch.no_grad():
    y_hat = model(X)                          # [B=1, N_CLASS]
    N_CLASSES = y_hat.shape[-1]               # N_CLASS
    assert N_CLASSES % args.batch_size == 0
  
  ''' Attack '''
  atk = PGD(model, eps=0.03, alpha=0.001, steps=40)
  atk.set_mode_targeted_by_function(lambda x, y: y)

  ''' Show Grad To Each Class '''
  def savefig_x_probdist(logits, x, y):
    prob = F.softmax(logits, dim=-1).squeeze()
    pred = logits.argmax().cpu().item()
    print(f'>> pred: {pred}, prob: {prob[pred]:%}')

    plt.subplot(211) ; plt.plot(prob.squeeze().cpu().numpy())
    plt.subplot(212) ; plt.plot(logits.squeeze().cpu().numpy())
    plt.suptitle(f'pred: {pred}' + (f', truth: {y}' if y else ''))
    plt.tight_layout()
    plt.savefig(os.path.join(args.img_path, save_dn, f'_{IMG_EXT}'), bbox_inches='tight', pad_inches=0.1)
    plt.clf()

    plt.imshow(x)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(args.img_path, save_dn, f'~{IMG_EXT}'), bbox_inches='tight', pad_inches=0.1)
    plt.clf()

  def savefig_grads(loss_each, grad_each, grad2_each, cls_s, suffix=''):
    for i in range(len(loss_each)):
      loss  = loss_each [i]
      grad  = grad_each [i]
      grad2 = grad2_each[i]
      
      def minmax_norm(x): return (x - x.min()) / (x.max() - x.min())

      plt.suptitle(f'loss: {loss:.2f}, ' + 
                    f'grad: [{grad.min():.2f}, {grad.max():.2f}, {np.median(np.abs(grad)):.2f}]\n' + 
                    f'grad2: [{grad2.min():.2f}, {grad2.max():.2f}, {np.median(np.abs(grad2)):.2f}]')
      plt.subplot(221) ; plt.imshow(grad.clip(0, 1))    ; plt.title('grad')        ; plt.axis('off')
      plt.subplot(222) ; plt.imshow(grad2.clip(0, 1))   ; plt.title('grad2')       ; plt.axis('off')
      plt.subplot(223) ; plt.imshow(minmax_norm(grad))  ; plt.title('norm(grad)')  ; plt.axis('off')
      plt.subplot(224) ; plt.imshow(minmax_norm(grad2)) ; plt.title('norm(grad2)') ; plt.axis('off')
      plt.subplots_adjust(top=1)
      plt.savefig(os.path.join(args.img_path, save_dn, f'{i + cls_s}{suffix}{IMG_EXT}'), bbox_inches='tight', pad_inches=0.1)
      plt.clf()

  def savefig_ax_probdist(logits, cls_s, y):
    for i in range(len(logits)):
      logit = logits[i]
      prob = F.softmax(logit, dim=-1).squeeze()
      pred = logit.argmax().cpu().item()

      plt.subplot(211) ; plt.plot(prob.squeeze().cpu().numpy())
      plt.subplot(212) ; plt.plot(logit.squeeze().cpu().numpy())
      plt.suptitle(f'pred: {pred}' + (f', truth: {y}' if y else ''))
      plt.tight_layout()
      plt.savefig(os.path.join(args.img_path, save_dn, f'_{i + cls_s}_adv{IMG_EXT}'), bbox_inches='tight', pad_inches=0.1)
      plt.clf()

  with torch.no_grad():
    logits = model(normalize(X, dataset=args.atk_dataset))
    savefig_x_probdist(logits, 
                       x.squeeze().cpu().numpy().transpose([1, 2, 0]),
                       y.squeeze().item())

  X_repeat = X.repeat([args.batch_size, 1, 1, 1])      # [B, C=3, H, W]
  for b in range(N_CLASSES // args.batch_size):
    cls_s = b * args.batch_size
    cls_e = (b + 1) * args.batch_size
    print(f'>> test for class {cls_s} to {cls_e - 1}')
    Y_tgt = torch.LongTensor([i for i in range(cls_s, cls_e)]).to(device)

    if 'original':
      loss_each, grad_each, grad2_each = calc_grad(model, X_repeat, Y_tgt)
      loss_each  = loss_each .detach().cpu().numpy()                          # [B]
      grad_each  = grad_each .detach().cpu().numpy().transpose([0, 2, 3, 1])  # [B, H, W, C]
      grad2_each = grad2_each.detach().cpu().numpy().transpose([0, 2, 3, 1])

      savefig_grads(loss_each, grad_each, grad2_each, cls_s)

    if 'adversarial':
      AX = atk(X_repeat, Y_tgt)

      logits = model(AX)
      savefig_ax_probdist(logits.detach(), cls_s, y.squeeze().item())

      loss_each, grad_each, grad2_each = calc_grad(model, AX, Y_tgt)
      loss_each  = loss_each .detach().cpu().numpy()                          # [B]
      grad_each  = grad_each .detach().cpu().numpy().transpose([0, 2, 3, 1])  # [B, H, W, C]
      grad2_each = grad2_each.detach().cpu().numpy().transpose([0, 2, 3, 1])

      savefig_grads(loss_each, grad_each, grad2_each, cls_s, '-adv')

    gc.collect()
    if device == 'cuda': torch.cuda.empty_cache()


if __name__ == '__main__':
  RANDOMS = ['uniform', 'normal']

  parser = ArgumentParser()
  parser.add_argument('-M', '--model', default='resnet18', choices=MODELS, help='victim model with pretrained weight')
  parser.add_argument('--mode', default='min', choices=['min', 'max'], help='find nearest point with min/max grad')

  parser.add_argument('-D', action='store_true', help='attack on dataset, use with --atk_dataset')
  parser.add_argument('-C', action='store_true', help='attack on pure color, use with --const' )
  parser.add_argument('-R', action='store_true', help='attack on random noise, use with --random')
  parser.add_argument('--atk_dataset', default='imagenet-1k', choices=DATASETS, help='victim dataset')
  parser.add_argument('--const',       default='127', type=str, help='const single pixel or comma separated RGB values like 11,45,14 (values in 0 ~ 255)')
  parser.add_argument('--rand',        default='uniform', choices=RANDOMS, help='victim model with pretrained weight')
  parser.add_argument('--low',   default=0.0, type=float, help='rand param for uniform')
  parser.add_argument('--high',  default=1.0, type=float, help='rand param for uniform')
  parser.add_argument('--mu',    default=0.5, type=float, help='rand param for normal')
  parser.add_argument('--sigma', default=0.1, type=float, help='rand param for normal')
  parser.add_argument('-H', '--height', default=224, type=int, help='image height')
  parser.add_argument('-W', '--width',  default=224, type=int, help='image width')

  parser.add_argument('-B', '--batch_size', type=int, default=100, help='process n_attacks on one image simultaneously, must be divisible by model n_classes')
  parser.add_argument('--overwrite', action='store_true', help='force overwrite')
  parser.add_argument('--data_path', default='data', help='folder path to downloaded dataset')
  parser.add_argument('--log_path', default='log', help='folder path to local trained model weights and logs')
  parser.add_argument('--img_path', default='img', help='folder path to image display')
  args = parser.parse_args()

  assert sum([args.D, args.C, args.R]) == 1, 'must specify one and only one from -D, -C, -R'

  print('[Ckpt] use pretrained weights from torchvision/torchhub')
  args.train_dataset = 'imagenet'     # NOTE: currently all `torchvision.models` are pretrained on `imagenet`

  show_grad(args)
