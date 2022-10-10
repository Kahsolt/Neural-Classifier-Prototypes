#!/usr/bin/env python3
# Author: Armit
# Create Time: 2022/09/28 

import torch
import torch.nn as nn
from torchattacks.attack import Attack

import logging

# 不依赖 true_label 的极性
ATK_METHODS = [
  # Linf
  'pgd', 
  'mifgsm',
  # L2
  'pgdl2', 
]

logger = logging.getLogger('ncp')

# 异化
class Alienater(Attack):

  def __init__(self, model, method='pgd', eps=8/255, alpha=1/255, steps=40, **kwargs):
    super().__init__("Alienater", model)
    self.eps = eps
    self.alpha = alpha
    self.steps = steps
    self.method = method
    self._supported_mode = ['targeted']
    self.kwargs = kwargs

    # set for compatibility
    self._attack_mode = 'targeted'
    self._targeted = True
    self._target_map_function = lambda x: x

    # data preprocess fixup
    self.normalizer = kwargs.get('normalizer', (lambda x: x))

    self.ce_loss = nn.CrossEntropyLoss(reduce=False)   # keep element-wise
    self.eps_for_division = 1e-9

  def forward(self, images, labels):
    images = images.clone()   # avoid inplace grad overwrite
    labels = labels.clone()
    return getattr(self, self.method)(images, labels)

  def pgd(self, images: torch.Tensor, labels: torch.Tensor):
    ''' modified from torchattacks.attacks.PGD '''

    beta = self.kwargs.get('beta', 1)     # balance between grad and grad^2

    B = images.shape[0]
    images = images.detach().to(self.device)
    labels = labels.detach().to(self.device)

    delta = torch.empty_like(images).uniform_(-self.eps, self.eps).to(self.device)

    adv_images = images.detach() + delta
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for i in range(self.steps):
      adv_images.requires_grad = True
      adv_images_norm = self.normalizer(adv_images)    # delayed normalize until here
      logits = self.model(adv_images_norm)
      
      pred = logits.argmax(dim=-1)
      mask_alienated = (pred == labels)      # [B=100]
      alienate_rate = mask_alienated.sum() / labels.numel()

      # Calculate loss (targeted); should be minimized element-wisely; [B=100]
      loss_each = self.ce_loss(logits, labels)
      # Calculate 1st & 2dn order gradient; [B=100, C, H, W]
      grad_each  = torch.autograd.grad(outputs=loss_each, inputs=adv_images, grad_outputs=loss_each, create_graph=True)[0]
      grad2_each = torch.autograd.grad(outputs=grad_each, inputs=adv_images, grad_outputs=grad_each, create_graph=False)[0]
      # 记录梯度误差 (仅作统计评估用)
      grad_flatten = grad_each.view(B, -1)
      grad_agg = grad_flatten.norm(p=2, dim=-1) / grad_flatten.shape[-1]    # [B]
      grad2_flatten = grad2_each.view(B, -1)
      grad2_agg = grad2_flatten.norm(p=2, dim=-1) / grad2_flatten.shape[-1]   # [B]
      
      # 对所有尚未异化的样本，朝梯度负向走一步，正常优化/演化使得 `f(x) = y_i`
      # NOTE: this is Fast-Gradient-Tanh-Method (FGTM), soft version of FGSM
      #       梯度越小改变量越小，没有梯度就不改变 (但要小心意外的梯度消失)
      if True or alienate_rate < 1.0:
        mask_expand = (~mask_alienated).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        adv_images = adv_images - self.alpha * grad_each.tanh() * mask_expand
      # 进一步尝试满足二阶性质，使得 `f'(x) -> 0 || inf`
      if True:
        if self.kwargs['mode'] == 'min':      # 最小化梯度即沿着二阶梯度负向走一步
          adv_images = adv_images - self.alpha * grad2_each.sign() * 0.1
        elif self.kwargs['mode'] == 'max':    # 最大化梯度即沿着二阶梯度正向走一步
          adv_images = adv_images + self.alpha * grad2_each.sign() * beta
        else:
          raise ValueError('unknown mode, choose from ["min", "max"]')

      # 规范化
      delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
      adv_images = torch.clamp(images + delta, min=0, max=1).detach()

      if i % 10 == 0:
        logger.info(f'   [{i}/{self.steps}] aliente_rate: {alienate_rate:.3%}, loss_avg: {loss_each.mean():.6}, ' + 
                    f'grad_avg: {grad_agg.mean():.6}, grad2_avg: {grad2_agg.mean():.6}')

      # 提前结束战斗 ;)
      if loss_each.mean() == 0.0: break
    
    return adv_images, delta, loss_each, grad_agg, pred

  def mifgsm(self, images: torch.Tensor, labels: torch.Tensor):
    ''' modified from torchattacks.attacks.MIFGSM '''

    momentum_decay = self.kwargs.get('momentum_decay', 1.0)

    images = images.detach().to(self.device)
    labels = labels.detach().to(self.device)

    momentum = torch.zeros_like(images).detach().to(self.device)

    adv_images = images.detach()
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(self.steps):
      adv_images.requires_grad = True
      outputs = self.model(adv_images)

      # Calculate loss
      loss = self.ce_loss(outputs, labels)

      # Update adversarial images
      grad_r = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
      grad = grad_r / torch.mean(torch.abs(grad_r), dim=(1,2,3), keepdim=True)
      grad = grad + momentum * momentum_decay
      momentum = grad

      adv_images = adv_images.detach() - self.alpha * grad.sign()
      delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
      adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    return adv_images, delta, grad_r

  def pgdl2(self, images: torch.Tensor, labels: torch.Tensor):
    ''' modified from torchattacks.attacks.PGDL2 '''

    B = images.shape[0]
    images = images.detach().to(self.device)
    labels = labels.detach().to(self.device)

    delta = torch.empty_like(adv_images).normal_()
    d_flat = delta.view(B, -1)
    n = d_flat.norm(p=2, dim=1).view(B, 1, 1, 1)
    r = torch.zeros_like(n).uniform_(0, 1)
    delta *= r / n * self.eps
    adv_images = torch.clamp(adv_images + delta, min=0, max=1).detach()

    for _ in range(self.steps):
      adv_images.requires_grad = True
      outputs = self.model(adv_images)

      # Calculate loss
      loss = self.ce_loss(outputs, labels)

      # Update adversarial images
      grad_r = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
      grad_norms = torch.norm(grad_r.reshape(B, -1), p=2, dim=1) + self.eps_for_division
      grad = grad_r / grad_norms.reshape(B, 1, 1, 1)
      adv_images = adv_images.detach() - self.alpha * grad     # 朝梯度负向走一步

      delta = adv_images - images
      delta_norms = torch.norm(delta.view(B, -1), p=2, dim=1)
      factor = self.eps / delta_norms
      factor = torch.min(factor, torch.ones_like(delta_norms))
      delta = delta * factor.reshape(-1, 1, 1, 1)
      adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    return adv_images, delta, grad_r
