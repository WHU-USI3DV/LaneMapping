import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# based on:
# https://github.com/zhezh/focalloss/blob/master/focalloss.py

class MeanLoss(nn.Module):
    def init(self):
        super(MeanLoss, self).init()
        self.l1 = nn.SmoothL1Loss(reduction = 'none')
    def forward(self, logits, label):
        n,c,h,w = logits.shape
        grid = torch.arange(c, device=logits.device).view(1,c,1,1)
        logits = (logits.softmax(1) * grid).sum(1)
        loss = self.l1(logits, label.float())[label != -1]
        return loss.mean()

def focal_loss(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'none',
    eps: Optional[float] = None):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    Where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        input: logits tensor with shape :math:`(N, C, *)` where C = number of classes.
        target: labels tensor with shape :math:`(N, *)` where each value is :math:`0 ≤ targets[i] ≤ C−1`.
        alpha: Weighting factor :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Deprecated: scalar to enforce numerical stabiliy. This is no longer used.

    Return:
        the computed loss.

    Example:
        >>> N = 5  # num_classes
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = focal_loss(input, target, alpha=0.5, gamma=2.0, reduction='mean')
        >>> output.backward()
    """
    if eps is not None and not torch.jit.is_scripting():
        warnings.warn(
            "`focal_loss` has been reworked for improved numerical stability "
            "and the `eps` argument is no longer necessary",
            DeprecationWarning,
            stacklevel=2,
        )

    n = input.shape[0]
    out_size = (n,) + input.shape[2:]

    assert (target.shape[1:] == input.shape[2:], f'Expected target size {out_size}, got {target.size()}')
    assert (
        input.device == target.device,
        f"input and target must be in the same device. Got: {input.device} and {target.device}",
    )

    # compute softmax over the classes axis
    input_soft: torch.Tensor = input.softmax(1)
    log_input_soft: torch.Tensor = input.log_softmax(1)

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = F.one_hot(target, num_classes=input.shape[1])

    # compute the actual focal loss
    weight = torch.pow(-input_soft + 1.0, gamma)

    focal = -alpha * weight * log_input_soft
    loss_tmp = torch.einsum('bc...,bc...->b...', (target_one_hot, focal))

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss


class FocalLoss(nn.Module):
    def __init__(self):
        super(FocalLoss, self).__init__()

    def forward(self, pred, gt):
        pos_inds = gt.eq(1).float()  # 比较是否等于1
        neg_inds = gt.lt(1).float()  # 判断ground truth 中的负样本的位置，即非center point的位置

        neg_weights = torch.pow(1 - gt, 4)  # pow为计算幂次方 （1-gt）的4次方
        loss = 0
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0.:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss


def binary_focal_loss_with_logits(
    input: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = 'none',
    eps: Optional[float] = None,
    pos_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Function that computes Binary Focal loss.

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        input: input data tensor of arbitrary shape.
        target: the target tensor with shape matching input.
        alpha: Weighting factor for the rare class :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        eps: Deprecated: scalar for numerically stability when dividing. This is no longer used.
        pos_weight: a weight of positive examples.
          It’s possible to trade off recall and precision by adding weights to positive examples.
          Must be a vector with length equal to the number of classes.

    Returns:
        the computed loss.

    Examples:
        >>> kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
        >>> logits = torch.tensor([[[6.325]],[[5.26]],[[87.49]]])
        >>> labels = torch.tensor([[[1.]],[[1.]],[[0.]]])
        >>> binary_focal_loss_with_logits(logits, labels, **kwargs)
        tensor(21.8725)
    """

    if eps is not None and not torch.jit.is_scripting():
        warnings.warn(
            "`binary_focal_loss_with_logits` has been reworked for improved numerical stability "
            "and the `eps` argument is no longer necessary",
            DeprecationWarning,
            stacklevel=2,
        )

    assert (input, ["B", "C", "*"])
    assert (
        input.shape[0] == target.shape[0],
        f'Expected input batch_size ({input.shape[0]}) to match target batch_size ({target.shape[0]}).',
    )

    if pos_weight is None:
        pos_weight = torch.ones(input.shape[-1], device=input.device, dtype=input.dtype)

    assert (input.shape[-1] == pos_weight.shape[0], "Expected pos_weight equals number of classes.")

    probs_pos = input.sigmoid()
    probs_neg = (-input).sigmoid()

    loss_tmp = (
        -alpha * pos_weight * probs_neg.pow(gamma) * target * probs_pos.log()
        - (1 - alpha) * probs_pos.pow(gamma) * (1.0 - target) * probs_neg.log()
    )

    if reduction == 'none':
        loss = loss_tmp
    elif reduction == 'mean':
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss

class BinaryFocalLossWithLogits(nn.Module):
    r"""Criterion that computes Focal loss.

    According to :cite:`lin2018focal`, the Focal loss is computed as follows:

    .. math::

        \text{FL}(p_t) = -\alpha_t (1 - p_t)^{\gamma} \, \text{log}(p_t)

    where:
       - :math:`p_t` is the model's estimated probability for each class.

    Args:
        alpha: Weighting factor for the rare class :math:`\alpha \in [0, 1]`.
        gamma: Focusing parameter :math:`\gamma >= 0`.
        reduction: Specifies the reduction to apply to the
          output: ``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
          will be applied, ``'mean'``: the sum of the output will be divided by
          the number of elements in the output, ``'sum'``: the output will be
          summed.
        pos_weight: a weight of positive examples.
          It’s possible to trade off recall and precision by adding weights to positive examples.
          Must be a vector with length equal to the number of classes.

    Shape:
        - Input: :math:`(N, *)`.
        - Target: :math:`(N, *)`.

    Examples:
        >>> kwargs = {"alpha": 0.25, "gamma": 2.0, "reduction": 'mean'}
        >>> loss = BinaryFocalLossWithLogits(**kwargs)
        >>> input = torch.randn(1, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(2)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(
        self, alpha: float, gamma: float = 2.0, reduction: str = 'none', pos_weight: Optional[torch.Tensor] = None
    ) -> None:
        super().__init__()
        self.alpha: float = alpha
        self.gamma: float = gamma
        self.reduction: str = reduction
        self.pos_weight: Optional[torch.Tensor] = pos_weight

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return binary_focal_loss_with_logits(
            input, target, self.alpha, self.gamma, self.reduction, pos_weight=self.pos_weight
        )
