import torch
from torch import nn
import torch.nn.functional as F

from segment_anything.utils.registry import LOSS_REGISTRY
from segment_anything.utils.utils import calc_iou


def create_loss_fn(args):
    """
    Ref from https://arxiv.org/abs/2304.02643 section3
    SAM loss is the combination of focal and dice loss.
    """
    loss_fn = LOSS_REGISTRY.instantiate(**args)
    return loss_fn


@LOSS_REGISTRY.registry_module()
class SAMLoss(nn.Module):
    """
    Ref from https://arxiv.org/abs/2304.02643 section3
    SAM loss is the combination of focal and dice loss.
    """
    def __init__(self, focal_factor=20.0, dice_factor=1.0, mse_factor=1.0, mask_threshold=0.0):
        super().__init__()
        self.focal_factor = focal_factor
        self.dice_factor = dice_factor
        self.mse_factor = mse_factor
        self.mask_threshold = mask_threshold

        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, pred_masks, pred_ious, gt_masks):
        """
        get loss, remove dynamic shape assisted with valid_boxes
        Args:
            pred_mask (List): predicted mask length b, each element is a tensor with shape (n_i, h, w)
            pred_iou (List): predicted iou length b, each element is a tensor with shape (n_i,)
        """
        bs = len(pred_masks)
        assert len(pred_ious) == bs and len(gt_masks) == bs

        focal_loss_list = []
        dice_loss_list = []
        mse_loss_list = []
        for i in range(bs):
            pred_mask = pred_masks[i].to(torch.float32)  # (b, n, h, w)  logits, un-simoid
            gt_mask = gt_masks[i].to(torch.float32)  # (b, n, h, w)
            pred_iou = pred_ious[i]

            pred_mask_01 = (pred_mask > self.mask_threshold).to(torch.float32)
            gt_iou = calc_iou(pred_mask_01, gt_mask).detach()  # (b, n)

            # if False: # show pred and gt mask for debug
            #     import matplotlib.pyplot as plt
            #     plt.imshow(pred_mask_01[0, 3].asnumpy())
            #     plt.show()
            #     plt.imshow(gt_mask[0, 3].asnumpy())
            #     plt.show()

            focal_loss_list.append(self.focal_loss(pred_mask, gt_mask).sum())   # (n) -> (1,)
            dice_loss_list.append(self.dice_loss(pred_mask, gt_mask).sum())
            mse_loss_list.append(self.mse_loss(pred_iou, gt_iou).sum())

        focal_loss = sum(focal_loss_list)  # (b,) -> (1)
        dice_loss = sum(dice_loss_list)
        mse_loss = sum(mse_loss_list)
        loss = self.focal_factor * focal_loss + self.dice_factor * dice_loss + self.mse_factor * mse_loss

        loss_dict = dict(loss=loss,
                         focal_loss=focal_loss,
                         dice_loss=dice_loss,
                         mse_loss=mse_loss
                         )

        return loss_dict


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, *args, **kwargs):
        """Dice loss for 2d segmentation. a replacement for mindspore.nn.DiceLoss that does not
        support reduction 'none' type."""
        super().__init__(*args, **kwargs)
        self.smooth = smooth

    def forward(self, logits, labels):
        logits = torch.sigmoid(logits)
        logits = logits.flatten(-2)  # (n, s)
        labels = labels.flatten(-2)  # (n, s)

        # formula refer to mindspore.nn.DiceLoss
        intersection = torch.sum(torch.mul(logits, labels), -1)  # (n, s) -> (n,)
        unionset = torch.sum(torch.mul(logits, logits) + torch.mul(labels, labels), -1)

        single_dice_coeff = (2 * intersection) / (unionset + self.smooth)
        dice_loss = 1 - single_dice_coeff  # (n,)

        return dice_loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2, *args, **kwargs):
        """
        Focal loss for 2D binary segmentation. a replacement for mindspore.nn.DiceLoss that only support multi-class.
        FL = - alpha * (1-Pt)**gamma * log(Pt)
        """
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, labels):
        logits = torch.sigmoid(logits)  # (n, h, w)
        logits = logits.flatten(-2)  # (n, s)
        labels = labels.flatten(-2)
        bce = F.binary_cross_entropy(logits, labels, reduction='none').mean(-1)  # (n,) equals to -log(pt)
        pt = torch.exp(-bce)  # pt
        focal_loss = self.alpha * (1- pt)**self.gamma * bce  # (n,)
        return focal_loss
