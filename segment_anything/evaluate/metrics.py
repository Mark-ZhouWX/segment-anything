import sys
from typing import List

import mindspore as ms
import torch

from segment_anything.utils.registry import METRIC_REGISTRY
from segment_anything.utils.utils import calc_iou

class BaseMetric:
    def clear(self, *args, **kwargs):
        return
    def update(self, *args, **kwargs):
        return
    def eval(self, *args, **kwargs):
        return


@METRIC_REGISTRY.registry_module()
class MaskMiou(BaseMetric):
    def __init__(self, max_box_pre_image=sys.maxsize):
        """
        calculate mean IOU of between mask prediction and ground truth
        Args:
            max_box_pre_image: maximum of box number per image. exceeding boxes are omitted. default, no limitation
        """
        super(MaskMiou, self).__init__()
        self.sum_iou = torch.tensor(0.0).cuda()
        self.num_step = torch.tensor(0.0).cuda()
        self.max_box_pre_image = max_box_pre_image
        self.clear()

    def clear(self):
        self.sum_iou = torch.tensor(0.0).cuda()
        self.num_step = torch.tensor(0.0).cuda()

    def update(self, *inputs):
        """
        update metrics of a batch
        """
        preds, gts = inputs[0]['masks'], inputs[1]['masks']
        iou_list = []

        num_valid = []

        for pred, gt in zip(preds, gts):
            v = min(self.max_box_pre_image, len(pred))
            pred_mask = pred.to(torch.float32)[:v]  # bool -> float32
            gt_mask = gt.to(torch.float32)[:v]

            assert pred_mask.shape == gt_mask.shape
            iou = calc_iou(pred_mask, gt_mask)  # (n,)
            iou_list.append(iou.sum())
            num_valid.append(v)

        self.num_step += sum(num_valid)   # (1,)
        self.sum_iou += sum(iou_list)
    
    def eval(self):
        # reduce from all the devices
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(self.sum_iou)
            torch.distributed.all_reduce(self.num_step)

        sum_iou = self.sum_iou
        num_step = self.num_step

        miou = sum_iou / num_step
        print(sum_iou, num_step, miou)
        miou = miou.item()
        res_dict = dict(miou=miou)
        return res_dict


def create_metric(metric: List):
    """
    instantiate metric class
    """
    metric_list = []
    for m in metric:
        metric_list.append(METRIC_REGISTRY.instantiate(**m))
    return metric_list
