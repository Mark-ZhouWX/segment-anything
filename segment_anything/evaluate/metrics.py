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
    def __init__(self):
        super(MaskMiou, self).__init__()
        self.sum_iou = torch.tensor(0.0).cuda()
        self.num_step = torch.tensor(0.0).cuda()
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

        num_valid = sum(len(g) for g in gts)

        for pred, gt in zip(preds, gts):  # (b, n)
            pred_mask = pred.to(torch.float32)  # bool -> float32
            gt_mask = gt.to(torch.float32)

            assert pred_mask.shape == gt_mask.shape
            iou = calc_iou(pred_mask, gt_mask)  # (n,)
            iou_list.append(iou.sum())
        miou_per_batch = sum(iou_list) / num_valid  # (1,)

        self.num_step += 1
        self.sum_iou += miou_per_batch
    
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
