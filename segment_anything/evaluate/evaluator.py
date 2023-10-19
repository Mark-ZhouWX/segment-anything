from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from segment_anything.evaluate.metrics import BaseMetric
from segment_anything.utils.logger import logger
from segment_anything.utils.utils import to_cuda


class Evaluator:
    """
    Evaluator for SAM segmentation predictions

    """
    def __init__(self,
                 network: nn.Module,
                 data_loader: DataLoader,
                 metric: List[BaseMetric],
                 input_column: List[List[str]],
                 ):
        self.network = network
        # Instantiate dataloader here instead of outside,
        # since there is no good way to feed eval_dataloader to training process
        self.data_loader =data_loader
        self.metric = metric if isinstance(metric, list) else [metric]
        self.model_columns = input_column[0]
        self.eval_columns = input_column[1]

    def eval(self, max_iter=None):
        logger.info(f'start evaluation, the metric is {self.metric}')

        train_status = self.network.training
        self.network.eval()
        with torch.no_grad():
            eval_res = dict()

            for m in self.metric:
                m.clear()

            dataset_size = len(self.data_loader)
            total = dataset_size if max_iter is None else max_iter
            for i, data in tqdm(enumerate(self.data_loader), total=total, desc='evaluating'):
                if max_iter is not None and i >= max_iter:
                    break
                inputs = [to_cuda(data[j]) for j in self.model_columns]

                preds = self.network(*inputs)

                gt_dict = {k:to_cuda(data[k]) for k in self.eval_columns} # dict

                # post-process
                pred_dict = mask_postprocess(preds)

                # evaluate
                for m in self.metric:
                    m.update(pred_dict, gt_dict)

            for m in self.metric:
                eval_res.update(m.eval())

        # recover train status before evaluation
        self.network.train(train_status)

        res_str = ', '.join([f'{k}: {v*100:.2f}' for k, v in eval_res.items()])
        logger.info(f'finish evaluation, the result is, {res_str}')

        return eval_res

    
def mask_postprocess(preds: Tuple, threshold=0):
    """
    1. mask thresholding applied to the logits output by model backbone.
    2. convert tuple to dict
    """
    pred_dict = dict()
    pred_dict['masks'] = [m > threshold for m in preds[0]]
    pred_dict['ious'] = preds[1]

    return pred_dict

