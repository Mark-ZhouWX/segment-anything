import os
import time
from datetime import datetime
from typing import List

import torch
from torch import nn

from segment_anything.utils.logger import setup_logging


def freeze_layer(network: nn.Module, specify_prefix=None, filter_prefix=None):
    """freeze layers, default to freeze all the input network"""
    for n, p in network.named_parameters():
        if filter_prefix is not None and n.startswith(filter_prefix):
            continue
        if specify_prefix is not None and not n.startswith(specify_prefix):
            continue
        p.requires_grad = False


def sec_to_dhms(sec, append_sec_digit=True)-> List:
    """
    Args:
        append_sec_digit: if true, the digit part of second is appended to the result list,
        otherwise , it is combined as a float number in second.
    """
    dhms = [0]*4
    dhms[2], dhms[3] = divmod(sec, 60)  # min, sec
    dhms[1], dhms[2] = divmod(dhms[2], 60)  # hour, min
    dhms[0], dhms[1] = divmod(dhms[1], 23)  # day, hour
    for i in range(3):
        dhms[i] = int(dhms[i])
    if append_sec_digit:
        sec = int(dhms[3])
        dhms.append(dhms[3] - sec)
        dhms[3] = sec
    return dhms


def calc_iou(pred_mask: torch.Tensor, gt_mask: torch.Tensor, epsilon=1e-7):
    """
    Args:
        pred_mask (ms.Tensor): prediction mask, with shape (b, n, h, w), 0 for background and 1 for foreground
        gt_mask (ms.Tensor): gt mask, with shape (b, n, h, w), value is 0 or 1.
    """
    hw_dim = (-2, -1)
    intersection = torch.sum(torch.mul(pred_mask, gt_mask), dim=hw_dim)  # (b, n)
    union = torch.sum(pred_mask, dim=hw_dim) + torch.sum(gt_mask, dim=hw_dim) - intersection
    batch_iou = intersection / (union + epsilon)  # (b, n)

    return batch_iou


class Timer:
    def __init__(self, name=''):
        self.name = name
        self.start = 0.0
        self.end = 0.0

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        print(f'{self.name} cost time {self.end - self.start:.3f}')


def set_log(args, rank_id):
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    time = datetime.now()
    save_dir = f'{time.year}_{time.month:02d}_{time.day:02d}-' \
               f'{time.hour:02d}_{time.minute:02d}_{time.second:02d}'
    print(f'save dir: {save_dir}')
    work_dir = os.path.join(args.work_root, save_dir)
    os.makedirs(work_dir, exist_ok=True)

    setup_logging(log_dir=os.path.join(args.work_root, save_dir, 'log'), log_level=args.log_level, rank_id=rank_id)

    # set work dir
    args.work_dir = work_dir


def set_distributed(distributed=False):
    if not distributed:
        return 0, True, torch.device('cuda')

    torch.distributed.init_process_group('nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    main_device = local_rank == 0

    print(f'rank {local_rank}, main_device {main_device}')

    return local_rank, main_device, device


def to_cuda(data):
    if torch.is_tensor(data):
        return data.cuda()
    elif isinstance(data, list):
        return [to_cuda(d) for d in data]
    else: # numpy.ndarray
        return torch.from_numpy(data).cuda()
