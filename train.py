import argparse
import os
import time
from datetime import datetime

import torch

from segment_anything.build_sam import create_model
from segment_anything.dataset.dataset import create_dataloader
from segment_anything.modeling.loss import create_loss_fn
from segment_anything.utils import logger
from segment_anything.utils.config import parse_args
from segment_anything.utils.logger import setup_logging
from segment_anything.utils.utils import sec_to_dhms


def main(args) -> None:
    # Step1: initialize environment
    torch.manual_seed(42)

    set_log(args)

    logger.info(args.pretty())

    # Step2: create dataset
    train_dataloader = create_dataloader(args.train_loader)
    eval_dataloader = create_dataloader(args.eval_loader)

    # create model, load pretrained ckpt, set amp level, also freeze layer if specified
    model = create_model(args.network.model)
    loss_fn = create_loss_fn(args.network.loss)
    model.train()

    # Step3: create optimizer, including learning rate scheduler and group parameter settings
    optimizer, scheduler = get_optimizer_and_scheduler(model, args.optimizer)

    # run training loop
    logger.info(f'start training')
    epoch_size = args.train_loader.epoch_size
    dataset_size = len(train_dataloader)
    model = model.to(args.device)
    train_start_time = time.time()
    for epoch_id in range(epoch_size):
        cur_epoch = epoch_id + 1
        step_start_time = time.time()
        for step_id, (image, gt_masks, boxes) in enumerate(train_dataloader):
            cur_step = step_id + 1
            # forward and backward
            image, gt_masks, gt_boxes = image.to(args.device), gt_masks.to(args.device), boxes.to(args.device)
            pred_masks, pred_ious = model(image, gt_boxes)
            loss_dict = loss_fn(pred_masks, pred_ious, gt_masks)
            loss = loss_dict['loss']
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # get time info
            step_cost = time.time() - step_start_time
            step_start_time = time.time()
            train_already_cost = sec_to_dhms(time.time() - train_start_time)
            train_left = sec_to_dhms((dataset_size * epoch_size - cur_step) * step_cost)
            step_time_str = f'{step_cost:.2f}s'
            train_already_cost_str = f'{train_already_cost[0]}d {train_already_cost[1]:02d}:' \
                                     f'{train_already_cost[2]:02d}:{train_already_cost[3]:02d}'
            train_time_left_str = f'{train_left[0]}d {train_left[1]:02d}:{train_left[2]:02d}:{train_left[3]:02d}'

            # print log
            lr = scheduler.get_lr()[0]
            logger.info(', '.join([
                f'glb_step[{cur_step}/{dataset_size * epoch_size}]',
                f'loc_step[{cur_step % dataset_size}/{dataset_size}]',
                f'epoch[{cur_epoch}/{epoch_size}]',
                f'loss[{loss.item():.4f}]',
                # f'smooth_loss[{smooth_loss.asnumpy():.4f}]',
                f'lr[{lr:.7f}]',
                f'step_time[{step_time_str}]',
                f'already_cost[{train_already_cost_str}]',
                f'train_left[{train_time_left_str}]',
            ]))
    logger.info(f'finish training')

def get_optimizer_and_scheduler(model, args_optimizer):

    def lr_lambda(step):

        if step < lr_config.warmup_steps:
            return step / lr_config.warmup_steps
        elif step < lr_config.decay_steps[0]:
            return 1.0
        elif step < lr_config.decay_steps[1]:
            return 1 / lr_config.decay_factor
        else:
            return 1 / (lr_config.decay_factor**2)

    lr_config = args_optimizer.lr_scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_config.learning_rate, weight_decay=args_optimizer.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def set_log(args):
    time = datetime.now()
    save_dir = f'{time.year}_{time.month:02d}_{time.day:02d}-' \
               f'{time.hour:02d}_{time.minute:02d}_{time.second:02d}'
    setup_logging(log_dir=os.path.join(args.work_root, save_dir, 'log'), log_level=args.log_level)


if __name__ == "__main__":
    parser_config = argparse.ArgumentParser(description="SAM Config", add_help=False)
    parser_config.add_argument(
        "-c", "--config", type=str, default="configs/coco_box_finetune.yaml",
        help="YAML config file specifying default arguments."
    )
    parser_config.add_argument('-o', '--override-cfg', nargs='+',
                               help="command line to override configuration in config file."
                                    "For dict, use key=value format, eg: device=False. "
                                    "For nested dict, use '.' to denote hierarchy, eg: optimizer.weight_decay=1e-3."
                                    "For list, use number to denote position, eg: callback.1.interval=100.")
    args = parse_args(parser_config)
    main(args)
