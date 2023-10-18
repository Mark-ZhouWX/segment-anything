import argparse
import os
import time
from datetime import datetime

import torch

from segment_anything.build_sam import create_model
from segment_anything.dataset.dataset import create_dataloader
from segment_anything.modeling.loss import create_loss_fn
from segment_anything.utils.logger import logger
from segment_anything.utils.config import parse_args
from segment_anything.utils.logger import setup_logging
from segment_anything.utils.utils import sec_to_dhms


def main(args) -> None:
    # Step1: initialize environment
    torch.manual_seed(42)

    local_rank, main_device, _ = set_distributed(args.distributed)

    set_log(args, local_rank)

    logger.info(args.pretty())

    # Step2: create dataset
    train_dataloader = create_dataloader(args.train_loader, args.distributed)
    eval_dataloader = create_dataloader(args.eval_loader, args.distributed)

    # create model, load pretrained ckpt, set amp level, also freeze layer if specified
    model = create_model(args.network.model)
    loss_fn = create_loss_fn(args.network.loss)
    model.train()
    model = model.cuda()
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # Step3: create optimizer, including learning rate scheduler and group parameter settings
    optimizer, scheduler = get_optimizer_and_scheduler(model, args.optimizer)

    # run training loop
    logger.info(f'start training')
    epoch_size = args.train_loader.epoch_size
    dataset_size = len(train_dataloader)
    train_start_time = time.time()
    log_interval = args.log_interval
    save_interval = args.save_interval
    accumulate_loss = []

    for epoch_id in range(epoch_size):
        cur_epoch = epoch_id + 1
        step_start_time = time.time()
        for step_id, (image, gt_masks, boxes) in enumerate(train_dataloader):
            cur_step = step_id + 1
            # forward and backward
            image, gt_masks, boxes = image.cuda(), gt_masks.cuda(), boxes.cuda()
            pred_masks, pred_ious = model(image, boxes)
            loss_dict = loss_fn(pred_masks, pred_ious, gt_masks)
            loss = loss_dict['loss']
            accumulate_loss.append(loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            if cur_step % log_interval == 0:
                # get time info
                step_cost = (time.time() - step_start_time) / log_interval # average step time
                step_start_time = time.time()
                train_already_cost = sec_to_dhms(time.time() - train_start_time)
                train_left = sec_to_dhms((dataset_size * epoch_size - cur_step) * step_cost)
                step_time_str = f'{step_cost:.2f}s'
                train_already_cost_str = f'{train_already_cost[0]}d {train_already_cost[1]:02d}:' \
                                         f'{train_already_cost[2]:02d}:{train_already_cost[3]:02d}'
                train_time_left_str = f'{train_left[0]}d {train_left[1]:02d}:{train_left[2]:02d}:{train_left[3]:02d}'

                lr = scheduler.get_lr()[0]
                smooth_loss = sum(accumulate_loss) / log_interval
                accumulate_loss = []

                logger.info(', '.join([
                    f'glb_step[{cur_step}/{dataset_size * epoch_size}]',
                    f'loc_step[{cur_step % dataset_size}/{dataset_size}]',
                    f'epoch[{cur_epoch}/{epoch_size}]',
                    f'loss[{loss.item():.4f}]',
                    f'smooth_loss[{smooth_loss.item():.4f}]',
                    f'lr[{lr:.7f}]',
                    f'step_time[{step_time_str}]',
                    f'already_cost[{train_already_cost_str}]',
                    f'train_left[{train_time_left_str}]',
                ]))
            if main_device and cur_epoch % save_interval == 0:
                save_path = os.path.join(args.work_dir, f'sam_{cur_epoch:03d}.pth')
                torch.save(model.state_dict(), save_path)
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


def set_log(args, rank_id):
    time = datetime.now()
    save_dir = f'{time.year}_{time.month:02d}_{time.day:02d}-' \
               f'{time.hour:02d}_{time.minute:02d}_{time.second:02d}'
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

    return local_rank, main_device, device


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
