import argparse

import torch

from segment_anything.build_sam import create_model
from segment_anything.dataset.dataset import create_dataloader
from segment_anything.evaluate.evaluator import Evaluator
from segment_anything.evaluate.metrics import create_metric
from segment_anything.utils.config import parse_args
from segment_anything.utils.logger import logger
from segment_anything.utils.utils import set_distributed, set_log


def main(args) -> None:
    # Step1: initialize environment
    torch.manual_seed(42)

    local_rank, main_device, _ = set_distributed(args.distributed)

    set_log(args, local_rank)

    logger.info(args.pretty())

    # Step2: create dataset
    eval_dataloader = create_dataloader(args.eval_loader, args.distributed)

    # Step3: create model, load pretrained ckpt, set amp level, also freeze layer if specified
    model = create_model(args.network.model)
    model.eval()
    model = model.cuda()
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    # Step4: create evaluator
    metric = create_metric(args.eval_metric)
    evaluator = Evaluator(model, eval_dataloader, metric=metric,
                          input_column=[args.eval_loader.model_column, args.eval_loader.eval_column])

    # Step5: eval
    logger.info(f'start evaluation')
    evaluator.eval(args.eval_loader.max_eval_iter)
    logger.info(f'finish evaluation')


if __name__ == "__main__":
    parser_config = argparse.ArgumentParser(description="SAM Config", add_help=False)
    parser_config.add_argument(
        "-c", "--config", type=str, default="./configs/coco_box_finetune.yaml",
        help="YAML config file specifying default arguments."
    )
    parser_config.add_argument('-o', '--override-cfg', nargs='+', help="command config to override that in config file")
    args = parse_args(parser_config)
    main(args)
