import argparse

import cv2
import numpy as np

from segment_anything.build_sam import sam_model_registry
from segment_anything.dataset.transform import TransformPipeline, ImageNorm, ImageResizeAndPad
import matplotlib.pyplot as plt

from segment_anything.utils.utils import Timer, to_cuda
from segment_anything.utils.visualize import show_mask, show_box


def infer(args):
    # Step1: data preparation
    with Timer('preprocess'):
        transform_list = [
            ImageResizeAndPad(target_size=1024, apply_mask=False),
            ImageNorm(),
        ]
        transform_pipeline = TransformPipeline(transform_list)

        image_path = args.image_path
        image_np = cv2.imread(image_path)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        boxes_np = np.array([[425, 600, 700, 875]])

        transformed = transform_pipeline(dict(image=image_np, boxes=boxes_np))
        image, boxes, origin_hw = transformed['image'], transformed['boxes'], transformed['origin_hw']
        # batch_size for speed test
        # image = ms.Tensor(np.expand_dims(image, 0).repeat(8, axis=0))  # b, 3, 1023
        # boxes = ms.Tensor(np.expand_dims(boxes, 0).repeat(8, axis=0))  # b, n, 4


    # Step2: inference
    with Timer('load weight and build net, move data and model to cuda'):
        network = sam_model_registry[args.model_type](checkpoint=args.checkpoint)
        network.eval()
        network = network.cuda()
        image = to_cuda(image).unsqueeze(0)  # b, 3, 1023
        boxes = to_cuda(boxes).unsqueeze(0)  # b, n, 4
    with Timer('model compile + inference'):
        mask_logits = network(image, boxes)[0][0]   # first dim [mask, iou], second [batch] (1, 1, 1024, 1024)

    with Timer('Second time inference'):
        mask_logits = network(image, boxes)[0][0]  # (1, 1, 1024, 1024)

    # Step3: post-process
    with Timer('post-process'):
        mask_logits = mask_logits.detach().cpu().numpy()[0] > 0.0
        mask_logits = mask_logits.astype(np.uint8)
        final_mask = cv2.resize(mask_logits[:origin_hw[2], :origin_hw[3]], tuple((origin_hw[1], origin_hw[0])),
                                interpolation=cv2.INTER_CUBIC)

    # Step4: visualize
    plt.imshow(image_np)
    show_box(boxes_np[0], plt.gca())
    show_mask(final_mask, plt.gca())
    plt.savefig(args.image_path + '_infer.jpg')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=("Runs inference on one image"))
    parser.add_argument("--image_path", type=str, default='./notebooks/images/truck.jpg', help="Path to an input image.")
    parser.add_argument(
        "--model-type",
        type=str,
        default='vit_b',
        help="The type of model to load, in ['vit_h', 'vit_l', 'vit_b']",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default='./models/sam_vit_b_01ec64.pth',
        help="The type of model to load, in ['default', 'vit_h', 'vit_l', 'vit_b']",
    )

    args = parser.parse_args()
    print(args)
    infer(args)
