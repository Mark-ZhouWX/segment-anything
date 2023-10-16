import time

import numpy as np
import matplotlib.pyplot as plt
import cv2

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)



def main():
    checkpoints = dict(vit_h="models/sam_vit_h_4b8939.pth",
                       vit_l="models/sam_vit_l_0b3195.pth",
                       vit_b="models/sam_vit_b_01ec64.pth")
    model_type = "vit_b"
    sam_checkpoint = checkpoints[model_type]
    print(f'running with {model_type}')

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    # mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=4)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # Requires open-cv to run post-processing
    )

    # image = cv2.imread('images/dog.jpg')
    image = cv2.imread('notebooks/images/dengta.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = cv2.resize(image, [image.shape[1]//5, image.shape[0]//5])
    s1 = time.time()
    masks = mask_generator.generate(image)
    s2 = time.time()
    print(f'amg time: {s2-s1:.1f}s')
    print('number of mask: ', len(masks))
    print('mask keys: ', masks[0].keys())

    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_anns(masks)
    plt.axis('off')
    plt.savefig('./test.jpg', bbox_inches='tight', pad_inches=0)
    plt.show()


if __name__ == '__main__':
    main()