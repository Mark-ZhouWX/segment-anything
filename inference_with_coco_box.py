import os
import time

import numpy as np
import matplotlib.pyplot as plt
import cv2



def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def main():
    from segment_anything import sam_model_registry, SamPredictor
    checkpoints = dict(vit_h="models/sam_vit_h_4b8939.pth",
                       vit_l="models/sam_vit_l_0b3195.pth",
                       vit_b="models/sam_vit_b_01ec64.pth")
    model_type = "vit_b"
    sam_checkpoint = checkpoints[model_type]
    print(f'running with {model_type}')

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    image = cv2.imread('/data1/detrgroup/datasets/coco_yolo/images/train2017/000000218926.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    out_dir = 'work_dirs'
    os.makedirs(out_dir, exist_ok=True)

    s = time.time()
    predictor.set_image(image)
    e = time.time()
    print(f'image encode time: {e-s:.1f}s')

    # predict_with_point(predictor, image)
    predict_with_box(predictor, image)
    # predict_with_point_and_box(predictor, image)

def predict_with_point(predictor, image):
    # predict the first point
    input_point = np.array([[500, 375]])
    input_label = np.array([1])

    s1 = time.time()
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )
    s2 = time.time()
    print(f'prompting with points takes: {s2-s1:.2f}s')
    print(f'out shape: mask {masks.shape}, logits {logits.shape}, scores {scores.shape}')
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca())
        show_points(input_point, input_label, plt.gca())
        plt.title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()
        # plt.savefig(os.path.join(out_dir, f'mask_({i+1}).png'))

    # predict the second and third points
    input_point = np.array([[500, 375], [1125, 625]])
    input_label = np.array([1, 0])

    mask_input = logits[np.argmax(scores), :, :]  # Choose the model's best mask
    print(f'mask input shape {mask_input.shape}')
    s3 = time.time()
    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        mask_input=mask_input[None, :, :],
        multimask_output=False,
    )
    s4 = time.time()
    print(f'prompting with mask tasks: {s4-s3:.2f}s')

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(masks, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.axis('off')
    plt.show()


def predict_with_box(predictor, image):
    input_box = np.array([51.5,   31.24, 290.11, 363.88])
    s3 = time.time()
    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )
    s4 = time.time()
    print(f'prompting with box tasks: {s4-s3:.2f}s')
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(masks[0], plt.gca())
    show_box(input_box, plt.gca())
    plt.axis('off')
    plt.show()


def predict_with_point_and_box(predictor, image):
    input_box = np.array([425, 600, 700, 875])
    input_point = np.array([[575, 750]])
    input_label = np.array([0])

    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        box=input_box,
        multimask_output=False,
    )

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    show_mask(masks[0], plt.gca())
    show_box(input_box, plt.gca())
    show_points(input_point, input_label, plt.gca())
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()