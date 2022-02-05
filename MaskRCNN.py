import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from matplotlib.patches import Rectangle

import matplotlib.cm as cm
import cv2

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def mask_mIOU(gt_masks, pred_masks):
    """ Mean Intersection over Union metric

    Params:
        gt_masks: np.ndarray or torch.Tensor containing B x N_masks x C x H x W
        pred_masks: same as gt_masks

    Returns:
        mIOU metric for given masks
    """
    num_gt_px = gt_masks.sum()
    num_pred_px = pred_masks.sum()
    if num_gt_px < 1:
        return np.nan

    intersection = (pred_masks & gt_masks).sum().astype(float)
    IoU = intersection / (num_gt_px + num_pred_px - intersection)
    return IoU

def mask_recall(gt_masks, pred_masks):
    num_gt_px = gt_masks.sum()
    num_pred_px = pred_masks.sum()
    if num_gt_px < 1:
        return np.nan

    intersection = (pred_masks & gt_masks).sum().astype(float)
    recall = intersection / np.maximum(num_gt_px, 1.0)
    return recall

def mask_precision(gt_masks, pred_masks):
    num_gt_px = gt_masks.sum()
    num_pred_px = pred_masks.sum()
    if num_gt_px < 1:
        return np.nan

    intersection = (pred_masks & gt_masks).sum().astype(float)
    precision = intersection / np.maximum(num_pred_px, 1.0)
    return precision

from notebooks.engine import train_one_epoch, evaluate
import utils


def main(model, save_path):
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = gestalt.MaskRCNNLoader(root_dir="/om2/user/yyf/CommonFate/scenes",
                              frames_per_scene=1, top_level=["train_voronoi", "train_noise"],
                              sub_level=["superquadric_1", "superquadric_2", "superquadric_3"],
                              passes=["images", "bounding_boxes", "masks"])

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=1,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=1,
        collate_fn=utils.collate_fn)



    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, save_path, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)
        torch.save({"weights": fine_tune_maskrcnn.state_dict()}, save_path)

    print("That's it!")

import random

def get_coloured_mask(mask, i):
    """
    random_colour_masks
      parameters:
        - image - predicted masks
      method:
        - the masks of each predicted object is given random colour for visualization
    """
    colours = [[10, 255, 10, 1],[10, 10, 255, 1],[255, 10, 10, 1],[10, 255, 255, 1],[255, 255, 10, 1],[255, 10, 255, 1],
               [80, 70, 180, 1],[250, 80, 190, 1],[245, 145, 50, 1],[70, 150, 250, 1],[50, 190, 190, 1]]

    i = i % len(colours)

    r = np.zeros_like(mask).astype(np.uint8)
    g = np.zeros_like(mask).astype(np.uint8)
    b = np.zeros_like(mask).astype(np.uint8)
    r[mask == 1], g[mask == 1], b[mask == 1] = colours[i][:-1]
    coloured_mask = np.stack([r, g, b], axis=2)
    return coloured_mask


def get_prediction(pred, confidence=0.6):
    """
    get_prediction
      parameters:
        - img_path - path of the input image
        - confidence - threshold to keep the prediction or not
      method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - masks, classes and bounding boxes are obtained from the model and soft masks are made binary(0 or 1) on masks
          ie: eg. segment of cat is made 1 and rest of the image is made 0

    """
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > confidence]
    if not pred_t:
        cutoff = -1
    else:
        cutoff = [pred_score.index(x) for x in pred_score if x > confidence][-1]

    masks = (pred[0]['masks'] > 0.5).squeeze().detach().cpu().numpy()

    if len(masks.shape) < 3:
        masks = np.expand_dims(masks, 0)

    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())]
    masks = masks[:cutoff + 1]
    pred_boxes = pred_boxes[:cutoff + 1]

    return masks, pred_boxes


def plot_results(images, pred_boxes, masks, normals):
    fig, ax = plt.subplots(1, 3, figsize=(20, 40))

    i = 0

    ax[1].imshow(images, cmap=cm.gray)
    ax[2].imshow(images, cmap=cm.gray)

    mask = images
    for i, mask_ in enumerate(masks):
        rgb_mask = get_coloured_mask(mask_, i) / 255.0
        rgb_mask = np.ma.masked_where(rgb_mask == 0, rgb_mask)
        mask = np.where(rgb_mask > 0, rgb_mask, mask)

    ax[2].imshow(mask)

    for box in pred_boxes:
        xy = (box[0], box[1])
        h, w = box[1][0] - box[0][0], box[1][1] - box[0][1]
        rect = Rectangle(box[0], h, w, linewidth=5, edgecolor="r", facecolor="none", alpha=0.75)
        ax[2].add_patch(rect)

    mask = np.zeros(images.shape)
    for i, mask_ in enumerate(normals):
        rgb_mask = get_coloured_mask(mask_, i) / 255.0
        rgb_mask = np.ma.masked_where(rgb_mask == 0, rgb_mask)
        mask = np.where(rgb_mask > 0, rgb_mask, mask)

    ax[0].imshow(mask)

    ax[0].set_title("Ground Truth Masks")
    ax[1].set_title("Input Image")
    ax[2].set_title("MaskRCNN Bounding Boxes + Mask")

    plt.show()


def get_metrics_and_figures(model):
    model.eval()
    model.to("cuda")
    mIOU = []
    plot = False

    metrics = {}
    test_data = gestalt.MaskRCNNLoader(top_level=["test_voronoi", "test_noise", "test_wave"],
                                        passes=["images", "masks", "normals"],
                                        frames_per_scene=1
                                        )

    print("="*60)
    print(" "*20 + "Running Evaluation")
    print("=" * 60)

    # Plot the first frame of each movie
    for i in range(len(test_data)):
        if i % 64 == 0:
            plot = True
        else:
            plot = False

        batch = test_data.__getitem__(i)
        images = batch[0].to("cuda")
        scene = batch[1]["scene_dir"] + f"/{i % 64:02d}"
        metrics[scene] = {}

        output = model(images)


        # Get outputs for plotting
        pred_masks, pred_boxes = get_prediction(output)
        images = images.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
        gt_masks = batch[1]["masks"].cpu().squeeze().numpy()

        if len(gt_masks.shape) < 3:
            gt_masks = np.expand_dims(gt_masks, 0)

        # Pad predicted masks if necessary
        if pred_masks.shape[0] < gt_masks.shape[0]:
            diff = gt_masks.shape[0] - pred_masks.shape[0]
            padding = np.zeros((diff, 128, 128))
            pred_masks = np.append(pred_masks, padding, axis=0)


        # Compute metrics
        n_gt_objs = gt_masks.shape[0]
        mIOU = mask_mIOU(gt_masks.sum(0).astype(np.uint8), pred_masks.sum(0).astype(np.uint8))
        recall = mask_recall(gt_masks.sum(0).astype(np.uint8), pred_masks.sum(0).astype(np.uint8))
        precision = mask_precision(gt_masks.sum(0).astype(np.uint8), pred_masks.sum(0).astype(np.uint8))
        metrics[scene]["precision"] = precision
        metrics[scene]["recall"] = recall
        metrics[scene]["mIOU"] = mIOU

        print("\n")
        print(scene)
        print(f"mIOU: {mIOU}, \tPrecision: {precision}, \tRecall: {recall}")
        if plot:
            plot_results(images,
                 pred_boxes=pred_boxes,
                 masks=pred_masks,
                 normals=gt_masks,
                )

    return metrics

def compute_test_set_extended_metrics(metrics):
    mIOU = 0
    precision = 0
    recall = 0
    for scene, metric in metrics.items():
        mIOU += metric["mIOU"]
        precision += metric["precision"]
        recall += metric["recall"]

    mIOU /= len(metrics)
    precision /= len(metrics)
    recall /= len(metrics)

    print(f"mIOU: {mIOU}, \tPrecision: {precision}, \tRecall: {recall}")
    return mIOU, precision, recall

def compute_test_set_metrics(metrics):
    mIOU = 0
    precision = 0
    recall = 0
    total = 0
    for scene, metric in metrics.items():
        if "wave" in scene:
            continue

        mIOU += metric["mIOU"]
        precision += metric["precision"]
        recall += metric["recall"]
        total += 1

    mIOU /= total
    precision /= total
    recall /= total

    print(f"mIOU: {mIOU}, \tPrecision: {precision}, \tRecall: {recall}")
    return mIOU, precision, recall

if __name__=="__main__":
    model = get_model_instance_segmentation(2, pretrained=False)
    save_path = "/om2/user/yyf/GestaltVision/saved_models/MaskRCNN/train-on-test.pt"
    main(model, save_path)

    metrics = get_metrics_and_figures(model)
    test_set_metrics = compute_test_set_extended_metrics(metrics)
    test_set_metrics_ext = compute_test_set_metrics(metrics)

    metrics["test_set_metrics"] = test_set_metrics
    metrics["test_set_metrics_ext"] = test_set_metrics_ext

    base = "/om2/user/yyf/GestaltVision/figures/MaskRCNN"
    with open(base + "/train-on-test-eval-set.json", "w") as f:
        json.dump(metrics, f)

