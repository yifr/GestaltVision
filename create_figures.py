import numpy as np
import matplotlib.pyplot as plt
import torch
import os
import sys
sys.path.append("../")
from data import gestalt
from models import SAVi
from torch.utils.data import DataLoader
from utils import make_video

model = SAVi.SlotAttentionVideo(num_slots=5)

model_checkpoint_dir = "/om2/user/yyf/GestaltVision/saved_models/SAVI/tex=all_shapes=2,3_slots=5"
models = os.listdir(model_checkpoint_dir)
models.sort()

latest = models[-1]
model_chkpt = os.path.join(model_checkpoint_dir, latest)
chkpt = torch.load(model_chkpt, map_location="cpu")
model.load_state_dict(chkpt["model"])

data = DataLoader(gestalt.Gestalt(root_dir="/om2/user/yyf/CommonFate/scenes",
                                top_level=[f"test_{tex}" for tex in ["voronoi", "wave", "noise"]],
                                  sub_level=[f"superquadric_{i}" for i in [1, 2, 3, 4]],
                                  frames_per_scene=16,
                                  train_split=0,
                                  training=False
                      ), batch_size=1)
data.training = False
model.eval()

print(len(data))


def plot_results(inputs, outputs):
    idx = inputs["scene"]
    scene_dir = inputs["scene_dir"]

    path_parts = scene_dir[0].split("/")
    tex = path_parts[-3]
    objs = path_parts[-2]
    scene_num = path_parts[-1]

    images = inputs["images"].detach().cpu().squeeze(0).numpy().transpose(0, 2, 3, 1)
    gt_flows = inputs["flows"].detach().cpu().squeeze(0).numpy().transpose(0, 2, 3, 1)
    gt_masks = inputs["masks"].detach().cpu().squeeze(0).numpy().sum(axis=1).transpose(0, 2, 3, 1)

    recons = outputs["recon_combined"].detach().cpu().squeeze(0).numpy().transpose(0, 2, 3, 1)
    pred_masks = outputs["masks"].detach().cpu().squeeze(0).numpy()
    slots = outputs["recons"].detach().cpu().squeeze(0).numpy()
    slots = [slots[:, i] for i in range(slots.shape[1])]

    pred_masks = pred_masks.sum(axis=1)
    pred_masks = np.where(pred_masks > pred_masks.mean(), 1, 0)

    print(pred_masks.shape, slots[0].shape, recons.shape, gt_masks.shape, gt_flows.shape, images.shape)

    for vid in ["flows", "masks", "slots"]:
        base_title = f"savi_{tex}_{objs}_scene-{idx[0]}_{vid}"
        titles = ["Input Images", f"Ground truth {vid}", f"Predicted {vid}"]
        if vid == "flows":
            sequences = [images, gt_flows, recons]
        elif vid == "slots":
            titles = [f"Slot {i}" for i in range(len(slots))]
            sequences = slots
        else:
            sequences = [images, gt_masks, pred_masks]

        print("Generating video: ", base_title)
        make_video(sequences, titles, base_title, format="mp4", output_dir="/om2/user/yyf/GestaltVision/figures/SAVI/slots=5")


def cat_dict(d1, d2):
    keys = d1.keys()
    out = {}
    for key in keys:
        d1_v = d1[key]
        d2_v = d2[key]

        if key == "frame_idxs":
            continue
        if key == "scene" or key == "scene_dir":
            out[key] = d1_v
            continue

        out[key] = torch.cat([d1_v, d2_v], 1)
    return out

scene_idx = 0
batch_inputs = {}
batch_outputs = {}
for i, batch in enumerate(data):

    scene = batch["scene"]

    images = batch["images"].to("cuda")
    flows = batch["flows"].to("cuda")
    masks = batch["masks"].to("cuda").permute((0, 2, 1, 3, 4, 5))    # B, T, M, C, H, W
    out = model(images, cues=masks[:, 0, :])
    batch["masks"] = batch["masks"].permute(0, 2, 1, 3, 4, 5)
    out["masks"] = out["masks"].permute((0, 1, 2, 3, 4, 5)) # B, T, M, H, W, C

    if scene != scene_idx:
        scene_idx = scene
        plot_results(batch_inputs, batch_outputs)
        batch_inputs = batch
        batch_outputs = out
    elif not batch_inputs:
        batch_inputs = batch
        batch_outputs = out
    else:
        batch_inputs = cat_dict(batch_inputs, batch)
        batch_outputs = cat_dict(batch_outputs, out)

    print(batch_inputs["masks"].shape, batch_outputs["masks"].shape)
