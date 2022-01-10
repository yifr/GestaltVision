import os
import glob
import time
import torch
import metrics
import numpy as np
import torch.nn.functional as F

from models import SAVi
from data import gestalt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from argparse import ArgumentParser

parser = ArgumentParser()

# Model params
parser.add_argument(
    "--num_slots", type=int, default=5, help="Number of slots in SAVi model"
)
parser.add_argument(
    "--num_iterations", type=int, default=2, help="Number of iterations"
)
parser.add_argument("--cue", type=str, default="masks", help="Object Cue for SAVi")

# Data params
parser.add_argument("--num_frames", type=int, default=6, help="Frames to train on")
parser.add_argument("--batch_size", type=int, default=32, help="Batch Size")

# Training params
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate")
parser.add_argument("--warmup_steps" type=int, default=2500, help="Warmup steps for learning rate")
parser.add_argument("--grad_clip", type=float, default=0.05, help="Gradient Clipping")
parser.add_argument(
    "--train_iters", type=int, default=10e4, help="Number of training steps"
)
parser.add_argument("--log_every", type=int, default=50, help="How often to log losses")
parser.add_argument(
    "--eval_every", type=int, default=1000, help="How often to run eval"
)
parser.add_argument("--seed", type=int, default=42, help="random seed")

# Paths
parser.add_argument(
    "--plot_n_videos", type=int, default=4, help="Number of videos to plot"
)
parser.add_argument(
    "--log_dir", type=str, default="/om2/user/yyf/GestaltVision/runs/SAVi"
)
parser.add_argument(
    "--checkpoint_dir", type=str, default="/om2/user/yyf/GestaltVision/models/SAVi"
)
parser.add_argument("--data_dir", type=str, default="/om/user/yyf/CommonFate/scenes")
parser.add_argument(
    "--top_level",
    type=str,
    nargs="+",
    default=["voronoi", "noise"],
    help="texture split",
)
parser.add_argument(
    "--sub_level",
    type=str,
    nargs="+",
    default=["superquadric_1", "superquadric_2", "superquadric_3"],
    help="object split",
)

parser.add_argument(
    "--load_latest_model",
    action="store_true",
    help="Continue training from latest checkpoint",
)
args = parser.parse_args()

class LRScheduler:
    def __init__(self, max_lr, max_steps, warmup_steps):
        self.max_lr = lr
        self.max_steps = max_steps
        self.warmup_steps = warmup_steps

    def update(optim, step):
        if step < self.warmup_steps:
            warmup_percent_done = step / self.warmup_steps
            lr = self.max_lr * warmup_percent_done
            optim.lr = lr
        else:
            lr = 0.5 * (self.max_lr)(1 + np.cos(step / self.max_steps * np.pi))
            optim.lr = lr

        return lr


def learning_rate_update(optim, step, warmup_steps, max_lr, max_steps):
    """ Learning Rate Scheduler with linear warmup and cosine annealing

        Params:
        ------
        optim: torch.optim:
            Torch optimizer
        step: int:
            current training step
        warmup_steps: int:
            number of warmup steps
        max_lr: float:
            maximum learning rate
        max_steps: int:
            total number of training steps

        Returns:
        --------
        Updates optimizer and returns updated learning rate
    """
    if step < warmup_steps:
        warmup_percent_done = step / warmup_steps
        lr = max_lr * warmup_percent_done
        optim.lr = lr
    else:
        lr = 0.5 * (max_lr)(1 + np.cos(step / max_steps * np.pi))
        optim.lr = lr

    return lr

def eval(model, data_loader, args, step, writer=None, save=True):
    """
    Runs eval loop on set set of data, logs results to tensorboard
    if writer is present
    Args:
        model: Model to evaluate
        data_loader: data loader
        args: args
        step: step to log
    """
    model.eval()
    data_loader.dataset.training = False

    print(f"Running Evaluation on {100} samples")
    with torch.no_grad():
        np.random.seed(args.seed)
        loss = 0
        fg_ari = 0
        mean_IOU = 0

        for i, batch in tqdm(enumerate(data_loader)):
            if i == 100:
                break
            images = batch["images"]
            flows = batch["flows"]
            if args.cue == "masks":
                cue = batch[args.cue][:, :, 0]  # Only take first time step of a cue
            else:
                cue = batch[args.cue][:, 0]
            out = model(images, cues=cue)
            pred_flows = out["recon_combined"]
            loss += F.mse_loss(flows, pred_flows).item()

            pred_masks = out["masks"].detach()
            gt_masks = (
                batch["masks"].detach().sum(dim=3, keepdim=True)
            )  # Combine RGB channels into one
            B, N, T, C, H, W = gt_masks.shape
            gt_masks = gt_masks.reshape((B, N, T, H, W, C))
            pred_masks = pred_masks.transpose(1, 2)

            pred_groups = pred_masks.reshape(args.batch_size, N, -1).permute(0, 2, 1)
            true_groups = gt_masks.reshape(args.batch_size, N, -1).permute(0, 2, 1)
            fg_ari += metrics.adjusted_rand_index(true_groups, pred_groups).mean()

            gt_masks = gt_masks[:, 1:, ...].sum(
                dim=1
            )  # Combine individual mask slots and ignore backgrounds
            pred_masks = pred_masks.sum(dim=1)
            mean_IOU += metrics.mean_IOU(gt_masks, pred_masks)

            gt_masks = gt_masks.reshape(B, T, C, H, W)
            pred_masks = pred_masks.reshape(B, T, C, H, W)

        mean_IOU /= len(data_loader)
        fg_ari /= len(data_loader)
        loss /= len(data_loader)

        if writer is not None:
            writer.add_scalar("eval/loss", loss, step)
            writer.add_scalar("eval/fg_ari", fg_ari, step)
            writer.add_scalar("eval/mean_IOU", mean_IOU, step)

            writer.add_video("eval/input_video", images[: args.plot_n_videos], step)
            writer.add_video("eval/pred_flow", pred_flows[: args.plot_n_videos], step)
            writer.add_video("eval/gt_flow", flows[: args.plot_n_videos], step)
            writer.add_video("eval/pred_masks", pred_masks[: args.plot_n_videos], step)
            writer.add_video("eval/gt_masks", gt_masks[: args.plot_n_videos], step)

        print("=" * 30 + " EVALUATION " + "=" * 30)
        print("Step: {}, Eval Loss: {}".format(step, i, loss))
        print("\tEval FG-ARI: {}, Eval Mean IOU: {}".format(fg_ari, mean_IOU))
        print("=" * 72)

    dataloader.dataset.training = True
    model.train()
    return


def train(model, data_loader, args, step=0):
    def train_step(batch, metric, model, optim):
        optim.zero_grad()
        images = batch["images"]
        flows = batch["flows"]
        if args.cue == "masks":
            # shape is B x max_num_objects x T x C x H x W
            cue = batch[args.cue][:, :, 0, ...]  # Only take first time step of a cue
        else:
            cue = batch[args.cue][:, 0]

        out = model(images, cues=cue)
        loss = metric(out["recon_combined"], flows)
        # torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        loss.backward()
        optim.step()

        return out, loss.item()

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    metric = F.mse_loss
    writer = SummaryWriter(args.log_dir)

    while step < args.train_iters:
        start = time.time()
        for i, batch in enumerate(tqdm(data_loader)):
            batch_load = time.time()
            lr = learning_rate_update(optim, step, args.warmup_steps, args.lr, args.train_iters)
            out, loss = train_step(batch, metric, model, optim)
            if i % args.log_every == 0:
                print(f"Time taken to load data: {batch_load - start}")
                # Reshape masks for foreground ari metric
                pred_masks = out["masks"].detach()
                gt_masks = (
                    batch["masks"].detach().sum(dim=3, keepdim=True)
                )  # Combine RGB channels into one
                B, N, T, C, H, W = gt_masks.shape
                gt_masks = gt_masks.reshape((B, N, T, H, W, C))
                pred_masks = pred_masks.transpose(1, 2)

                pred_groups = pred_masks.reshape(args.batch_size, N, -1).permute(
                    0, 2, 1
                )
                true_groups = gt_masks.reshape(args.batch_size, N, -1).permute(0, 2, 1)
                fg_ari = metrics.adjusted_rand_index(true_groups, pred_groups)

                gt_masks = gt_masks[:, 1:, ...].sum(
                    dim=1
                )  # Combine individual mask slots and ignore backgrounds
                pred_masks = pred_masks.sum(dim=1)
                mean_IOU = metrics.mean_IOU(gt_masks, pred_masks)

                gt_masks = gt_masks.reshape(B, T, C, H, W)
                pred_masks = pred_masks.reshape(B, T, C, H, W)
                writer.add_scalar("train/loss", loss, step)
                writer.add_scalar("train/fg-ARI", fg_ari.mean(), step)
                writer.add_scalar("train/mean_IOU", mean_IOU, step)

                writer.add_video(
                    "train/gt_flows", batch["flows"][: args.plot_n_videos], step
                )
                writer.add_video(
                    "train/pred_flows",
                    out["recon_combined"][: args.plot_n_videos],
                    step,
                )
                writer.add_video(
                    "train/input_video", batch["images"][: args.plot_n_videos], step
                )
                writer.add_video("train/gt_masks", gt_masks[: args.plot_n_videos], step)
                writer.add_video(
                    "train/pred_masks", pred_masks[: args.plot_n_videos], step
                )

                print("Step: {}, Loss: {}".format(step, loss))
                print("\tFG-ARI: {}, Mean IOU: {}".format(fg_ari.mean(), mean_IOU))

            if step % args.eval_every == 0:
                # eval(model, data_loader, args, step, writer)

                checkpoint = {"model": model.state_dict(), "optim": optim, "step": step}
                if not os.path.exists(args.checkpoint_dir):
                    os.makedirs(args.checkpoint_dir, exist_ok=True)
                torch.save(
                    checkpoint,
                    os.path.join(args.checkpoint_dir, "checkpoint_{}.pth".format(step)),
                )

            step += 1

    print("Reached maximum training number of training steps...")
    eval(model, data_loader, args, step, writer)

    checkpoint = {"model": model.state_dict(), "optim": optim, "step": step}
    torch.save(
        checkpoint,
        os.path.join(args.checkpoint_dir, "FINAL.pth".format(step)),
    )
    return


def load_latest(args):
    checkpoint_path = os.path.join(args.checkpoint_dir, "checkpoint_*.pth")
    if not os.path.exists(checkpoint_path):
        return None
    checkpoints = glob.glob(checkpoint_path)
    if len(checkpoints) == 0:
        return None
    checkpoints.sort()
    return torch.load(checkpoints[-1])


if __name__ == "__main__":
    dataloader = DataLoader(
        gestalt.Gestalt(
            root_dir=args.data_dir,
            top_level=args.top_level,
            sub_level=args.sub_level,
            frames_per_scene=args.num_frames,
            train_test_split=0.9,
        ),
        batch_size=args.batch_size,
        shuffle=True,
    )

    model = SAVi.SlotAttentionVideo(
        num_slots=args.num_slots, slot_iterations=args.num_iterations
    ).float()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    step = 0
    if args.load_latest_model:
        model, optim, step = torch.load(
            os.path.join(args.checkpoint_dir, "latest_model.pt")
        )
        model.load_state_dict(model)
    train(model, dataloader, args, step)
