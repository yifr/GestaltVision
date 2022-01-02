import os
import glob
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
parser.add_argument("--num_frames", type=int, default=10, help="Frames to train on")
parser.add_argument("--batch_size", type=int, default=10, help="Batch Size")

# Training params
parser.add_argument("--lr", type=float, default=1e-2, help="Learning Rate")
parser.add_argument("--grad_clip", type=float, default=0.5, help="Gradient Clipping")
parser.add_argument(
    "--train_iters", type=int, default=10e4, help="Number of training steps"
)
parser.add_argument("--log_every", type=int, default=50, help="How often to log losses")
parser.add_argument(
    "--eval_every", type=int, default=1000, help="How often to run eval"
)

# Paths
parser.add_argument(
    "--plot_n_videos", type=int, default=4, help="Number of videos to plot"
)
parser.add_argument(
    "--log_dir", type=str, default="/om2/user/yyf/GestaltVision/runs/SAVi"
)
parser.add_argument(
    "--save_dir", type=str, default="/om2/user/yyf/GestaltVision/models/SAVi"
)
parser.add_argument("--data_dir", type=str, default="/om/user/yyf/CommonFate/scenes")
parser.add_argument(
    "--load_latest_model",
    action="store_true",
    help="Continue training from latest checkpoint",
)
args = parser.parse_args()


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

    with torch.no_grad():
        np.random.seed(args.seed)
        loss = 0
        fg_ari = 0
        mean_IOU = 0

        for i, batch in enumerate(data_loader):
            images = batch["images"]
            flows = batch["flows"]
            cue = batch[args.cue][:, 0]
            out = model(images, cue=cue)
            pred_flows = out["recon_combined"]
            loss += F.mse_loss(flows, pred_flows).item()

            pred_masks = out["masks"].detach().cpu().numpy()
            gt_masks = batch["masks"].detach().cpu().numpy()
            pred_groups = pred_masks.reshape(
                args.batch_size, args.num_slots, -1
            ).permute(0, 2, 1)
            true_groups = gt_masks.reshape(args.batch_size, args.num_slots, -1).permute(
                0, 2, 1
            )
            fg_ari += metrics.adjusted_rand_index(true_groups, pred_groups)

            gt_masks = batch["masks"].sum(dim=1)  # Combine individual mask slots
            pred_masks = out["masks"].sum(dim=1)
            mean_IOU += metrics.mean_IOU(gt_masks, pred_masks)

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
    def train_step(data, metric, model, optim):
        optim.zero_grad()
        images = data["images"]
        flows = data["flows"]
        cue = data[args.cue][:, 0]  # Only take first time step of a cue
        out = model(images, cue=cue)
        loss = metric(out["recon_combined"], flows)
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()
        return out, loss.item()

    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    metric = F.mse_loss
    writer = SummaryWriter(args.log_dir)

    for epoch in range(1, args.epochs + 1):
        for i, batch in enumerate(tqdm(data_loader)):
            out, loss = train_step(batch, metric, model, optim)
            if i % args.log_every == 0:
                # Reshape masks for foreground ari metric
                pred_masks = out["masks"].detach().cpu().numpy()
                gt_masks = batch["masks"].detach().cpu().numpy()
                pred_groups = pred_masks.reshape(
                    args.batch_size, args.num_slots, -1
                ).permute(0, 2, 1)
                true_groups = gt_masks.reshape(
                    args.batch_size, args.num_slots, -1
                ).permute(0, 2, 1)
                fg_ari = metrics.adjusted_rand_index(true_groups, pred_groups)

                gt_masks = batch["masks"].sum(dim=1)  # Combine individual mask slots
                pred_masks = out["masks"].sum(dim=1)
                mean_IOU = metrics.mean_IOU(gt_masks, pred_masks)

                writer.add_scalar("train/loss", loss, step)
                writer.add_scalar("train/fg-ARI", fg_ari, step)
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
                    "train/input_video", out["images"][: args.plot_n_videos], step
                )
                writer.add_video("train/gt_masks", gt_masks[: args.plot_n_videos], step)
                writer.add_video(
                    "train/pred_masks", pred_masks[: args.plot_n_videos], step
                )

                print("Step: {}, Loss: {}".format(step, loss))
                print("\tFG-ARI: {}, Mean IOU: {}".format(fg_ari, mean_IOU))

            if step % args.eval_every == 0:
                eval(model, data_loader, args, step, writer)

                checkpoint = {"model": model.state_dict(), "optim": optim, "step": step}
                torch.save(
                    checkpoint,
                    os.path.join(args.checkpoint_dir, "checkpoint_{}.pth".format(step)),
                )

            step += 1


def load_latest(args):
    checkpoint_path = os.path.join(args.save_dir, "checkpoint_*.pth")
    if not os.path.exists(checkpoint_path):
        return None
    checkpoints = glob.glob(checkpoint_path)
    if len(checkpoints) == 0:
        return None
    checkpoints.sort()
    return torch.load(checkpoints[-1])


if __name__ == "__main__":
    dataloader = DataLoader(
        gestalt.Gestalt(root_dir=args.data_dir, frames_per_scene=args.num_frames),
        batch_size=args.batch_size,
        shuffle=True,
    )

    model = SAVi.SlotAttentionVideo(
        num_slots=args.num_slots, slot_iterations=args.num_iterations
    ).float()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataloader.to(device)
    model.to(device)

    step = 0
    if args.load_latest_model:
        model, optim, step = torch.load(os.path.join(args.save_dir, "latest_model.pt"))
        model.load_state_dict(model)
    train(model, dataloader, args, step)
