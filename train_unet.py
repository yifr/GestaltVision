import os
import sys
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import torchvision
import torchvision.transforms as T

from data import gestalt
from models import unet

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
        lr = 0.5 * (max_lr) * (1 + np.cos(step / max_steps * np.pi))
        optim.lr = lr

    return lr

def train(args, train_loader, model, start_step=0):
    print_steps = args.print_steps
    eval_every = args.eval_every
    target = args.target

    def train_step(data, criterion, optimizer, train=True):
        optimizer.zero_grad()
        images = data["images"]
        target = data[args.target]
        print(images.shape, target.shape)
        loss, metrics = model.get_metrics(images, target, criterion, target_name=args.target)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        loss.backward()
        optimizer.step()
        return loss, metrics

    n_params = sum([np.prod(v.shape) for v in net.parameters()])
    print("Parameters in network:", n_params)
    print("Number training samples: ", len(train_loader))
    writer = SummaryWriter(log_dir=args.log_dir + args.run_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=args.betas, eps=args.eps
    )
    if args.target == "masks":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()

    step = start_step
    steps_per_epoch = len(train_loader)
    start_epoch = int(step / steps_per_epoch)
    for epoch in range(start_epoch, args.epochs):
        for i, batch in tqdm(enumerate(train_loader)):
            lr = learning_rate_update(optimizer, step, args.warmup_steps, args.lr, args.train_iters)
            loss, metrics = train_step(batch, criterion, optimizer)

            if (i + 1) % print_steps == 0:
                print(f"[Epoch: {epoch}] Step: {step} ==> Loss: {loss.item()}")
                model.log_metrics(writer, step, metrics, target=args.target, phase="train")

            if step % eval_every == 0:
                model.eval()

                testing_loss = []
                testing_preds = []
                testing_metrics = []
                with torch.no_grad():
                    train_loader.dataset.training = False
                    print(
                        f"Running evaluation step on {len(train_loader)} test videos..."
                    )
                    for test_step, batch in tqdm(enumerate(train_loader)):
                        # print(data.shape)
                        loss, metrics = model.get_metrics(
                            batch["images"], batch[target], criterion, target_name=args.target
                        )
                        testing_loss.append(loss.item())
                        testing_metrics.append(metrics)

                def avg(metric_list, key, dim=0):
                    return np.mean(
                        [metric[key].detach().cpu().numpy() for metric in metric_list],
                        axis=dim,
                    )

                metrics = {}
                test_loss = np.mean(testing_loss)
                metrics["loss"] = test_loss
                metrics[f"predicted_{target}"] = testing_metrics[0][
                    f"predicted_{target}"
                ]
                metrics[f"gt_{target}"] = testing_metrics[0][f"gt_{target}"]
                metrics["images"] = testing_metrics[0]["images"]
                model.log_metrics(writer, step, metrics, target=args.target, phase="test")

                print("=" * 66)
                print("=" * 30 + " EVAL " + "=" * 30)
                print(f"[Epoch: {epoch}] Eval loss ==> {test_loss}")
                print("=" * 66)
                sys.stdout.flush()
                state = {
                    "args": args,
                    "step": step,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }

                checkpoint_dir = os.path.join(args.checkpoint_dir, args.run_name)
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pt")
                print("Saving model to: ", checkpoint_path)
                torch.save(state, checkpoint_path)

                train_loader.dataset.train = True
                model.train()
            step += 1


if __name__ == "__main__":
    parser = ArgumentParser()

    # Run parameters
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/om2/user/yyf/CommonFate/scenes/",
    )
    parser.add_argument("--run_name", type=str, default="", help="Name of run")
    parser.add_argument(
        "--log_dir", type=str, default="/om2/user/yyf/GestaltVision/runs/UNet"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/om2/user/yyf/GestaltVision/saved_models/UNet",
    )

    # Dataset params
    parser.add_argument("--resize", type=int, default=128, help="input size of image")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--frames_per_scene", type=int, default=6, help="frames per scene")
    parser.add_argument("--top_level", default=["voronoi", "noise"], nargs=2, type=str, help="Texture level of dataset")
    parser.add_argument("--sub_level", default=["superquadric_2", "superquadric_3"], nargs=2, type=str, help="Object level of dataset")

    # UNet Params
    parser.add_argument("--target", type=str, default="masks", help="Target to decode")
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--num_channels", type=int, default=3)

    # Optimizer hyperparameters
    parser.add_argument("--betas", type=tuple, default=(0.9, 0.999))
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eps", type=float, default=1e-8)

    # Train / Logging hyperparameters
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--print_steps", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--load_from_last_checkpoint", action="store_true")
    parser.add_argument("--load_checkpoint", type=str, default="")

    parser.add_argument("--warmup_steps", type=int, default=2500, help="Warmup steps for learning rate")
    parser.add_argument("--grad_clip", type=float, default=0.05, help="Gradient Clipping")
    parser.add_argument(
        "--train_iters", type=int, default=10e4, help="Number of training steps"
    )


    args = parser.parse_args()

    net = unet.UNet(
        n_channels=args.num_channels,
        n_classes=args.n_classes,
    )
    print(args.top_level, args.sub_level)
    train_data = DataLoader(
        gestalt.Gestalt(args.data_dir, frames_per_scene=args.frames_per_scene,
                        top_level=args.top_level,
                        sub_level=args.sub_level,
                        passes=["images", args.target],
                        train_split=0.9,
                        resolution=(args.resize, args.resize)),
        batch_size=args.batch_size,
        shuffle=True,
    )

    start_step = 0
    if args.load_from_last_checkpoint or args.load_checkpoint:
        if args.load_from_last_checkpoint:

            def get_last_checkpoint(checkpoint_dir, run_name):
                path = os.path.join(checkpoint_dir, run_name)
                if not os.path.exists(path):
                    return None
                files = os.listdir(path)
                files = [f for f in files if f.startswith("checkpoint")]
                files.sort(key=lambda x: int(x.split(".")[0].split("_")[-1]))
                return os.path.join(path, files[-1])

            checkpoint_path = get_last_checkpoint(args.checkpoint_dir, args.run_name)
            if checkpoint_path is None:
                print("No checkpoint found. Exiting...")
                sys.exit(1)
        else:
            checkpoint_path = args.load_checkpoint

        print("Loading model from: ", checkpoint_path)

        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint["state_dict"])
        start_step = checkpoint.get("step", 0)

    train(args, train_data, net, start_step=start_step)
