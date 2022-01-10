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


def train(args, train_loader, test_loader, model, start_step=0):
    print_steps = args.print_steps
    eval_every = args.eval_every
    target = args.target

    def train_step(data, criterion, optimizer, train=True):
        optimizer.zero_grad()
        images = data["images"]
        masks = data[target]
        loss, metrics = model.get_metrics(images, masks, criterion)
        loss.backward()
        optimizer.step()
        return loss, metrics

    n_params = sum([np.prod(v.shape) for v in net.parameters()])
    print("Parameters in network:", n_params)
    writer = SummaryWriter(log_dir=args.log_dir + args.run_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=args.betas, eps=args.eps
    )
    criterion = nn.CrossEntropyLoss()

    step = start_step
    steps_per_epoch = len(train_loader)
    start_epoch = int(step / steps_per_epoch)
    for epoch in range(start_epoch, args.epochs):
        for i, batch in tqdm(enumerate(train_loader)):
            loss, metrics = train_step(batch, criterion, optimizer)

            if (i + 1) % print_steps == 0:
                print(f"[Epoch: {epoch}] Step: {step} ==> Loss: {loss.item()}")
                model.log_metrics(writer, step, metrics, "train")

            if step % eval_every == 0:
                model.eval()

                testing_loss = []
                testing_preds = []
                testing_metrics = []
                with torch.no_grad():
                    print(
                        f"Running evaluation step on {len(test_loader)} test videos..."
                    )
                    for test_step, batch in tqdm(enumerate(test_loader)):
                        # print(data.shape)
                        loss, metrics = model.get_metrics(
                            batch["images"], batch[target], criterion
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
                model.log_metrics(writer, step, metrics, "test")

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

                checkpoint_dir = os.path.join(args.save_path, args.run_name)
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{step}.pt")
                print("Saving model to: ", checkpoint_path)
                torch.save(state, checkpoint_path)

                model.train()
            step += 1


if __name__ == "__main__":
    parser = ArgumentParser()

    # Run parameters
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/om2/user/yyf/CommonFate/scenes/gestalt_masks_multiscene",
    )
    parser.add_argument("--run_name", type=str)
    parser.add_argument(
        "--log_dir", type=str, default="/om2/user/yyf/GestaltVision/UNet/runs/"
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="/om2/user/yyf/GestaltVision/UNet/saved_models/",
    )
    parser.add_argument("--load_from_last_checkpoint", action="store_true")
    parser.add_argument("--load_checkpoint", type=str, default="")

    # FitVid hyperparameters
    parser.add_argument("--kl_beta", type=float, default=1.0)
    parser.add_argument("--g_dim", type=int, default=16)
    parser.add_argument("--rnn_size", type=int, default=32)

    # UNet Params
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--num_channels", type=int, default=1)

    # Optimizer hyperparameters
    parser.add_argument("--betas", type=tuple, default=(0.9, 0.999))
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=1e-8)

    # Train / Logging hyperparameters
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--print_steps", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=20)

    args = parser.parse_args()

    net = unet.UNet(
        n_channels=args.num_channels,
        n_classes=args.n_classes,
        trilinear=True,
    )

    train_data = DataLoader(
        gestalt.Gestalt(args.data_dir, 16, train_split=0.9, train=True),
        batch_size=1,
        shuffle=True,
    )

    start_step = 0
    if args.load_from_last_checkpoint or args.load_checkpoint:
        if args.load_from_last_checkpoint:

            def get_last_checkpoint(save_path, run_name):
                path = os.path.join(save_path, run_name)
                if not os.path.exists(path):
                    return None
                files = os.listdir(path)
                files = [f for f in files if f.startswith("checkpoint")]
                files.sort(key=lambda x: int(x.split(".")[0].split("_")[-1]))
                return os.path.join(path, files[-1])

            checkpoint_path = get_last_checkpoint(args.save_path, args.run_name)
            if checkpoint_path is None:
                print("No checkpoint found. Exiting...")
                sys.exit(1)
        else:
            checkpoint_path = args.load_checkpoint

        print("Loading model from: ", checkpoint_path)

        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint["state_dict"])
        start_step = checkpoint.get("step", 0)

    train(args, train_data, test_data, net, start_step=start_step)
