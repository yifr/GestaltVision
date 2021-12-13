import os
import sys
import time
import torch
import pprint
from tqdm import tqdm
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="./runs")


# def init_logger(project, config, model, run_name="", log_freq=5, notes=None, tags=None):
#     wandb.init(
#         project=project,
#         config=config,
#         notes="",
#         tags=tags,
#         settings=wandb.Settings(start_method="fork"),
#     )
#     wandb.watch(model, log_freq=log_freq)
#     if run_name:
#         wandb.run.name = run_name


def train(dataset, model, epochs, config):
    meta_config = config.get("meta")
    experiment_config = config.get("experiment")
    batch_size = experiment_config.get("batch_size")
    learning_rate = experiment_config.get("lr")

    project = meta_config.get("project")
    run_name = meta_config.get("run_name")
    log_freq = meta_config.get("log_freq", 5)
    save_freq = meta_config.get("save_freq", 5)
    save_dir = meta_config.get("save_dir")
    model_save_dir = os.path.join(save_dir, model.name)
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

    notes = meta_config.get("notes")
    tags = meta_config.get("tags")

    # init_logger(project, config, model, run_name, log_freq, notes, tags)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)

    for epoch in range(epochs):
        epoch_start_time = time.time()

        running_loss = 0
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("./runs/iodine"),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            for i, data in tqdm(enumerate(dataset)):

                data = data.to(device)
                print(data.shape)
                optimizer.zero_grad()
                losses = model(data)
                total_loss = losses.get("total_loss")
                total_loss.backward()
                optimizer.step()

                running_loss += total_loss.item()

                if (i + 1) % log_freq == 0:
                    print(
                        f"[Epoch: {epoch}, iter: {i}] Running Loss: {running_loss / i}"
                    )
                    print(
                        "\n\t".join(
                            [f"{loss}: {val.item()}" for loss, val in losses.items()]
                        )
                    )
                    sys.stdout.flush()
                    writer.add_scalars(losses)
                    prof.step()

        latest = os.path.join(model_save_dir, "latest.pt")
        torch.save(model, model_save_dir)

        if (epoch + 1) % save_freq == 0:
            chckpt = os.path.join(model_save_dir, f"checkpoint_{epoch}.pt")
            torch.save(model, chckpt)

        epoch_end_time = time.time()
        print("=" * 80)
        print(
            f"[Epoch {epoch}] Average Loss: {running_loss / len(dataset)}, Total time taken: {epoch_end_time - epoch_start_time}."
        )

