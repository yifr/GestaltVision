import os
import time
import tqdm
import torch
import wandb
import pprint


def init_logger(
    project, config, model, run_name="", log_freq=100, notes=None, tags=None
):
    wandb.init(project=project, config=config, notes="", tags=tags)
    wandb.watch(model, log_freq=log_freq)
    if run_name:
        wandb.run.name = run_name
        wandb.run.save()


def train(dataset, model, epochs, config):
    meta_config = config.get("meta")
    experiment_config = config.get("experiment")

    project = meta_config.get("project")
    run_name = meta_config.get("run_name")
    log_freq = meta_config.get("log_freq", 10)
    notes = meta_config.get("notes")
    tags = meta_config.get("tags")

    init_logger(project, experiment_config, model, run_name, log_freq, notes, tags)

    device = experiment_config.get("device", "cuda")

    model.train()
    for epoch in range(epochs):
        epoch_start_time = time.time()

        running_loss = 0
        for i, data in tqdm(dataset):
            iter_start_time = time.time()

            data = data.to(device)
            losses = model.compute_loss(data)
            total_loss = losses.get("total_loss")
            running_loss += total_loss

            if i % (log_freq + 1) == 0:
                iter_end_time = time.time()
                print(
                    f"[Epoch: {epoch}, iter: {i}] Running Loss: {running_loss / i}, Time per iteration: {iter_end_time - iter_start_time}."
                )
                pprint.pprint(losses)
                wandb.log(losses)

        epoch_end_time = time.time()
        print("=" * 80)
        print(
            f"[Epoch {epoch}] Total Loss: {running_loss / i}, Total time taken: {epoch_end_time - epoch_start_time}."
        )


