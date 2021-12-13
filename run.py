import os
import json
import argparse
from models import models
import train
from data import clevr
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument(
    "--config", type=str, required=True, help="Path to config.json file"
)

args = parser.parse_args()

config_path = args.config
with open(config_path, "rb") as f:
    config = json.load(f)

experiment_config = config.get("experiment")
model_name = experiment_config.get("model")
model_params = experiment_config.get("model_params")
load_existing = experiment_config.get("load_existing")
model = models.get_model(model_name, model_params, load_existing)

data_config = experiment_config.get("data_config")
data_path = data_config.get("data_path")
dataset = DataLoader(
    clevr.CLEVR(data_path),
    batch_size=experiment_config.get("batch_size", 32),
)

train.train(dataset, model, 100, config)
