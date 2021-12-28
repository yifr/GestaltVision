import torch
from models import SAVi
from data import gestalt
from metrics import adjusted_rand_index
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm
from argparse import ArgumentParser()

parser = ArgumentParser()

# Model params
parser.add_argument("--num_slots", type=int, default=5, help="Number of slots in SAVi model")
parser.add_argument("--cue", type=str, default="masks", help="Object Cue for SAVi")

# Data params
parser.add_argument("--num_frames", type=int, default=10, help="Frames to train on")
parser.add_argument("--batch_size", type=int, default=10, help="Batch Size")

# Training params
parser.add_argument("--lr", type=float, default=1e-2, help="Learning Rate")
parser.add_argument("--train_iters", type=int, default=10e4, help="Number of training steps")
parser.add_argument("--log_every", type=int, default=50, help="How often to log losses")
parser.add_argument("--eval_every", type=int, default=1000, help="How often to run eval")

# Paths
parser.add_argument("--log_dir" type=str, default="/om2/user/yyf/GestaltVision/runs/SAVi")
parser.add_argument("--save_dir", type=str, default="/om2/user/yyf/GestaltVision/models/SAVi")
parser.add_argument("--data_dir", type=str, default="/om/user/yyf/CommonFate/scenes")

args = parser.parse_args()

def train(model, data, args):

    def train_step(data, metric, model, optim):
        optim.zero_grad()
        images = data["images"]
        flows = data["flows"]
        cue = data[args.cue][:, 0] # Only take first slice of a cue
        out = model(images, cue=cue)
        loss = metric(out["recon_combined"], flows)
        loss.backward()
        optim.step()
        return out, loss


dataloader = DataLoader(gestalt.Gestalt(root_dir="/om/user/yyf/CommonFate/scenes"))
model = SAVi.SlotAttentionVideo(num_slots=5, slot_iterations=2).float()
for i, batch in enumerate(dataloader):
    images, masks, flows = batch['images'], batch['masks'], batch['flows']
    print(images.shape, masks.shape, flows.shape)
    out = model(images, cues=masks[:, 0])
    print(out.keys())
