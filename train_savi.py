import torch
from models import SAVi
from data import gestalt
from torch.utils.data import DataLoader

dataloader = DataLoader(gestalt.Gestalt(root_dir="/om/user/yyf/CommonFate/scenes"))
model = SAVi.SlotAttentionVideo().float()
for i, batch in enumerate(dataloader):
    images, masks, flows = batch['images'], batch['masks'], batch['flows']
    print(images.shape, masks.shape, flows.shape)
    out = model(images, cues=masks[:, 0])
    print(out.keys())
