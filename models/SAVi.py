import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from attention import MultiheadAttention, SlotAttention


class Predictor(nn.Module):
    def __init__(self, input_dim, num_heads, dim_feedforward, dropout=0.0):
        """
        Inputs:
            input_dim - Dimensionality of the input
            num_heads - Number of heads to use in the attention block
            dim_feedforward - Dimensionality of the hidden layer in the MLP
            dropout - Dropout probability to use in the dropout layers
        """
        super().__init__()

        # Attention layer
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads)

        # Two-layer MLP
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim),
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention part
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # MLP part
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x


def build_grid(resolution):
    ranges = [np.linspace(0.0, 1.0, num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1))


class SoftPositionEmbedding(nn.Module):
    """Adds soft positional embedding with learnable projection."""

    def __init__(self, hidden_size, resolution):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.embedding = nn.Linear(4, hidden_size, bias=True)
        self.grid = build_grid(resolution)

    def forward(self, inputs):
        grid = self.embedding(self.grid)
        return inputs + grid


def spatial_broadcast(slots, broadcast_dims):
    """
    Spatial broadcast

    Args:
        slots: slots to be broadcasted
        broadcast_dims: shape to broadcast to
    """
    slots = slots.reshape((-1, slots.shape[-1]))[:, None, None, :]
    slots = torch.tile(slots, (1, broadcast_dims[0], broadcast_dims[1], 1))
    return slots


class Encoder(nn.Module):
    def __init__(self, hid_dim, resolution):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(3, hid_dim, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(hid_dim, hid_dim, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(hid_dim, hid_dim, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(hid_dim, hid_dim, kernel_size=5, stride=1, padding=2)
        self.pos_encoder = SoftPositionEmbedding(hid_dim, resolution)
        self.layer_norm = nn.LayerNorm(hid_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hid_dim, hid_dim), nn.ReLU(), nn.Linear(hid_dim, hid_dim)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pos_encoder(x)
        x = torch.flatten(x, 1, 2)
        x = self.layer_norm(x)
        x = self.mlp(x)

        return x


class Decoder(nn.Module):
    def __init__(self, hid_dim, resolution):
        super(Decoder, self).__init__()

        self.conv1 = nn.ConvTranspose2d(hid_dim, hid_dim, kernel_size=5, stride=2)
        self.conv2 = nn.ConvTranspose2d(hid_dim, hid_dim, kernel_size=5, stride=2)
        self.conv3 = nn.ConvTranspose2d(hid_dim, hid_dim, kernel_size=5, stride=2)
        self.conv4 = nn.ConvTranspose2d(hid_dim, hid_dim, kernel_size=5, stride=2)
        self.conv5 = nn.ConvTranspose2d(hid_dim, hid_dim, kernel_size=5, stride=1)
        self.conv6 = nn.ConvTranspose2d(hid_dim, 4, kernel_size=3, stride=1)
        self.decoder_initial_state = (8, 8)
        self.pos_decoder = SoftPositionEmbedding(hid_dim, self.decoder_initial_state)
        self.resolution = resolution

    def forward(self, x):
        x = self.pos_decoder(x)
        x = x.permute(0, 3, 1, 2)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        x = self.conv6(x)
        x = x[:, :, : self.resolution[0], : self.resolution[1]]
        x = x.permute(0, 2, 3, 1)

        return x


class Initializer(nn.Module):
    """
    Provides slot initialization for segmentation mask conditioning signals
    """

    def __init__(self, hid_dim, resolution):
        super(Initializer, self).__init__()

        self.conv1 = nn.Conv2d(3, hid_dim, kernel_size=5, stride=2)
        self.conv2 = nn.Conv2d(hid_dim, hid_dim, kernel_size=5, stride=2)
        self.conv3 = nn.Conv2d(hid_dim, hid_dim, kernel_size=5, stride=2)
        self.conv4 = nn.Conv2d(hid_dim, hid_dim, kernel_size=5, stride=1)
        self.pos_embed = SoftPositionEmbedding(hid_dim, resolution)
        self.mlp1 = nn.Sequential(
            nn.Conv2d(hid_dim, hid_dim * 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hid_dim * 2, hid_dim * 2, kernel_size=1),
        )

        self.mlp2 = nn.Sequential(
            nn.Linear(hid_dim * 2, hid_dim * 4),
            nn.ReLU(),
            nn.Linear(hid_dim * 4, hid_dim * 2),
        )

        self.layer_norm1 = nn.LayerNorm(hid_dim)
        self.layer_norm2 = nn.LayerNorm(hid_dim * 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = self.pos_embed(x)
        x = self.layer_norm1(x)
        x = self.mlp1(x)
        x = self.layer_norm2(x)
        x = self.mlp2(x)
        return x


class SlotAttentionImages(nn.Module):
    def __init__(
        self,
        hid_dim=64,
        resolution=(128, 128),
        num_slots=8,
        slot_dim=64,
        slot_iterations=3,
    ):
        super(SlotAttentionImages, self).__init__()

        self.encoder = Encoder(hid_dim, resolution)
        self.decoder = Decoder(hid_dim, resolution)

        self.slot_attention = SlotAttention(
            slot_iterations, num_slots, slot_dim, slot_dim * 2
        )

    def forward(self, image):
        x = self.encoder(image)

        slots = self.slot_attention(x)
        slots = spatial_broadcast(slots, (8, 8))
        x = self.decoder(slots)

        # Undo combination of slot and batch dimension; split alpha masks.
        recons, masks = x.reshape(
            image.shape[0], -1, x.shape[1], x.shape[2], x.shape[3]
        ).split([3, 1], dim=-1)
        # `recons` has shape: [batch_size, num_slots, width, height, num_channels].
        # `masks` has shape: [batch_size, num_slots, width, height, 1].

        # Normalize alpha masks over slots.
        masks = nn.Softmax(dim=1)(masks)
        recon_combined = torch.sum(recons * masks, dim=1)  # Recombine image.
        recon_combined = recon_combined.permute(0, 3, 1, 2)
        # `recon_combined` has shape: [batch_size, width, height, num_channels].

        return recon_combined, recons, masks, slots


class SlotAttentionVideo(nn.Module):
    def __init__(
        self,
        hid_dim=64,
        resolution=(128, 128),
        num_slots=8,
        slot_dim=64,
        slot_iterations=3,
    ):
        super(SlotAttentionVideo, self).__init__()

        self.encoder = Encoder(hid_dim, resolution)
        self.decoder = Decoder(hid_dim, resolution)
        self.initializer = Initializer(hid_dim, resolution)
        self.predictor = Predictor(hid_dim, 4, 256)
        self.corrector = SlotAttention(
            slot_iterations, num_slots, slot_dim, slot_dim * 2
        )

    def forward(self, images, cues=None):
        if cues:
            slot_initialization = self.initializer(cues)
        else:
            slot_initialization = None

        # Encode frames
        B, T, C, H, W = images.shape
        preds = []
        for t in range(T):
            image = images[:, t]
            x = self.encoder(image)
            slots = self.corrector(x, slot_initialization)
            slot_initialization = self.predictor(slots)
            slots = spatial_broadcast(slots, (8, 8))
            x = self.decoder(slots)

            # Undo combination of slot and batch dimension; split alpha masks.
            recons, masks = x.reshape(B, -1, x.shape[1], x.shape[2], x.shape[3]).split(
                [3, 1], dim=-1
            )
            # `recons` has shape: [batch_size, num_slots, width, height, num_channels].
            # `masks` has shape: [batch_size, num_slots, width, height, 1].

            # Normalize alpha masks over slots.
            masks = nn.Softmax(dim=1)(masks)
            recon_combined = torch.sum(recons * masks, dim=1)  # Recombine image.
            recon_combined = recon_combined.permute(0, 3, 1, 2)
            # `recon_combined` has shape: [batch_size, width, height, num_channels].

            preds.append(
                {
                    "recon_combined": recon_combined,
                    "recons": recons,
                    "masks": masks,
                    "slots": slots,
                }
            )

        return recon_combined, recons, masks, slots
