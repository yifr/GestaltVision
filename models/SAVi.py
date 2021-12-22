import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SlotAttention(nn.Module):
    def __init__(
        self, num_iterations, num_slots, slot_dim, mlp_hidden_dim, epsilon=1e-8
    ):
        """
        Slot attention
        Params:
            num_iterations: int: how many iterations to pay attention over
            num_slots: int: how many slots to initialize
            slot_dim: int: dimensions of each slot
            mlp_hidden_dim: int: size of hidden layer in mlp
        """
        super(SlotAttention, self).__init__()

        self.num_iterations = num_iterations
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.mlp_hidden_dim = mlp_hidden_dim
        self.epsilon = epsilon

        self.queries = nn.Linear(slot_dim, slot_dim)
        self.keys = nn.Linear(slot_dim, slot_dim)
        self.values = nn.Linear(slot_dim, slot_dim)

        self.input_norm = nn.LayerNorm(slot_dim)
        self.mlp_norm = nn.LayerNorm(slot_dim)
        self.slot_norm = nn.LayerNorm(slot_dim)

        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, slot_dim),
        )

        self.gru = nn.GRUCell(slot_dim, slot_dim)

        self.slots_mu = torch.empty((1, 1, slot_dim))
        self.slots_log_sigma = torch.empty((1, 1, slot_dim))
        nn.init.xavier_uniform_(self.slots_mu)
        nn.init.xavier_uniform_(self.slots_log_sigma)

    def forward(self, x):
        b, n, d = x.shape

        x = self.input_norm(x)
        keys = self.keys(x)  # [batch_size, input_size, slot_dim]
        values = self.values(x)  # [batch_size, input_size, slot_dim]

        mu = self.slots_mu.expand(b, self.num_slots, -1)
        sigma = self.slots_log_sigma.exp().expand(b, self.num_slots, -1)

        slots = mu + sigma * torch.randn(
            mu.shape, device=x.device
        )  # [batch_size, num_slots, slot_dim]

        for i in range(self.num_iterations):
            slots_prev = slots
            slots = self.slot_norm(slots)  # [batch_size, num_slots, slot_dim]

            # Compute attention
            queries = self.queries(slots)  # [batch_size, num_slots, slot_dim]
            queries *= self.slot_dim ** -0.5  # normalization

            # b = batch_size, i = input_size, d = slot_dim, k = num_slots
            attn_logits = torch.einsum(
                "bid,bkd->bik", keys, queries
            )  # [batch_size, input_size, num_slots]
            attn = F.softmax(attn_logits, dim=1)

            # Weighted Mean
            attn += self.epsilon
            attn /= torch.sum(attn, dim=-1, keepdim=True)

            updates = torch.einsum(
                "bid,bik->bkd", values, attn
            )  # [batch_size, num_slots, slot_dim]

            updates = updates.reshape(-1, d)
            slots_prev = slots_prev.reshape(-1, d)

            slots = self.gru(updates, slots_prev)

            slots = slots.reshape(b, -1, d)
            slots += self.mlp(self.mlp_norm(slots))

        return slots


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


def spatial_broadcast(z, shape):
    """
    Spatial broadcast

    Args:
        z: latents
        shape: tuple of ints containing width and height of broadcast
    """
    z_b = torch.tile(z, (shape[0], shape[1], 1))
    x = torch.linspace(-1, 1, shape[0])
    y = torch.linspace(-1, 1, shape[0])
    x_b, y_b = torch.meshgrid(x, y)
    z_sb = torch.concatenate([z_b, x_b, y_b], dim=-1)
    return z_sb


class Encoder(nn.Module):
    def __init__(self, hid_dim, resolution):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(3, hid_dim, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(hid_dim, hid_dim, kernel_size=5, stride=1, padding=2)
        self.conv3 = nn.Conv2d(hid_dim, hid_dim, kernel_size=5, stride=1, padding=2)
        self.conv4 = nn.Conv2d(hid_dim, hid_dim, kernel_size=5, stride=1, padding=2)
        self.pos_encoder = SoftPositionEmbedding(hid_dim, resolution)

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


class Savi(nn.Module):
    def __init__(
        self,
        hid_dim=64,
        resolution=(128, 128),
        num_slots=8,
        slot_dim=128,
        slot_iterations=3,
    ):
        super(Savi, self).__init__()

        self.encoder = Encoder(hid_dim, resolution)
        self.decoder = Decoder(hid_dim, resolution)
        self.fc1 = nn.Linear(hid_dim, hid_dim)
        self.fc2 = nn.Linear(hid_dim, hid_dim)
        self.slot_attention = SlotAttention(
            slot_iterations, num_slots, slot_dim, slot_dim * 2
        )

    def forward(self, image):
        x = self.encoder(image)
        print(x.shape)
        x = nn.LayerNorm(x.shape[1:])(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

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
