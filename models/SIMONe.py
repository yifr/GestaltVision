import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import einops
from .position_encoding import PositionalEncodingPermute3D


def ConvNet(
    num_layers, in_channels, mid_channels, out_channels, kernel=4, stride=2, padding=1,
    dim=2
):
    layers = []
    if dim == 1:
        net = nn.Conv1d
    elif dim == 2:
        net = nn.Conv2d

    for i in range(int(num_layers)):
        if i == 0:
            conv = net(
                in_channels,
                mid_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
            )
        elif i == num_layers - 1:
            conv = net(
                mid_channels,
                out_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
            )
        else:
            conv = net(
                mid_channels,
                mid_channels,
                kernel_size=kernel,
                stride=stride,
                padding=padding,
            )
        layers.append(conv)
        layers.append(nn.ReLU())

    convnet = nn.Sequential(*layers)
    return convnet


class SIMONE(nn.Module):
    def __init__(
        self,
        input_size,
        cnn_encoder_hidden_dim,
        n_transformer_layers=4,
        transformer_heads=4,
        transformer_dim=64,
        transformer_feedforward=1024,
        cnn_decoder_dim_size=128,
        K_slots=16,
        recon_alpha=0.2,
        obj_kl_beta=1e-4,
        frame_kl_beta=1e-4,
        pixel_std=0.05,
        device="cuda"
    ):
        """
        SIMONe model
        Args:
            input_size: tuple of (B, T, C, H, W)
            hidden_dim: channels in conv layers
        """

        super(SIMONE, self).__init__()

        B, T, C, H, W = input_size

        self.cnn_encoder_hidden_dim = cnn_encoder_hidden_dim
        self.n_transformer_layers = n_transformer_layers
        self.transformer_heads = transformer_heads
        self.transformer_dim = transformer_dim
        self.transformer_feedforward = transformer_feedforward
        self.cnn_decoder_dim_size = cnn_decoder_dim_size
        self.K_slots = K_slots
        self.I, self.J = 8, 8  # size of output feature map for each frame

        # CNN Outputs 8*8 feature map for each frame
        self.position_embedding1 = PositionalEncodingPermute3D(128)
        self.position_embedding2 = PositionalEncodingPermute3D(128)

        n_layers = np.log2(W / 8)
        self.cnn_encoder = ConvNet(
            n_layers, C, cnn_encoder_hidden_dim, cnn_encoder_hidden_dim
        )  # Output should be B x T x C x 8 x 8

        # input_dim = torch.prod(input_size)
        # Transformers have 4 layers, 5 heads, value=64, MLPs with hidden_layer=1024
        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=transformer_heads,
            dim_feedforward=transformer_feedforward,
            dropout=0.0,
        )
        self.transformer_1 = nn.TransformerEncoder(
            transformer_encoder_layer, n_transformer_layers
        )

        if self.K_slots < self.I * self.J:
            kernel_size = (
                1,
                int(self.I / np.sqrt(self.K_slots)),
                int(self.J / np.sqrt(self.K_slots)),
            )

            self.spatial_pool = nn.AvgPool3d(kernel_size, stride=kernel_size)
            self.spatial_pool_scaling = np.sqrt(
                self.K_slots / (self.I * self.J))

        self.transformer_2 = nn.TransformerEncoder(
            transformer_encoder_layer, n_transformer_layers
        )

        # MLP over objects
        self.lambda_frame_mlp = nn.Sequential(
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 128)
        )

        self.lambda_obj_mlp = nn.Sequential(
            nn.Linear(128, 256),
            nn.SiLU(),
            nn.Linear(256, 128)
        )

        self.decoder = ConvNet(4, H + 3, 128, 4, kernel=1,
                               stride=1, padding=0, dim=1)
        self.layer_norm = nn.LayerNorm((T, H, W, K_slots, 1))

        self.recon_alpha = recon_alpha
        self.obj_kl_beta = obj_kl_beta
        self.frame_kl_beta = frame_kl_beta
        self.pixel_std = torch.tensor(pixel_std).to(device)

        self.device = device

    def reparameterize(self, dist, repeat_pattern=""):
        """ Draws samples from unit spherical Guassian

            Params:
            -------
                dist: torch.Tensor:
                    B x {K,T} x C tensor containing means+logvars for latents
                    tensor will be split in half to obtain means and logvars
                repeat_pattern: str:
                    einops string equation for repeating tensor along dimensions

            Returns:
            --------
                Samples with same size as fill_tensor tensor
        """
        mu, logvar = torch.split(dist, dist.shape[-1] // 2, -1)
        std = torch.exp(logvar)

        if repeat_pattern:
            # Broadcast means and stds to fill size
            mu = einops.repeat(mu, repeat_pattern)
            std = einops.repeat(std, repeat_pattern)

        eps = torch.randn_like(std)
        latents = mu + std * eps

        return latents

    def encode(self, x):
        """ Encodes sequence of images into frame and object latents

        Params:
        -------
            x: torch.Tensor:
                B, T, C, H, W sized tensor of input images
        Returns:
        --------
            frame_means, object_means:
                Means of object and frame latents
        """
        B, T, C, H, W = x.shape

        # encode frames:
        frames = []
        for i in range(x.shape[1]):
            frame_encoding = self.cnn_encoder(x[:, i])
            frames.append(frame_encoding)
        frames = torch.stack(frames, dim=1).reshape(B, T, self.I, self.J, -1)
        frames_emb = self.position_embedding1(frames)
        frames = frames + frames_emb

        conv_frames_shape = frames.shape
        flattened_frames = frames.reshape((B, -1, self.transformer_dim))
        x = self.transformer_1(flattened_frames)

        # Reshape and spatial pool
        x = x.reshape(B, T, -1, self.I, self.J)
        if self.K_slots < self.I * self.J:
            x = self.spatial_pool(x)
            x = x.reshape((B, T, self.I // 2, self.J // 2, -1))

        x_emb = self.position_embedding2(x)
        x = x + x_emb
        B, T, I, J, C = x.shape
        K = I * J
        x = x.reshape(B, -1, self.transformer_dim)
        x = self.transformer_2(x)

        temporal_mean = x.reshape(B, K, C, T).mean(dim=-1)
        lambda_obj = self.lambda_obj_mlp(temporal_mean)

        spatial_mean = x.reshape(B, T, C, K).mean(dim=-1)
        lambda_frame = self.lambda_frame_mlp(spatial_mean)

        return lambda_obj, lambda_frame

    def decode(self, object_dists, frame_dists, batch_size):
        """ Decodes pixel means and Gaussian mixture logits given
            object and frame means and log vars. Samples independent
            object latents for each pixel across (batch, time, height, width)
            and passes them through 1x1 convnet

        Parameters:
        -----------
        object_dists: torch.Tensor
            (B, K, latent_dim * 2) tensor containing means and logvars for object latents
        frame_dists: torch.Tensor
            (B, T, latent_dim * 2) tensor containing means and logvars for frame latents
        batch_size: tuple
            tuple containing batch size info
        """
        B, T, C, H, W = batch_size
        K = object_dists.shape[1]

        assert H == W

        # Stack temporal coordinate across (B, T, H, W)
        time_idxs = torch.linspace(0, 1, T)
        time_frames = torch.ones((B, T, H, W, 1)).to(self.device)
        for t in range(T):
            time_frames[:, t] = time_frames[:, t] * time_idxs[t]

        # Stack spatial coordinates and repeat across batch and time
        spatial_coords = torch.linspace(-1, 1, H)
        xv, yv = torch.meshgrid([spatial_coords, spatial_coords], indexing="xy")
        ll = torch.stack([xv, yv], -1)
        decode_idxs = einops.repeat(ll, "H W X -> B T H W X", B=B, T=T)

        decode_idxs = decode_idxs.to(self.device)
        k_samples = []

        for k in range(K):
            # Broadcast temporal latents spatially
            frame_latents = self.reparameterize(
                frame_dists, repeat_pattern=f"b t d -> b t {H} {W} d")

            # Broadcast object latents temporally and spatially
            object_latents = self.reparameterize(
                object_dists[:, k], repeat_pattern=f"b d -> b {T} {H} {W} d")

            # Inputs are (B, T, H, W, latent_dim * 2 + 3) --> 3 is for x, y, t coords
            inputs = torch.cat(
                [object_latents, frame_latents, decode_idxs, time_frames], dim=-1)

            channels = inputs.shape[-1]
            inputs = inputs.reshape(B * T, channels,  -1)

            out = self.decoder(inputs)
            out = out.reshape(B, T, H, W, -1)

            k_samples.append(out)

        decoded = torch.stack(k_samples, dim=-2)
        recons, mixture_logits = decoded.split(3, -1)

        return recons, mixture_logits

    def forward(self, x, decode_idxs=None):
        B, T, C, H, W = x.shape
        object_dists, frame_dists = self.encode(x)
        pixel_means, mixture_logits = self.decode(
            object_dists, frame_dists, (B, T, C, H, W))

        # Normalize and take softmax over object dim of logits
        mixture_logits = self.layer_norm(mixture_logits)
        mixture_weights = F.softmax(mixture_logits, -2)  # (B, T, H, W, K, 1)

        # Sample RGB values given pixel means
        pixel_dists = torch.distributions.Normal(pixel_means,
                                                 self.pixel_std * torch.ones_like(pixel_means))

        p_x = torch.sum(mixture_weights * pixel_dists.rsample(), dim=-2) # (B, T, H, W, C)
        object_means, object_vars = torch.split(
            object_dists, object_dists.shape[-1] // 2, -1)
        frame_means, frame_vars = torch.split(
            frame_dists, frame_dists.shape[-1] // 2, -1)

        return {"recons": p_x,
                "pixel_means": pixel_means,
                "mixture_weights": mixture_weights,
                "object_dists": (object_means, object_vars),
                "frame_dists": (frame_means, frame_vars)
                }

    def kld(self, mu, logvar):
        return torch.mean(-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

    def compute_loss(self, inputs, outputs):
        B, T, C, H, W = inputs.shape
        recons = outputs["recons"].reshape(B, T, C, H, W)
        obj_mean, obj_var = outputs["object_dists"]
        frame_mean, frame_var = outputs["frame_dists"]

        obj_kl = self.kld(obj_mean, obj_var) * \
            (self.obj_kl_beta / obj_mean.shape[1])
        frame_kl = self.kld(frame_mean, frame_var) * \
            (self.frame_kl_beta / frame_mean.shape[1])

        recon_dist = torch.distributions.Normal(recons, torch.ones_like(recons) * self.pixel_std)
        log_prob = -1 * recon_dist.log_prob(inputs).mean() * self.recon_alpha
        total_loss = log_prob + obj_kl + frame_kl

        return {"total_loss": total_loss,
                "obj_kl_loss": obj_kl,
                "frame_kl_loss": frame_kl}


if __name__ == "__main__":
    device = "cpu"
    img = torch.rand(1, 10, 3, 128, 128).to(device)
    model = SIMONE(img.shape, 128, device=device).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=2e-4)
    for i in range(1000):
        optim.zero_grad()
        out = model(img)
        losses = model.compute_loss(img, out)
        losses["total_loss"].backward()
        print(losses["total_loss"].item())
        optim.step()

    from ..utils import make_video
    make_video([img.detach().cpu().numpy(), out["recons"].detach().cpu().numpy()],
               titles=["ground truth", "reconstruction"],
               output_name="simone_overfit_test_1")

