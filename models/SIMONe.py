import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from position_encoding import PositionalEncoding3D, PositionalEncodingPermute3D


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
        K_slots=8,
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
        self.position_embeding = PositionalEncodingPermute3D(C)

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
        self.layer_norm = nn.LayerNorm((T, 4, H * W))

        self.recon_alpha = 1
        self.obj_kl_beta = 1
        self.frame_kl_beta = 1

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def reparameterize(self, dist, fill_tensor=None):
        """ Draws samples from unit spherical Guassian

            Params:
            -------
                dist: torch.Tensor:
                    B x {K,T} x C tensor containing means+logvars for latents
                    tensor will be split in half to obtain means and logvars
                fill_tensor: torch.Tensor:
                    Tensor to fill samples
            Returns:
            --------
                Samples with same size as fill_tensor tensor
        """
        B = dist.shape[0]
        mu, logvar = torch.split(dist, dist.shape[-1] // 2, -1)
        std = torch.exp(logvar)

        if fill_tensor is not None:
            # Broadcast means and stds to fill size
            mu = torch.tile(mu, fill_tensor.shape).reshape(
                *fill_tensor.shape, -1)
            std = torch.tile(std, fill_tensor.shape).reshape(
                *fill_tensor.shape, -1)

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
        frames = torch.stack(frames, dim=1)

        conv_frames_shape = frames.shape
        flattened_frames = frames.reshape((B, -1, self.transformer_dim))
        x = self.transformer_1(flattened_frames)

        # Reshape and spatial pool
        x = x.reshape(conv_frames_shape)
        if self.K_slots < self.I * self.J:
            x = self.spatial_pool(x)

        B, T, C, I, J = x.shape
        K = I * J
        x = x.reshape(B, -1, self.transformer_dim)
        x = self.transformer_2(x)

        temporal_mean = x.reshape(B, K, C, T).mean(dim=-1)
        lambda_obj = self.lambda_obj_mlp(temporal_mean)

        spatial_mean = x.reshape(B, T, C, K).mean(dim=-1)
        lambda_frame = self.lambda_frame_mlp(spatial_mean)

        return lambda_obj, lambda_frame

    def decode(self, object_dists, frame_dists, batch_size):
        B, T, C, H, W = batch_size
        K = object_dists.shape[1]

        assert H == W
        time_idxs = torch.linspace(0, 1, T)
        time_frames = torch.ones((B, T, H, W, 1))
        for t in range(T):
            time_frames[:, t] = time_frames[:, t] * time_idxs[t]

        spatial_pos = torch.linspace(-1, 1, H)
        decode_idxs = torch.stack(torch.meshgrid(spatial_pos, spatial_pos)).repeat(
            B, T, 1, 1, 1).permute(0, 1, 3, 4, 2)

        k_samples = []
        for k in range(K):
            fill_tensor = torch.zeros((B, H, W))
            frame_latents = self.reparameterize(
                frame_dists, fill_tensor=fill_tensor)
            frame_latents = frame_latents.reshape(B, T, H, W, -1)

            fill_tensor = torch.zeros((B, T, H, W))
            object_latents = self.reparameterize(
                object_dists[:, k], fill_tensor=fill_tensor)

            inputs = torch.cat(
                [object_latents, frame_latents, decode_idxs, time_frames], dim=-1)

            channels = inputs.shape[-1]
            inputs = inputs.reshape(B * T, channels,  -1)

            out = self.decoder(inputs)
            # out = self.layer_norm(out)
            out = out.reshape(B, T, H, W, -1)

            k_samples.append(out)

        decoded = torch.stack(k_samples, dim=-2)
        recons, mixture_logits = decoded.split(3, -1)

        return recons, mixture_logits

    def forward(self, x, decode_idxs=None):
        B, T, C, H, W = x.shape
        object_dists, frame_dists = self.encode(x)
        recons, mixture_logits_hat = self.decode(
            object_dists, frame_dists, (B, T, C, H, W))

        mixture_logits = F.softmax(mixture_logits_hat, -2)
        pixel_means = torch.distributions.Normal(
            recons, torch.ones_like(recons))
        p_x = torch.sum(mixture_logits * pixel_means.rsample(), dim=-2)
        p_x = p_x.reshape(B, T, 3, H, W)

        object_means, object_vars = torch.split(
            object_dists, object_dists.shape[-1] // 2, -1)
        frame_means, frame_vars = torch.split(
            frame_dists, frame_dists.shape[-1] // 2, -1)

        return {"recons": p_x,
                "object_dists": (object_means, object_vars),
                "frame_dists": (frame_means, frame_vars),
                "mixture_logits": mixture_logits
                }

    def kl(self, mu, logvar):
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    def compute_loss(self, inputs, outputs):
        recons = outputs["recons"]
        obj_mean, obj_var = outputs["object_dists"]
        frame_mean, frame_var = outputs["frame_dists"]

        obj_kl = self.kl(obj_mean, obj_var) * \
            (self.obj_kl_beta / obj_mean.shape[1])
        frame_kl = self.kl(frame_mean, frame_var) * \
            (self.frame_kl_beta / frame_mean.shape[1])

        B, T, C, H, W = inputs.shape
        K = obj_mean.shape[1]
        recon_scaling = self.recon_alpha / T * H * W
        print(inputs.shape, recons.shape)
        recon_loss = F.binary_cross_entropy(
            inputs, recons.detach(), reduction="sum") * recon_scaling

        total_loss = recon_loss + obj_kl + frame_kl

        return {"total_loss": total_loss,
                "obj_kl": obj_kl,
                "frame_kl": frame_kl}


if __name__ == "__main__":
    img = torch.rand((1, 10, 3, 128, 128))
    model = SIMONE(img.shape, 128)
    out = model(img)

    losses = model.compute_loss(img, out)

    print(losses)
    losses["total_loss"].backward()
    print("Backward pass done")
