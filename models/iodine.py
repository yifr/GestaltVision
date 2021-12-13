import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm


def softplus_to_std(softplus):
    softplus = torch.min(softplus, torch.ones_like(softplus) * 80)
    return torch.sqrt(torch.log(1 + softplus.exp()) + 1e-5)


def mv_normal(loc, softplus):
    return torch.distributions.independent.Independent(
        torch.distributions.normal.Normal(loc, softplus_to_std(softplus)), 1
    )


def std_mv_normal(shape, device):
    loc = torch.zeros(shape).to(device)
    scale = torch.ones(shape).to(device)
    return torch.distributions.independent.Independent(
        torch.distributions.normal.Normal(loc, scale), 1
    )


def gmm_loglikelihood(x, x_loc, log_var, mask_logprobs):
    sq_err = (x.unsqueeze(1) - x_loc).pow(2)
    normal_ll = -0.5 * log_var - 0.5 * (sq_err / torch.exp(log_var))
    log_p_k = mask_logprobs + normal_ll
    log_p = torch.logsumexp(log_p_k, dim=1)
    nll = -torch.sum(log_p, dim=(1, 2, 3))
    return nll, {"log_p_k": log_p_k, "normal_ll": normal_ll}


class RefinementNetwork(nn.Module):
    """
    Iteratively refine posterior estimate. Concatenates ground truth image along with
    several intermediate outputs and recurrently processes a new mean and log variance
    estimate.
    """

    def __init__(
        self, latent_dim, input_size, refinement_channels_in, conv_channels, lstm_dim
    ):
        super(RefinementNetwork, self).__init__()

        self.input_size = input_size
        self.latent_dim = latent_dim

        self.conv = nn.Sequential(
            nn.Conv2d(refinement_channels_in, conv_channels, kernel_size=3, stride=2),
            nn.ELU(True),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, stride=2),
            nn.ELU(True),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=3, stride=2),
            nn.ELU(True),
            nn.Conv2d(conv_channels, conv_channels, kernel_size=5, stride=2),
            nn.ELU(True),
            nn.AvgPool2d(4),
            nn.Flatten(),
        )

        self.mlp = nn.Sequential(
            nn.Linear((input_size[1] // 4) * (input_size[1] // 4), lstm_dim),
            nn.ELU(True),
            nn.Linear(lstm_dim, lstm_dim),
            nn.ELU(True),
        )

        self.input_proj = nn.Sequential(
            nn.Linear(lstm_dim + 4 * self.latent_dim, lstm_dim), nn.ELU(True)
        )

        self.lstm = nn.LSTMCell(lstm_dim, latent_dim)
        self.loc = nn.Linear(lstm_dim, latent_dim)
        self.softplus = nn.Linear(lstm_dim, latent_dim)

    def forward(self, img_inputs, vector_inputs, h, c):
        x = self.conv(img_inputs)
        x = self.mlp(x)

        x = torch.cat([x, vector_inputs], 1)
        x = self.input_proj(x)
        x = x.unsqueeze(0)  # sequence dim

        self.lstm.flatten_parameters()
        out, (h, c) = self.lstm(x, (h, c))
        out = out.squeeze(0)

        loc = self.loc(out)
        softplus = self.softplus(out)
        lamda = torch.cat([loc, softplus], 1)
        return lamda, (h, c)


class SpatialBroadcastDecoder(nn.Module):
    def __init__(self, input_size, z_size, conv_channels):
        super(SpatialBroadcastDecoder, self).__init__()
        self.h, self.w = input_size[1], input_size[2]
        self.decode = nn.Sequential(
            nn.Conv2d(z_size + 2, conv_channels, 3, 1, 1),
            nn.ELU(True),
            nn.Conv2d(conv_channels, conv_channels, 3, 1, 1),
            nn.ELU(True),
            nn.Conv2d(conv_channels, conv_channels, 3, 1, 1),
            nn.ELU(True),
            nn.Conv2d(conv_channels, conv_channels, 1, 1),
            nn.ELU(True),
            nn.Conv2d(conv_channels, 4, 3, 1, 1),
        )

    def spatial_broadcast(self, z, h, w):
        n = z.shape[0]
        z_b = z.view((n, -1, 1, 1)).expand(-1, -1, h, w)

        x = torch.linspace(-1, 1, w, device=z.device)
        y = torch.linspace(-1, 1, h, device=z.device)
        y_b, x_b = torch.meshgrid(y, x)

        x_b = x_b.expand(n, 1, -1, -1)
        y_b = y_b.expand(n, 1, -1, -1)

        z_sb = torch.cat((z_b, x_b, y_b), dim=1)
        return z_sb

    def forward(self, z):
        z_sb = self.spatial_broadcast(z, self.h + 8, self.w + 8)
        out = self.decode(z_sb)
        return torch.sigmoid(out[:, :3]), out[:, 3]


class IODINE(nn.Module):
    def __init__(
        self,
        latent_dim,
        iters,
        slots,
        batch_size,
        log_scale,
        kl_beta,
        lstm_dim,
        img_size,
    ):
        super(IODINE, self).__init__()
        self.name = "IODINE"

        self.latent_dim = latent_dim
        self.inference_iters = iters
        self.K = slots
        self.batch_size = batch_size
        self.img_size = img_size
        self.kl_beta = kl_beta
        self.gmm_log_scale = log_scale * torch.ones(self.K)
        self.gmm_log_scale = self.gmm_log_scale.view(1, self.K, 1, 1, 1)
        self.input_size = (3, self.img_size[0], self.img_size[1])
        self.image_decoder = SpatialBroadcastDecoder(self.input_size, latent_dim, 64)
        self.refinement_network = RefinementNetwork(
            latent_dim, self.input_size, 16, 64, lstm_dim
        )

        self.lamda_0 = nn.Parameter(
            torch.cat(
                [torch.zeros(1, self.latent_dim), torch.ones(1, self.latent_dim)], dim=1
            )
        )

        # layernorms for iterative inference input
        n = self.input_size[1]
        self.layer_norms = nn.ModuleList(
            [
                nn.LayerNorm((1, n, n), elementwise_affine=False),
                nn.LayerNorm((1, n, n), elementwise_affine=False),
                nn.LayerNorm((3, n, n), elementwise_affine=False),
                nn.LayerNorm((1, n, n), elementwise_affine=False),
                nn.LayerNorm((self.latent_dim,), elementwise_affine=False),
                nn.LayerNorm((self.latent_dim,), elementwise_affine=False),
            ]
        )

        self.h_0, self.c_0 = (
            torch.zeros(1, self.batch_size * self.K, lstm_dim),
            torch.zeros(1, self.batch_size * self.K, lstm_dim),
        )

    def refine_inputs(
        image,
        means,
        masks,
        mask_logits,
        log_p_k,
        normal_ll,
        lamda,
        loss,
        layer_norms,
        eval_mode,
    ):
        # Non Gradient inputs:
        # 1) image [N, K, C, H, W]
        # 2) means [N, K, C, H, W]
        # 3) masks [N, K, 1, H, W]
        # 4) mask logits [N, K, 1, H, W]
        # 5) mask posterior [N, K, 1, H, W
        N, K, C, H, W = image.shape
        normal_ll = torch.sum(normal_ll, dim=2)
        mask_posterior = (
            normal_ll - torch.logsumexp(normal_ll, dim=1).unsqueeze(1)
        ).unsqueeze(2)

        # 6) pixelwise likelihood [N, K, 1, H, W]
        log_p_k = torch.logsumexp(log_p_k, dim=(1, 2))
        log_p_k = log_p_k.view(-1, 1, 1, H, W).repeat(1, K, 1, 1, 1)

        # Coordinate channels
        device = "cuda" if torch.cuda.is_available() else "cpu"
        x = torch.linspace(-1, 1, W, device=device)
        y = torch.linspace(-1, 1, H, device=device)

        y_b, x_b = torch.meshgrid(y, x)
        x_mesh = x_b.expand(N, K, 1, -1, -1)
        y_mesh = y_b.expand(N, K, 1, -1, -1)

        d_means, d_masks, d_lamda = torch.autograd.grad(
            loss,
            [means, masks, lamda],
            create_graph=not eval_mode,
            retain_graph=not eval_mode,
            only_inputs=True,
        )

        d_loc_z, d_sp_z = d_lamda.chunk(2, dim=1)
        d_loc_z, d_sp_z = d_loc_z.contiguous(), d_sp_z.contiguous()

        # apply layernorms
        log_p_k = layer_norms[0](log_p_k).detach()
        d_means = layer_norms[2](d_means).detach()
        d_masks = layer_norms[3](d_masks).detach()
        d_loc_z = layer_norms[4](d_loc_z).detach()
        d_sp_z = layer_norms[5](d_sp_z).detach()

        image_inputs = torch.cat(
            [
                image,
                means,
                masks,
                mask_logits,
                mask_posterior,
                log_p_k,
                d_means,
                d_masks,
                x_mesh,
                y_mesh,
            ],
            2,
        )
        vec_inputs = torch.cat([lamda, d_loc_z, d_sp_z], 1)

        return image_inputs.view(N * K, -1, H, W), vec_inputs

    def forward(self, x):
        C, H, W = self.input_size

        lamda = self.lamda_0.repeat(self.batch_size * self.K, 1)
        p_z = std_mv_normal(
            shape=(self.batch_size * self.K, self.latent_dim), device=x.device
        )

        total_loss = 0.0
        losses = []
        x_means = []
        masks = []
        h, c = self.h_0, self.c_0
        device = x.device
        h = h.to(device)
        c = c.to(device)

        for i in range(self.inference_iters):
            # sample posterior
            loc_z, sp_z = lamda.chunk(2, dim=1)
            loc_z, sp_z = loc_z.contiguous(), sp_z.contiguous()
            q_z = mv_normal(loc_z, sp_z)
            z = q_z.rsample()

            # get Means and Masks
            x_loc, mask_logits = self.image_decoder(z)
            x_loc = x_loc.view(self.batch_size, self.K, C, H, W)

            # softmax the slots
            mask_logits = mask_logits.view(self.batch_size, self.K, 1, H, W)
            mask_logprobs = F.log_softmax(mask_logits, dim=1)

            # NLL
            log_var = (2 * self.gmm_log_scale).to(device)
            nll, ll_outs = gmm_loglikelihood(x, x_loc, log_var, mask_logprobs)

            # KL Divergence
            kl_div = torch.distributions.kl.kl_divergence(q_z, p_z)
            kl_div = kl_div.view(self.batch_size, self.K).sum(1)

            if self.kl_beta == 0:
                loss = torch.mean(nll + self.kl_beta + kl_div)
            else:
                loss = self.kl_beta * torch.mean(kl_div)

            scaled_loss = (i + 1) / self.inference_iters * loss
            losses += [scaled_loss]
            total_loss += scaled_loss

            x_means += [x_loc]
            masks += [mask_logprobs]

            if i == self.inference_iters - 1:
                continue

            x_ = x.repeat(self.K, 1, 1, 1).view(self.batch_size, self.K, C, H, W)

            img_inps, vec_inps = self.refine_inputs(
                x_,
                x_loc,
                mask_logprobs,
                mask_logits,
                ll_outs["log_p_k"],
                lamda,
                loss,
                self.layer_norms,
                not self.training,
            )

            delta, (h, c) = self.refinement_network(img_inps, vec_inps, h, c)
            lamda = lamda + delta

        return {
            "total_loss": total_loss,
            "nll": torch.mean("nll"),
            "kl": torch.mean(x_means),
            "x_means": x_means,
            "masks": masks,
            "z": z,
        }
