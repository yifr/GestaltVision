"""Adapted from https://github.com/milesial/Pytorch-UNet/tree/master/unet"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        """A simple 3D Unet, adapted from a 2D Unet from https://github.com/milesial/Pytorch-UNet/tree/master/unet
        Arguments:
          n_channels = number of input channels; 3 for RGB, 1 for grayscale input
          n_classes = number of output channels/classes
        """
        super(UNet, self).__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Setting model to ", self.device)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.channels = [32, 64, 128, 256, 512]
        self.convtype = nn.Conv3d

        self.enc = Conv(n_channels, self.channels[0], conv_type=self.convtype)
        self.down1 = Down(self.channels[0], self.channels[1], conv_type=self.convtype)
        self.down2 = Down(self.channels[1], self.channels[2], conv_type=self.convtype)
        self.down3 = Down(self.channels[2], self.channels[3], conv_type=self.convtype)

        self.down4 = Down(self.channels[3], self.channels[4] , conv_type=self.convtype)
        self.up1 = Up(self.channels[4], self.channels[3])

        self.up2 = Up(self.channels[3], self.channels[2])
        self.up3 = Up(self.channels[2], self.channels[1])
        self.up4 = Up(self.channels[1], self.channels[0])
        self.outc = OutConv(self.channels[0], n_classes)

    def forward(self, x):
        x1 = self.enc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        x = self.up3(x3, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

    def get_metrics(self, images, targets, criterion, target_name="masks"):
        images = images.to(self.device).type(torch.float32)
        targets = targets.to(self.device).type(torch.float32)
        if images.shape[2] == self.n_channels:
            images = images.permute(0, 2, 1, 3, 4)  # B x C x T x H x W
            targets = targets.permute(0, 2, 1, 3, 4)

        output_targets = self.forward(images)
        if target_name == "masks":
            output_targets = F.softmax(output_targets, dim=1)
            pred_targets = output_targets.reshape(1, self.n_classes, -1).to(self.device)
            gt_targets = targets.reshape(1, -1).to(self.device)
        else:
            pred_targets = output_targets.to(self.device)
            gt_targets = targets.to(self.device)
        loss = criterion(pred_targets, gt_targets)

        metrics = {}
        metrics["loss"] = loss
        metrics["images"] = images
        metrics[f"gt_{target_name}"] = targets
        metrics[f"predicted_{target_name}"] = pred_targets

        return loss, metrics

    def log_metrics(self, writer, step, metrics, target="masks", phase="train", n_per_batch=4):
        print("Logging metrics...")
        writer.add_scalar(f"{phase}/loss", metrics["loss"].item(), step)
        predicted_target = metrics[f"predicted_{target}"].detach().cpu()[:n_per_batch]
        predicted_target = predicted_target.permute(0, 2, 1, 3, 4)
        gt_target = metrics[f"gt_{target}"].detach().cpu().permute(0, 2, 1, 3, 4)[:n_per_batch]
        images = metrics["images"].detach().cpu().permute(0, 2, 1, 3, 4)[:n_per_batch]
        writer.add_video(f"{phase}/predicted_{target}", predicted_target, step)
        writer.add_video(f"{phase}/gt_{target}", gt_target, step)
        writer.add_video(f"{phase}/input_video", images, step)

        return

class Conv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, conv_type=nn.Conv3d, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.conv = nn.Sequential(
            conv_type(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, conv_type=nn.Conv3d):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            Conv(in_channels, out_channels, conv_type=conv_type)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = Conv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, kernels_per_layer=1):
        super(DepthwiseSeparableConv3d, self).__init__()
        self.depthwise = nn.Conv3d(nin, nin * kernels_per_layer, kernel_size=kernel_size, padding=padding, groups=nin)
        self.pointwise = nn.Conv3d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


if __name__=="__main__":
    import sys
    sys.path.append("../")
    from data import gestalt
    from torch.utils.data import DataLoader
    import torchvision.transforms as T

    model = UNet(1, 2)
    transform = T.Compose([T.PILToTensor(), T.Resize((128, 128))])
    transforms = {"masks": transform, "images": transform}

    data_dir = "/om2/user/yyf/CommonFate/scenes/noise"
    data = DataLoader(Gestalt(data_dir, top_level=["voronoi"], sub_level=["superquadric_3"], frames_per_scene=32))
    criterion = nn.CrossEntropyLoss()

    batch = next(iter(data))
    print("images: ", batch["images"].shape, "masks: ", batch["masks"].shape)
    metrics = model.get_metrics(batch["images"], batch["masks"], criterion, "masks")

    print("loss:" , metrics["loss"].item())
    print("predicted masks:", metrics["predicted_masks"].shape)
    print("gt_masks: ", metrics["gt_masks"].shape)
    print("mean pred:", metrics["predicted_masks"].sum())
    print("gt_mask mean:", metrics["gt_masks"].sum())
