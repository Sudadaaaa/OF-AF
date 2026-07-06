import torch
import numpy as np
from torch import nn

class ResBlock(nn.Module):
    def __init__(self, in_channel, base_channel):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channel, base_channel, 3, 1, 1, bias=True),
        )
        self.conv_block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x_ = self.conv_in(x)
        x_res = self.conv_block(x_)
        return x_res + x_


class ResBlockIF(nn.Module):
    def __init__(self, in_channel, base_channel):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channel, base_channel, 3, 1, 1, bias=True),
        )
        self.conv_block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
        )
        self.conv_block1 = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        x_ = self.conv_in(x)
        x_res = self.conv_block(x_) + x_
        x_res1 = self.conv_block1(x_res) + x_res
        return x_res1


class BackwardWarp(nn.Module):
    def __init__(self):
        super().__init__()
        self.grid = {}

    def forward(self, img, flow):
        N, _, H, W = img.size()

        u = flow[:, 0, :, :]
        v = flow[:, 1, :, :]

        key = f"{N}_{H}_{W}_{img.device}_{img.dtype}"
        if key not in self.grid:
            gridX, gridY = np.meshgrid(np.arange(W), np.arange(H))
            gridX = torch.tensor(gridX, requires_grad=False, device=img.device, dtype=img.dtype).unsqueeze(0).expand_as(u)
            gridY = torch.tensor(gridY, requires_grad=False, device=img.device, dtype=img.dtype).unsqueeze(0).expand_as(v)
            self.grid.update({
                key: [gridX, gridY]
            })
        gridX, gridY = self.grid[key]

        x = gridX + u
        y = gridY + v
        # range -1 to 1
        x = 2 * (x / W - 0.5)
        y = 2 * (y / H - 0.5)
        # stacking X and Y
        grid = torch.stack((x, y), dim=3)
        # Sample pixels using bilinear interpolation.
        imgOut = torch.nn.functional.grid_sample(img, grid, align_corners=False)

        return imgOut
