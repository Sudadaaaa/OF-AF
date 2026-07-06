import torch
from torch import nn


class ResBlock(nn.Module):
    def __init__(self, in_channel, base_channel):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(in_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2)
        )
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channel, base_channel, 3, 1, 1, bias=True)
        )

    def forward(self, x):
        x_ = self.conv_in(x)
        x_res = self.conv_block(x)
        return x_res + x_


class EventsEncoder(nn.Module):
	def __init__(self, voxel_bins, base_channel, bins_per_flow=8):
		super().__init__()
		self.bins_per_flow = bins_per_flow
		self.pre_conv = ResBlock(voxel_bins + 2 * 3, voxel_bins)
		self.ct = nn.Parameter(torch.ones((1, voxel_bins, 1, 1)) * 0.2, requires_grad=True)
		self.down0 = nn.Sequential(
			nn.Conv2d(self.bins_per_flow, 2 * base_channel, 3, 2, 1, bias=True),
			nn.LeakyReLU(0.2)
		)
		self.down1 = nn.Sequential(
			nn.Conv2d(2 * base_channel, 4 * base_channel, 3, 2, 1, bias=True),
			nn.LeakyReLU(0.2)
		)

	def forward(self, im0, im1, events):
		pre_conv = self.pre_conv(torch.cat((im0, events, im1), 1)) * self.ct
		events_split_list = pre_conv.split(self.bins_per_flow, 1)
		n = events_split_list[0].shape[0]

		events_stack = torch.cat(events_split_list, 0)
		e_half = self.down0(events_stack)
		e_quater = self.down1(e_half)
		e_out = torch.stack(e_quater.split(n, 0), 1)
		return e_out


class ImageEncoder(nn.Module):
	def __init__(self, img_ch, base_channel):
		super().__init__()
		self.img_ch = img_ch
		self.preblock = ResBlock(img_ch, base_channel)
		self.down0 = nn.Conv2d(base_channel, 2*base_channel, 3, 2, 1, bias=True)
		self.down1 = nn.Conv2d(2*base_channel, 4*base_channel, 3, 2, 1, bias=True)

	def forward(self, img):
		if_1 = self.preblock(img)
		if_2 = self.down0(if_1)
		if_4 = self.down1(if_2)
		return if_1, if_4