import torch
from torch import nn
from torch.nn import functional as F
from .blocks import ResBlock, ResBlockIF, BackwardWarp
from .softsplat import softsplat

class RefBlock(nn.Module):
    def __init__(self, base_channel):
        super().__init__()
        self.conv0 = nn.Conv2d(base_channel*4, base_channel*2, 3, 1, 1)
        self.conv1 = nn.Conv2d(base_channel*4+1, base_channel*2, 3, 1, 1)
        self.res = ResBlock(base_channel*8+1, base_channel*4+1)

    def forward(self, mask, feature):
        mask_feature_s = self.conv0(mask * feature)
        mask_feature_w = self.conv1(torch.cat([feature, mask], dim=1))
        out = self.res(torch.cat([mask_feature_s, mask_feature_w, feature, mask], dim=1))
        mask = mask + out[:, :1]
        ref = out[:, 1:]
        return mask, ref


class MaskGuideNet(nn.Module):
    def __init__(self, base_channel):
        super().__init__()
        self.bwarp = BackwardWarp()
        self.refblock0 = RefBlock(base_channel)
        self.refblock1 = RefBlock(base_channel)
        self.res0 = ResBlock(base_channel*8+3, base_channel*4)
        self.conv0 = nn.Conv2d(3+3+1, base_channel*4, 3, 1, 1)
        self.conv1 = nn.Conv2d(base_channel*8, 3, 3, 1, 1)

    def forward(self, im0, im1, flow0t, flowt0, ctxt0, ctxt1):
        b, _, h, w = flow0t.shape
        ones = torch.ones((b, 1, h, w), device=im0.device, dtype=im0.dtype)

        #warp
        Flowt0 = F.interpolate(flowt0, scale_factor=4, mode="bilinear")*4
        Flow0t = F.interpolate(flow0t, scale_factor=4, mode="bilinear")*4
        im0t = softsplat(im0, Flow0t, tenMetric=None, strMode="avg")
        imt0 = self.bwarp(im0, Flowt0)

        #get mask
        warp_mask = softsplat(ones, flow0t, tenMetric=None, strMode="avg")
        diff_map = torch.sum(torch.abs(imt0 - im0t), dim=1, keepdim=True)
        hole_mask = 1 - warp_mask
        diff_mask = (diff_map > 0.1).to(im0.dtype)

        #get ref and mask
        hole_mask, ref0 = self.refblock0(hole_mask, ctxt0)
        hole_mask, ref1 = self.refblock1(hole_mask, ctxt1)
        im1_4 = F.interpolate(im1, scale_factor=1/4, mode="bilinear")

        #fuse ref
        ref = self.res0(torch.cat([ref0, ref1, im1_4], dim=1))
        Ref = F.interpolate(ref, scale_factor=4, mode="bilinear")
        Mask0t = F.interpolate(hole_mask, scale_factor=4, mode="bilinear") + diff_mask
        Ref_source = self.conv0(torch.cat([imt0*Mask0t, imt0, Mask0t], dim=1))

        imt0 = imt0 + self.conv1(torch.cat([Ref, Ref_source], dim=1))
        return imt0, im0t, Mask0t


class BiW(nn.Module):
    def __init__(self, base_channel):
        super().__init__()
        self.fuse_warp = MaskGuideNet(base_channel)
        self.channel_squeeze0 = nn.Conv2d(128, 32, 1, 1, 0, bias=True)
        self.channel_squeeze1 = nn.Conv2d(128, 32, 1, 1, 0, bias=True)
        self.refine_blocks = ResBlockIF(3+128, 64)
        self.conv_out = nn.Conv2d(64, 3, 1, 1, bias=True)

    def forward(self, im0, im1, t, flow0t, flow1t, flowt0, flowt1, ctxt0, ctxt1, if0_1, if1_1):
        imt0, im0t, mask0t = self.fuse_warp(im0, im1, flow0t, flowt0, ctxt0, ctxt1)
        imt1, im1t, mask1t = self.fuse_warp(im1, im0, flow1t, flowt1, ctxt1, ctxt0)
        mask = torch.softmax(torch.cat([(1 - mask0t)*(1 - t), (1 - mask1t)*t], 1), 1)[:, :1]  # (B,1,H,W)
        fuse_out = imt0*mask+(1-mask)*imt1

        forard_feat_ld = self.channel_squeeze0(ctxt0)
        backward_feat_ld = self.channel_squeeze1(ctxt1)
        forard_feat_hd, backward_feat_hd = F.interpolate(forard_feat_ld, scale_factor=4, mode='bilinear'), F.interpolate(backward_feat_ld, scale_factor=4, mode='bilinear')
        refine_in = torch.cat((forard_feat_hd, backward_feat_hd, fuse_out, if0_1, if1_1), 1)
        refine_out = self.conv_out(self.refine_blocks(refine_in))+fuse_out
        return refine_out
