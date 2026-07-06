import torch
import torch.nn as nn
from .encoder import EventsEncoder, ImageEncoder
from .BiFEB import BidirFlow, Query
from .BiW import BiW

class MyModel(nn.Module):
	def __init__(self, args):
		super().__init__()
		self.base_channel = 32
		self.voxel_bins = args.voxel_bins
		self.nb_of_flow = args.nb_of_flow
		self.bins_per_flow = self.voxel_bins//self.nb_of_flow

		self.events_encoder = EventsEncoder(self.voxel_bins, self.base_channel, bins_per_flow=self.bins_per_flow)
		self.im_encoder = ImageEncoder(3, self.base_channel)
		self.bidir_flow0 = BidirFlow(self.base_channel)
		self.bidir_flow1 = BidirFlow(self.base_channel)
		self.query = Query(self.base_channel)
		self.biw = BiW(self.base_channel)

	def forward(self, im0, im1, event, interp_times):
		efs01 = self.events_encoder(im0, im1, event)
		if0_1, if0_4 = self.im_encoder(im0)
		if1_1, if1_4 = self.im_encoder(im1)
		b, t, c, h, w = efs01.shape

		ift0, ift1 = if0_4, if1_4
		ctxt0, ctxt1 = if0_4, if1_4
		ctxst0, ctxst1 = [if0_4], [if1_4]
		motion0t, motion1t = torch.zeros_like(if0_4), torch.zeros_like(if1_4)
		zero_flow = torch.zeros((b, 2, h, w), device=im0.device, dtype=im0.dtype, requires_grad=False)
		flowt0, flowt1 = zero_flow, zero_flow
		flows0t, flows1t = [zero_flow], [zero_flow]
		flowst0, flowst1 = [zero_flow], [zero_flow]
		for t_ in range(self.nb_of_flow):
			ift0, motion0t, flow0t, flowt0, ctxt0 = self.bidir_flow0(if0_4, ift0, flowt0, motion0t, ctxt0, efs01[:, t_])
			ift1, motion1t, flow1t, flowt1, ctxt1 = self.bidir_flow1(if1_4, ift1, flowt1, motion1t, ctxt1, efs01[:, t - t_ - 1])
			ctxst0.append(ctxt0)
			ctxst1.insert(0, ctxt1)
			flowst0.append(flowt0)
			flowst1.insert(0, flowt1)
			flows0t.append(flow0t)
			flows1t.insert(0, flow1t)

		imst = []
		for i in range(len(interp_times[0])):
			t = interp_times[0, i]
			index = t * self.nb_of_flow
			int_ = index.floor().long()
			pct_ = index - int_

			flow0t = flows0t[int_]*(1-pct_) + flows0t[int_+1]*pct_
			flow1t = flows1t[int_+1]*(1-pct_) + flows1t[int_]*pct_
			flowt0, ctxt0 = self.query(if0_4, flowst0[int_], flowst0[int_+1], ctxst0[int_], ctxst0[int_+1], pct_)
			flowt1, ctxt1 = self.query(if1_4, flowst1[int_], flowst1[int_+1], ctxst1[int_], ctxst1[int_+1], pct_)
			imt = self.biw(im0, im1, t, flow0t, flow1t, flowt0, flowt1, ctxt0, ctxt1, if0_1, if1_1)
			imst.append(imt)
		return torch.stack(imst, 1)
