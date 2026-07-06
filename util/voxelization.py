import torch
from . import event
from functools import partial
import numpy as np


def flatten_and_unflatten(x):
    original_shape = x.shape
    return x.flatten(), partial(np.reshape, newshape=original_shape)


def _split_coordinate(c):
    c = c.float()
    left_c = c.floor()
    right_weight = c - left_c
    left_c = left_c.int()
    right_c = left_c + 1
    return left_c, right_c, right_weight


def _to_lin_idx(t, x, y, W, H, B):
    mask = (0 <= x) & (0 <= y) & (0 <= t) & (x <= W - 1) & (y <= H - 1) & (t <= B - 1)
    lin_idx = x.long() + y.long() * W + t.long() * W * H
    return lin_idx, mask


def to_voxel_grid(event_sequence, nb_of_time_bins=5, remapping_maps=None):
    """
        返回事件流的体素网格表示。

        在体素网格表示中，时间维度被离散为“nb_of_time_bins”单元。
        使用双线性插值在两个邻近箱之间插值事件的极性并求和。
        如果事件流为空，体素网格将为空。

        In voxel grid representation, temporal dimension is
        discretized into "nb_of_time_bins" bins. The events fir
        polarities are interpolated between two near-by bins
        using bilinear interpolation and summed up.

        If eventprocessing stream is empty, voxel grid will be empty.
    """
    voxel_grid = torch.zeros(nb_of_time_bins,
                             event_sequence._image_height,
                             event_sequence._image_width,
                             dtype=torch.float32,
                             device='cpu')

    voxel_grid_flat, unflatten = flatten_and_unflatten(voxel_grid)
    # voxel_grid_flat = voxel_grid.flatten()
    # 将时间戳转换为[0，nb_of_time_bins]范围。
    duration = event_sequence.duration()
    start_timestamp = event_sequence.start_time()
    features = torch.from_numpy(event_sequence._features)
    x = features[:, event.X_COLUMN]
    y = features[:, event.Y_COLUMN]
    polarity = features[:, event.POLARITY_COLUMN].float()
    t = (features[:, event.TIMESTAMP_COLUMN] - start_timestamp) * (nb_of_time_bins - 1) / duration
    t = t.float()

    if remapping_maps is not None:
        remapping_maps = torch.from_numpy(remapping_maps)
        x, y = remapping_maps[:, y, x]

    left_t, right_t = t.floor(), t.floor() + 1
    left_x, right_x = x.floor(), x.floor() + 1
    left_y, right_y = y.floor(), y.floor() + 1

    for lim_x in [left_x, right_x]:
        for lim_y in [left_y, right_y]:
            for lim_t in [left_t, right_t]:
                mask = (0 <= lim_x) & (0 <= lim_y) & (0 <= lim_t) & (lim_x <= event_sequence._image_width - 1) \
                       & (lim_y <= event_sequence._image_height - 1) & (lim_t <= nb_of_time_bins - 1)

                # 在这里转换为long，否则掩码计算不正确
                lin_idx = lim_x.long() \
                          + lim_y.long() * event_sequence._image_width \
                          + lim_t.long() * event_sequence._image_width * event_sequence._image_height

                weight = polarity * (1 - (lim_x - x).abs()) * (1 - (lim_y - y).abs()) * (1 - (lim_t - t).abs())
                new_voxel_grid_flat = voxel_grid_flat.index_add_(dim=0, index=lin_idx[mask], source=weight[mask].float())
                # voxel_grid_flat.index_add_(dim=0, index=lin_idx[mask], source=weight[mask].float())

    voxel_grid = unflatten(new_voxel_grid_flat)

    return voxel_grid
