import os
import numpy as np
import torch
from functools import partial
from PIL import Image

from . import os_tools
from .encodings import events_to_image
from concurrent.futures import ThreadPoolExecutor

TIMESTAMP_COLUMN = 2
X_COLUMN = 0
Y_COLUMN = 1
POLARITY_COLUMN = 3


def save_events(events, file):
    """Save events to ".npy" file.

    In the "events" array columns correspond to: x, y, timestamp, polarity.

    We store:
    (1) x,y coordinates with uint16 precision.
    (2) timestamp with float32 precision.
    (3) polarity with binary precision, by converting it to {0,1} representation.

    """
    if (0 > events[:, X_COLUMN]).any() or (events[:, X_COLUMN] > 2 ** 16 - 1).any():
        raise ValueError("Coordinates should be in [0; 2**16-1].")
    if (0 > events[:, Y_COLUMN]).any() or (events[:, Y_COLUMN] > 2 ** 16 - 1).any():
        raise ValueError("Coordinates should be in [0; 2**16-1].")
    # if ((events[:, POLARITY_COLUMN] != -1) & (events[:, POLARITY_COLUMN] != 1)).any():
    #     raise ValueError("Polarity should be in {-1,1}.")
    events = np.copy(events)
    x, y, timestamp, polarity = np.hsplit(events, events.shape[1])
    polarity = (polarity + 1) / 2
    np.savez(
        file,
        x=x.astype(np.uint16),
        y=y.astype(np.uint16),
        t=timestamp.astype(np.float32),
        p=polarity.astype(np.bool),
    )


def load_events(file, hsergb=False, bsergb=False, hqf=False, size=None):
    """
    加载事件数据并返回一个numpy数组
    """
    tmp = np.load(file, allow_pickle=True)
    if hsergb:
        (x, y, timestamp, polarity) = (
            tmp["x"].astype(np.float32).reshape((-1,)),
            tmp["y"].astype(np.float32).reshape((-1,)),
            tmp["t"].astype(np.float32).reshape((-1,)),
            tmp["p"].astype(np.float32).reshape((-1,)) * 2 - 1
        )
    elif bsergb:
        (x, y, timestamp, polarity) = (
            tmp["x"].astype(np.float32).reshape((-1,)),
            tmp["y"].astype(np.float32).reshape((-1,)),
            tmp["timestamp"].astype(np.float32).reshape((-1,)),
            tmp["polarity"].astype(np.float32).reshape((-1,)) * 2 - 1,  # to convert from 0,1 to 1,-1 polarity
        )
        x = (2 * 970 * x) // 62000  # Since the authors did not provided necessary information about scaling these numbers are carefully detemined by others.
        y = (630 * y) // 20160 
    elif hqf:
        (x, y, timestamp, polarity) = (
            tmp["x"].astype(np.float64).reshape((-1,)),
            tmp["y"].astype(np.float64).reshape((-1,)),
            tmp["t"].astype(np.float64).reshape((-1,)),
            tmp["p"].astype(np.float32).reshape((-1,))
        )
    else:
        (x, y, timestamp, polarity) = (
            tmp["x"].astype(np.float64).reshape((-1,)),
            tmp["y"].astype(np.float64).reshape((-1,)),
            tmp["t"].astype(np.float64).reshape((-1,)),
            tmp["p"].astype(np.float32).reshape((-1,))
        )
    events = np.stack((x, y, timestamp, polarity), axis=-1)

    # [g, k, g_h, k_w]
    if size is not None:
        events = np.delete(events, np.where(
            (events[:, 0] < size[1]) | (events[:, 0] >= size[3]) | (events[:, 1] < size[0]) | (events[:, 1] >= size[2])
        )[0], axis=0)
        events[:, 0] = events[:, 0] - size[1]
        events[:, 1] = events[:, 1] - size[0]
    return events

# def load_events(file):
#     """
#     加载事件数据并返回一个numpy数组
#     """
#     tmp = np.load(file, allow_pickle=True)
#     (x, y, timestamp, polarity) = (
#         tmp["x"].astype(np.float64).reshape((-1,)) / 32768 * 970,
#         tmp["y"].astype(np.float64).reshape((-1,)) / 20160 * 625,
#         tmp["timestamp"].astype(np.float64).reshape((-1,)),
#         tmp["polarity"].astype(np.float32).reshape((-1,)) * 2 -1
#     )
#     events = np.stack((x, y, timestamp, polarity), axis=-1)
#     return events


class EventJITSequenceIterator(object):
    """事件序列迭代器"""

    def __init__(self, filenames):
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        return load_events(self.filenames[index])

    def __iter__(self):
        for filename in self.filenames:
            features = load_events(filename)
            yield features


class EventJITSequence(object):
    """JIT文件序列阅读器"""

    def __init__(self, filenames, height, width):
        self._evseq = EventJITSequenceIterator(filenames)
        self._image_height = height
        self._image_width = width

    def make_sequential_iterator(self, timestamps):
        ev_seq_iter = iter(self._evseq)
        curbuf = next(ev_seq_iter)
        for start_timestamp, end_timestamp in zip(timestamps[:-1], timestamps[1:]):
            events = []
            # 搜索第一个事件
            while not len(curbuf) or curbuf[-1, 2] < start_timestamp:
                curbuf = next(ev_seq_iter)
            start_index = np.searchsorted(curbuf[:, 2], start_timestamp, side='right')
            events.append(curbuf[start_index:])
            curbuf = next(ev_seq_iter)
            # 搜索最后一个事件
            while not len(curbuf) or curbuf[-1, 2] < end_timestamp:
                events.append(curbuf)
                curbuf = next(ev_seq_iter)
            # 切换到最后一个事件
            end_index = np.searchsorted(curbuf[:, 2], end_timestamp, side='right')
            events.append(curbuf[:end_index])
            curbuf = curbuf[end_index:]

            features = np.concatenate(events)

            yield EventSequence(
                features=features,
                image_height=self._image_height,
                image_width=self._image_width,
                start_time=start_timestamp,
                end_time=end_timestamp,
            )

    @classmethod
    def from_folder(
            cls, folder, image_height, image_width, event_file_template="{:06d}.npz"
    ):
        filename_iterator = os_tools.make_glob_filename_iterator(
            os.path.join(folder, event_file_template)
        )
        filenames = [filename for filename in filename_iterator]
        return cls(filenames, image_height, image_width)


class EventSequence(object):
    """以旧优先顺序存储事件。"""

    def __init__(
            self, features, image_height, image_width, start_time=None, end_time=None
    ):
        """返回EventSequence类的对象。

        Args:
            features: 事件序列，行为事件，列为特征(x, y, timestamp, polarity)
            image_height, image_width: 事件传感器的宽度和高度。
            start_time, end_time: 事件序列的开始和结束时间。
        """
        self._image_width = image_width
        self._image_height = image_height
        if torch.is_tensor(features):
            self._features = features.numpy()
            self._features_torch = features
        else:
            self._features = features
            self._features_torch = torch.from_numpy(features)
        self._start_time = (
            start_time if start_time is not None else self._features[:, TIMESTAMP_COLUMN].min()
        )
        self._end_time = (
            end_time if end_time is not None else self._features[:, TIMESTAMP_COLUMN].max()
        )
    def __len__(self):
        return self._features.shape[0]

    # 检查是否自洽，空间坐标是否在范围内，时间戳是否为升序，极性是否为1或负1，时间戳是否在范围内
    def is_self_consistent(self):
        return (
                self.are_spatial_coordinates_within_range()
                and self.are_timestamps_ascending()
                and self.are_polarities_one_and_minus_one()
                and self.are_timestamps_within_range()
        )

    def are_spatial_coordinates_within_range(self):
        x = self._features[:, X_COLUMN]
        y = self._features[:, Y_COLUMN]
        return np.all((x >= 0) & (x < self._image_width)) and np.all(
            (y >= 0) & (y < self._image_height)
        )

    def are_timestamps_ascending(self):
        timestamp = self._features[:, TIMESTAMP_COLUMN]
        return np.all((timestamp[1:] - timestamp[:-1]) >= 0)

    def are_timestamps_within_range(self):
        timestamp = self._features[:, TIMESTAMP_COLUMN]
        return np.all((timestamp <= self.end_time()) & (timestamp >= self.start_time()))

    def are_polarities_one_and_minus_one(self):
        polarity = self._features[:, POLARITY_COLUMN]
        return np.all((polarity == -1) | (polarity == 1))

    def flip_horizontally(self):
        self._features[:, X_COLUMN] = (
                self._image_width - 1 - self._features[:, X_COLUMN]
        )

    def flip_vertically(self):
        self._features[:, Y_COLUMN] = (
                self._image_height - 1 - self._features[:, Y_COLUMN]
        )

    def create_cnt_encoding(self):
        """
        Generate a two-channel event image containing event counters.
        """
        sensor_size = (self._image_width, self._image_height)
        xs = self._features_torch[:, 0]
        ys = self._features_torch[:, 1]
        ps = self._features[:, 3]
        mask_pos = ps.copy()
        mask_neg = ps.copy()
        mask_pos[ps < 0] = 0
        mask_neg[ps > 0] = 0

        pos_cnt = events_to_image(xs, ys, ps * mask_pos, sensor_size=sensor_size)
        neg_cnt = events_to_image(xs, ys, ps * mask_neg, sensor_size=sensor_size)

        return torch.stack([pos_cnt, neg_cnt])

    def create_polarity_mask(self):
        """
        Creates a two channel tensor that acts as a mask for the input event list.
        :param ps: [N] tensor with event polarity ([-1, 1])
        :return [N x 2] event representation
        """
        ps = self._features_torch[:, 3]
        inp_pol_mask = torch.stack([ps, ps])
        inp_pol_mask[0, :][inp_pol_mask[0, :] < 0] = 0
        inp_pol_mask[1, :][inp_pol_mask[1, :] > 0] = 0
        inp_pol_mask[1, :] *= -1
        inp_pol_mask = inp_pol_mask.clone().permute(1, 0)
        return inp_pol_mask

    def reverse(self):
        """Reverse temporal direction of the eventprocessing stream.
            事件流的反向时间方向。
        Polarities of the events reversed.

                          (-)       (+)
        --------|----------|---------|------------|----> time
           t_start        t_1       t_2        t_end

                          (+)       (-)
        --------|----------|---------|------------|----> time
                0    (t_end-t_2) (t_end-t_1) (t_end-t_start)

        """
        if len(self) == 0:
            return
        self._features[:, TIMESTAMP_COLUMN] = (
                self._end_time - self._features[:, TIMESTAMP_COLUMN]
        )
        self._features[:, POLARITY_COLUMN] = -self._features[:, POLARITY_COLUMN]
        self._start_time, self._end_time = 0, self._end_time - self._start_time
        # 翻转“feature”矩阵的行，因为它首先按最早的排序。
        self._features = np.copy(np.flipud(self._features))
        return EventSequence(self._features, self._image_height, self._image_width, self._start_time, self._end_time)

    def duration(self):
        return self.end_time() - self.start_time()

    def start_time(self):
        return self._start_time

    def end_time(self):
        return self._end_time

    def min_timestamp(self):
        return self._features[:, TIMESTAMP_COLUMN].min()

    def max_timestamp(self):
        return self._features[:, TIMESTAMP_COLUMN].max()

    def filter_by_polarity(self, polarity, make_deep_copy=True):
        mask = self._features[:, POLARITY_COLUMN] == polarity
        return self.filter_by_mask(mask, make_deep_copy)

    def copy(self):
        return EventSequence(
            features=np.copy(self._features),
            image_height=self._image_height,
            image_width=self._image_width,
            start_time=self._start_time,
            end_time=self._end_time,
        )

    def filter_by_mask(self, mask, make_deep_copy=True):
        if make_deep_copy:
            return EventSequence(
                features=np.copy(self._features[mask]),
                image_height=self._image_height,
                image_width=self._image_width,
                start_time=self._start_time,
                end_time=self._end_time,
            )
        else:
            return EventSequence(
                features=self._features[mask],
                image_height=self._image_height,
                image_width=self._image_width,
                start_time=self._start_time,
                end_time=self._end_time,
            )

    def filter_by_timestamp(self, start_time, end_time, make_deep_copy=True):
        """
            返回按时间戳筛选的事件序列。
            新序列包括[start_time，end_time]中的事件。
        """
        mask = (start_time <= self._features[:, TIMESTAMP_COLUMN]) & (
                end_time > self._features[:, TIMESTAMP_COLUMN]
        )

        event_sequence = self.filter_by_mask(mask, make_deep_copy)
        event_sequence._start_time = start_time
        event_sequence._end_time = end_time
        return event_sequence

    def to_image(self, step, background=None):
        """
            将事件流可视化为PIL图像。
            如果像素事件的主极性为1，则像素显示为红色；
            如果像素事件的主极性为-1，则像素显示为蓝色；
            如果像素没有接收到任何事件，或者其事件没有主极性，则显示为白色。
            Args:
                background: is PIL image.
        """
        fea = self._features[::step]
        polarity = fea[:, POLARITY_COLUMN] == 1
        x_negative = fea[~polarity, X_COLUMN].astype(int)
        y_negative = fea[~polarity, Y_COLUMN].astype(int)
        x_positive = fea[polarity, X_COLUMN].astype(int)
        y_positive = fea[polarity, Y_COLUMN].astype(int)
        # polarity = self._features[:, POLARITY_COLUMN] == 1
        # x_negative = self._features[~polarity, X_COLUMN].astype(np.int)
        # y_negative = self._features[~polarity, Y_COLUMN].astype(np.int)
        # x_positive = self._features[polarity, X_COLUMN].astype(np.int)
        # y_positive = self._features[polarity, Y_COLUMN].astype(np.int)

        positive_histogram, _, _ = np.histogram2d(
            x_positive,
            y_positive,
            bins=(self._image_width, self._image_height),
            range=[[0, self._image_width], [0, self._image_height]],
        )
        negative_histogram, _, _ = np.histogram2d(
            x_negative,
            y_negative,
            bins=(self._image_width, self._image_height),
            range=[[0, self._image_width], [0, self._image_height]],
        )

        red = np.transpose(positive_histogram > negative_histogram)
        blue = np.transpose(positive_histogram < negative_histogram)

        if background is None:
            height, width = red.shape
            background = Image.fromarray(
                np.full((height, width, 3), 255, dtype=np.uint8)
            )
        from util.visualization_tools import plot_points_on_background
        y, x = np.nonzero(red)
        points_on_background = plot_points_on_background(
            y, x, background, [255, 0, 0]
        )
        y, x = np.nonzero(blue)
        points_on_background = plot_points_on_background(
            y, x, points_on_background, [0, 0, 255]
        )
        return points_on_background

    def _advance_index_to_timestamp(self, timestamp, side='left', start_index=0):
        """
            Binary search从“start_index”返回第一个带有timestamp>“timestamp”的事件的索引。
        """
        # left = start_index
        # right = self.__len__() - 1
        # while left <= right:
        #     mid = left + (right - left) // 2
        #     midval = self._features[mid, TIMESTAMP_COLUMN]
        #     if midval == timestamp:
        #         return mid
        #     elif midval < timestamp:
        #         left = mid + 1
        #     else:
        #         right = mid - 1
        # if side == 'left':
        #     return left
        # return right

        index = start_index
        while index < len(self):
            if self._features[index, TIMESTAMP_COLUMN] >= timestamp:
                return index
            index += 1
        return index

    def split_in_two(self, timestamp):
        """
            将原始序列一分为二，返回两个序列。
        """
        if not (self.start_time() <= timestamp <= self.end_time()):
            raise ValueError(
                '"timestamps" should be between start and end of the sequence.'
            )
        first_sequence = self.filter_by_timestamp(
            self.start_time(), timestamp
        )
        second_sequence = self.filter_by_timestamp(timestamp, self.end_time())
        return first_sequence, second_sequence

    def make_iterator_over_splits(self, number_of_splits):
        """
            在拆分上生成迭代器。
            例如，如果“number_of_splits”=3，那么迭代器将输出

                (t_start->t_0, t_0->t_end)
                (t_start->t_1, t_1->t_end)
                (t_start->t_2, t_2->t_end)

                ---|------|------|------|------|--->
                 t_start  t0     t1    t2     t_end

            t0 = (t_end - t_start) / (number_of_splits + 1), and ect.
        """
        start_time = self.start_time()
        end_time = self.end_time()
        split_timestamps = np.linspace(start_time, end_time, number_of_splits + 2)[1:-1]

        for split_timestamp in split_timestamps:
            left_events, right_events = self.split_in_two(split_timestamp)
            yield left_events, right_events

    def make_sequential_iterator(self, timestamps):
        """
            生成顺序迭代器。
            Args:
                timestamps: list of timestamps that specify bining of events into the sub-sequences.
                            E.g. iterator will return events:
                            from timestamps[0] to timestamps[1],
                            from timestamps[1] to timestamps[2], and e.c.t.
        """
        if len(timestamps) < 2:
            raise ValueError("There should be at least two timestamps")
        start_timestamp = timestamps[0]
        start_index = self._advance_index_to_timestamp(start_timestamp)

        for end_timestamp in timestamps[1:]:
            end_index = self._advance_index_to_timestamp(end_timestamp, start_index)
            # self._features[start_index:end_index, :].size == 0
            yield EventSequence(
                features=np.copy(self._features[start_index:end_index, :]),
                image_height=self._image_height,
                image_width=self._image_width,
                start_time=start_timestamp,
                end_time=end_timestamp,
            )
            start_index = end_index
            start_timestamp = end_timestamp

    def format_events_list(self):
        x = self._features[:, 0]
        y = self._features[:, 1]
        ts = self._features[:, 2]
        p = self._features[:, 3]
        ts = (ts - self.start_time()) / (self.end_time() - self.start_time() + 1e-9)
        events = np.stack((ts, y, x, p), axis=-1)
        events = torch.from_numpy(events)
        return events

    def to_folder(self, folder, timestamps, event_file_template="{:06d}"):
        """将事件处理序列保存到npz。.

        Args:
            folder: folder where events will be saved in the files events_000000.npz,
                    events_000001.npz, etc.
            timestamps: iterator that outputs eventprocessing sequences.
        """
        event_iterator = self.make_sequential_iterator(timestamps)
        for sequence_index, sequence in enumerate(event_iterator):
            filename = os.path.join(folder, event_file_template.format(sequence_index))
            save_events(sequence._features, filename)

    @classmethod
    def from_folder(cls, folder, image_height, image_width, event_file_template="*.npz"):
        filename_iterator = os_tools.make_glob_filename_iterator(
            os.path.join(folder, event_file_template)
        )
        filenames = [filename for filename in filename_iterator]
        return cls.from_npz_files(filenames, image_height, image_width)

    @classmethod
    def from_npz_files(cls, list_of_filenames, image_height, image_width, start_time=None, end_time=None, hsergb=False, bsergb=False, hqf=False, size=None):
        """从numpy文件列表中读取事件处理序列。"""
        num = len(list_of_filenames)
        if num > 1:
            features_list = []
            for f in list_of_filenames:
                features_list += [load_events(f, hsergb, bsergb, hqf, size)]  # for filename in list_of_filenames
            features = np.concatenate(features_list)
        elif num == 1:
            features = load_events(list_of_filenames[0], hsergb, bsergb, hqf, size)
        else:
            return None
        if len(features) == 0:
            return None
        return EventSequence(features, image_height, image_width, start_time, end_time)