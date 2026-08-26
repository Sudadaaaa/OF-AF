import os
import cv2
import time
import torch
import random

import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from util.event import EventSequence
from util.voxelization import to_voxel_grid
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from lpips import LPIPS

class GOPRO_Train(Dataset):
    def __init__(self, voxel_bins, scene, skip_frame):
        self.h, self.w = 720, 1280
        self.crop = [256, 256]

        self.skip_frame = skip_frame
        self.cur_skip = skip_frame
        self.voxel_bins = voxel_bins

        self.data_root = '/DATASSD1/GOPRO/'
        self.data_path = os.path.join(self.data_root, 'train', scene)
        self.img_folder = os.path.join(self.data_path, 'imgs')
        self.img_names = os.listdir(os.path.join(self.data_path, 'imgs'))
        self.img_names.sort()

        self.event_folder = os.path.join(self.data_path, 'events_clean_8x_v2')
        self.event_names = os.listdir(self.event_folder)
        self.event_names.sort()
        with open(os.path.join(self.data_path, "timestamps_clean_8x_v2.txt"), 'r') as f:
            self.timestamps = f.read().splitlines()

    def __len__(self):
        len_nums = len(self.img_names) - self.skip_frame - 1
        return len_nums

    def get_img(self, index):
        ind = list(range(index, index+self.cur_skip+2))
        imgs = torch.cat([torch.from_numpy(cv2.cvtColor(cv2.imread(os.path.join(self.img_folder, self.img_names[i])), cv2.COLOR_BGR2RGB)) for i in ind], dim=2)
        return imgs, ind

    def get_events(self, ind, size=None):
        start_ind = self.timestamps.index(str(ind[0] / 240))
        end_ind = self.timestamps.index(str(ind[-1] / 240))
        events_names = [os.path.join(self.event_folder, i) for i in self.event_names[start_ind:end_ind]]
        events_01 = EventSequence.from_npz_files(events_names, self.crop[0], self.crop[1], size=size)
        return events_01

    def random_crop(self, imgs):
        h, w, _ = imgs.shape
        hs = random.randint(0, h - self.crop[0])
        ws = random.randint(0, w - self.crop[1])
        he = hs+self.crop[0]
        we = ws+self.crop[1]
        imgs = imgs[hs:he, ws:we, :]
        return imgs, [hs, ws, he, we]

    def __getitem__(self, index):
        imgs, ind = self.get_img(index)
        imgs, size = self.random_crop(imgs)
        events_01 = self.get_events(ind, size)

        if events_01 is None:
            voxel_01 = torch.zeros([self.voxel_bins, self.crop[0], self.crop[1]])
        else: 
            voxel_01 = to_voxel_grid(events_01, self.voxel_bins)
        imgs = imgs.permute(2, 0, 1)/255
        times = (torch.tensor(ind[1:-1])-ind[0])/(self.cur_skip+1)
        
        if random.uniform(0, 1) < 0.5:
            imgs = torch.flip(imgs, dims=[1])
            voxel_01 = torch.flip(voxel_01, dims=[1])
        if random.uniform(0, 1) < 0.5:
            imgs = torch.flip(imgs, dims=[2])
            voxel_01 = torch.flip(voxel_01, dims=[2])
        return imgs, voxel_01, times

class GOPRO_Test():
    def __init__(self, voxel_bins, skips=[7, 15], save_root='eval/gopro',save_pred=False):
        self.h, self.w = 720, 1280
        self.skips = skips
        self.voxel_bins = voxel_bins
        self.save_root = save_root
        self.save_pred = save_pred
        self.test_root = '/DATASSD1/GOPRO/test'
        self.all_scenes = ['GOPR0384_11_00', 'GOPR0384_11_05', 'GOPR0385_11_01', 'GOPR0396_11_00', 'GOPR0410_11_00',
                       'GOPR0854_11_00', 'GOPR0862_11_00', 'GOPR0868_11_00', 'GOPR0869_11_00', 'GOPR0871_11_00', 'GOPR0881_11_01']
        self.test_scenes = self.all_scenes
        self.lpips = LPIPS(net="alex").cuda()
        from flolpipsloss.flolpips import Flolpips
        self.flolpips = Flolpips().cuda()

    def init_scene(self, scene, skip):
        self.cur_skip = skip
        self.cur_scene = scene
        self.skip_dir = os.path.join(self.save_root, str(skip))
        self.save_dir = os.path.join(self.skip_dir, scene)
        os.makedirs(self.save_dir, exist_ok=True)
        self.log_path = os.path.join(self.save_dir, 'log.txt')

        self.img_folder = os.path.join(self.test_root, scene, 'imgs')
        self.img_names = os.listdir(self.img_folder)
        self.img_names.sort()
        self.img_paths = [os.path.join(self.img_folder, i) for i in self.img_names]
        self.event_folder = os.path.join(self.test_root, scene, 'events_clean_8x_v2')
        self.event_names = os.listdir(self.event_folder)
        self.event_names.sort()
        with open(os.path.join(self.test_root, scene, "timestamps_clean_8x_v2.txt"), 'r') as f:
            self.timestamps = f.read().splitlines()
    def get_img(self, index):
        ind = list(range(index, index+self.cur_skip+2))
        imgs = torch.cat([torch.from_numpy(cv2.cvtColor(cv2.imread(os.path.join(self.img_folder, self.img_names[i])), cv2.COLOR_BGR2RGB)) for i in ind], dim=2)
        return imgs, ind

    def get_events(self, ind, size=None):
        start_ind = self.timestamps.index(str(ind[0] / 240))
        end_ind = self.timestamps.index(str(ind[-1] / 240))
        events_names = [os.path.join(self.event_folder, i) for i in self.event_names[start_ind:end_ind]]
        events_01 = EventSequence.from_npz_files(events_names, self.h, self.w, size=size)
        return events_01

    def get_batch(self, index):
        imgs, ind = self.get_img(index)
        events_01 = self.get_events(ind)
        voxel_01 = to_voxel_grid(events_01, self.voxel_bins)
        imgs = imgs.permute(2, 0, 1)/255
        times = (torch.tensor(ind[1:-1])-ind[0])/(self.cur_skip+1)

        imgs = imgs.unsqueeze(0)
        voxel_01 = voxel_01.unsqueeze(0)
        times = times.unsqueeze(0)
        return imgs, voxel_01, times

    def cal_psnr(self, gt, pred):
        return peak_signal_noise_ratio(gt, pred, data_range=255)

    def cal_ssim(self, gt, pred):
        return structural_similarity(gt, pred, data_range=255, channel_axis=2)

    def cal_flolpips(self, img0, img1, pred, gt):
        flolpips = self.flolpips.forward(img0, img1, pred, gt)
        return flolpips.item()

    def cal_lpips(self, gt, pred):
        return self.lpips(gt, pred).item()

    def write_log(self, log, path):
        with open(path, 'a') as f:
            f.write(log + '\n')

    def test(self, model):
        model.eval()
        with torch.no_grad():
            for skip in self.skips:
                all_psnr = []
                all_ssim = []
                all_flolpips = []
                all_lpips = []
                for scene in self.test_scenes:
                    self.init_scene(scene, skip)
                    psnrs = []
                    ssims = []
                    flolpipss = []
                    lpipss = []
                    for index in tqdm(range(0, len(self.img_names)-self.cur_skip-1, skip), desc='Current: Test on {}_{}'.format(scene, skip)):
                        imgs, voxel01, times = self.get_batch(index)
                        imgs = imgs.cuda()
                        im0 = imgs[:, :3]
                        gts = imgs[:, 3:-3]
                        im1 = imgs[:, -3:]
                        voxel01 = voxel01.cuda()
                        times = times.cuda()
                        res = model(im0, im1, voxel01, times)

                        preds = torch.clamp(res, 0, 1)
                        gts = torch.stack(gts.split(3, dim=1), dim=1) #1, t, c, h, w
                        for i in range(1, skip, 2):
                            flolpips = self.cal_flolpips(im0, im1, preds[:, i], gts[:, i])
                            lpips = self.cal_lpips(preds[:, i], gts[:, i])
                            pred = (preds[0, i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                            gt = (gts[0, i].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                            psnr = self.cal_psnr(gt, pred)
                            ssim = self.cal_ssim(gt, pred)

                            log = "[{}_{}/{}]\tPSNR:{:.2f}\tSSIM:{:.4f}\tLPIPS:{:.4f}\tFloLPIPS:{:.4f}".format(scene, skip, self.img_names[index + i + 1], psnr, ssim, lpips, flolpips)
                            self.write_log(log, self.log_path)
                            if self.save_pred:
                                save_path = os.path.join(self.save_dir, self.img_names[index + i + 1])
                                cv2.imwrite(save_path, cv2.cvtColor(pred, cv2.COLOR_RGB2BGR))
                            psnrs.append(psnr)
                            ssims.append(ssim)
                            flolpipss.append(flolpips)
                            lpipss.append(lpips)
                            all_psnr.append(psnr)
                            all_ssim.append(ssim)
                            all_flolpips.append(flolpips)
                            all_lpips.append(lpips)
                    log = "[{}_{}]\tMean PSNR:{:.2f}\tMean SSIM:{:.4f}\tMean LPIPS:{:.4f}\tMean FloLPIPS:{:.4f}".format(scene, skip, np.mean(psnrs), np.mean(ssims), np.mean(lpipss), np.mean(flolpipss))
                    print(log)
                    self.write_log(log, self.log_path)
                    self.write_log(log, os.path.join(self.skip_dir, 'gopro.txt'))
                log = "[gopro_{}]\tMean PSNR:{:.2f}\tMean SSIM:{:.4f}\tMean LPIPS:{:.4f}\tMean FloLPIPS:{:.4f}".format(skip, np.mean(all_psnr), np.mean(all_ssim), np.mean(all_lpips), np.mean(all_flolpips))
                print(log)
                self.write_log(log, os.path.join(self.skip_dir, 'gopro.txt'))