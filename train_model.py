import os
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from lpips import LPIPS
from datasets import get_dataset
from models.model import MyModel
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR

def load_weights(args, model, dataloader):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, betas=[0.9, 0.999])
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epoch * len(dataloader), eta_min=1e-7)
    epoch = 0
    if args.weights_path is not None:
        state = torch.load(args.weights_path)
        if args.only_weights:
            model_dict = {k.replace('module.', ''): v for k, v in state['model'].items()}
            missing_keys, unexpected_keys = model.load_state_dict(model_dict, strict=False)
            print("Missing:", missing_keys)
            print("Unexpected:", unexpected_keys)
        else:
            model_dict = {k.replace('module.', ''): v for k, v in state['model'].items()}
            opt_dict = state['optimizer']
            sche_dict = state['scheduler']
            epoch = state['epoch'] + 1
            model.load_state_dict(model_dict)
            optimizer.load_state_dict(opt_dict)
            scheduler.load_state_dict(sche_dict)
    return model, optimizer, scheduler, epoch


def train(model, args):
    model = model.cuda()
    dataset = get_dataset(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_worker, shuffle=True, drop_last=True, pin_memory=True)
    model, optimizer, scheduler, epoch = load_weights(args, model, dataloader)
    lps = LPIPS(net="vgg").cuda()

    while epoch < args.epoch:
        sum_loss = 0
        bar = tqdm(dataloader)
        bar.set_description('Train Epoch({}/{})'.format(epoch, args.epoch))
        for i, data in enumerate(bar):
            imgs, voxel_01, times = data
            num = len(times[0])
            imgs = imgs.cuda()
            img0 = imgs[:, :3]
            gts = imgs[:, 3:(num + 1) * 3]
            img1 = imgs[:, (num + 1) * 3:]
            voxel_01 = voxel_01.cuda()
            times = times.cuda()

            imst = model(img0, img1, voxel_01, times)
            gts = torch.stack(gts.split(3, dim=1), dim=1)
            b, t, c, h, w = imst.shape

            optimizer.zero_grad()
            gts = gts.view(b * t, c, h, w)
            imst = imst.view(b * t, c, h, w)
            loss_p = 0.1 * lps(gts, imst)
            loss_1 = torch.sqrt((gts - imst) ** 2 + 1e-6)
            loss = loss_p.mean() + torch.mean(loss_1)
            loss.backward()

            optimizer.step()
            scheduler.step()
            learn_rate = optimizer.param_groups[0]['lr']
            sum_loss += loss.item()
            bar.set_postfix({'sum_loss': f'{sum_loss:.4f}', 'learn_rate': f'{learn_rate:.6f}'})

        os.makedirs(args.save_dir, exist_ok=True)
        path = os.path.join(args.save_dir, "epoch{}.pth".format(epoch))
        cpk = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch
        }
        torch.save(cpk, path)
        epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # train settings
    parser.add_argument('--epoch', default=10, type=int)
    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--only_weights', action='store_true', help='only load model weights')
    parser.add_argument('--weights_path', default=None, type=str, help='weights path')
    parser.add_argument('--save_dir', default='train', type=str, help='train save_dir')
    parser.add_argument('--seed', default=1234, type=int, help='all random seed')
    # data settings
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--num_worker', default=8, type=int, help='num worker')
    parser.add_argument('--dataset', default='gopro', type=str, help='v90k gopro')
    parser.add_argument('--voxel_bins', default=128, type=int, help='event voxel bins')
    # model settings
    parser.add_argument('--nb_of_flow', default=16, type=int)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    model = MyModel(args)
    all_params = sum(p.numel() for p in model.parameters())
    print('Total params:', all_params)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Trainable params:', trainable)
    train(model, args)
