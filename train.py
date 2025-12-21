# MIT License
# Copyright (c) 2025 LinJ0866
#
# This file is part of an extended version of the original MIT-licensed project.
# See the root LICENSE file for the original license.

import os
import argparse
import datetime
import numpy as np
import torch
import torch.optim as optim

from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler as Sampler

from models.rdvsod import rdvsod
from utils.data_us import RGBDTrnVideoDataset
from utils.loss import get_criterion
from utils.tools import to_cuda, fix_randseed
from utils.logger import Logger

def train(epoch, model, dataloader, optimizer, training, criterion, args):
    r""" Train """

    # Force randomness during training / freeze randomness during testing
    fix_randseed(None) if training else fix_randseed(0)
    model.train() if training else model.eval()
    
    loop = tqdm(dataloader)
    loss_all_list = []
    loss_mask_list = []
    loss_dm_list = []
    for datapack in loop:
        filenames = np.array(datapack['file_name']).transpose()
        ori_sizes = np.array(datapack['ori_size']).transpose()
        # 1. forward pass
        datapack = to_cuda(datapack)

        imgs = datapack['image']

        depths = datapack['depth']
        masks = datapack['label']

        coarse_mask_by_dense_matching, pred = model(imgs, depths, is_mem=args.enable_memory, is_training=True)

        loss_dm = 0
        for coarse_mask in coarse_mask_by_dense_matching:
            loss_dm = loss_dm + criterion(coarse_mask, masks)

        loss_mask = criterion(pred, masks)
        train_loss = loss_mask + 0.5 * loss_dm

        loss_mask_list.append(loss_mask.detach())
        loss_all_list.append(train_loss.detach())
        loss_dm_list.append(loss_dm.detach())
        
        if training:
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        
        loop.set_postfix({
            'avg_loss': torch.stack(loss_all_list).mean().item(),
            'avg_loss_mask': torch.stack(loss_mask_list).mean().item(),
            'avg_loss_dm': torch.stack(loss_dm_list).mean().item(),
        })
        break
    return torch.stack(loss_all_list).mean().item(), torch.stack(loss_mask_list).mean().item(), torch.stack(loss_dm_list).mean().item()


def main():

    #  ============================================================================= parameters setting ====================================================================================
    parser = argparse.ArgumentParser(description='SAM-DAQ Training')
    parser.add_argument('--data_prefix', default='/opt/vsod/datasets', type=str, help='dataset path prefix')
    parser.add_argument('--task', default='RDVS', help='name of dataset: RDVS, ViDSOD-100 or DViSal')
    parser.add_argument('--iters', type=int, default=3000, help='epoch number for training')
    parser.add_argument('--sam2_config', type=str, default='configs/sam2.1/sam2.1_hiera_b+.yaml', help='Pretrained checkpoint of SAM2')
    parser.add_argument('--sam2_ckpt', type=str, default='./sam2.1_hiera_base_plus.pt', help='Pretrained checkpoint of SAM2')
    parser.add_argument('--batch_size', type=int, default=1, help='batch_size per gpu') # SAMed is 12 bs with 2n_gpu and lr is 0.005
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--local-rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--base_lr', type=float, default=0.001, help='segmentation network learning rate, 0.005 for SAMed, 0.0001 for MSA') #0.0006
    parser.add_argument('--warmup', action="store_true", help='If activated, warp up the learning from a lower lr to the base_lr') 
    parser.add_argument('--warmup_period', type=int, default=10, help='Warp up iterations, only valid whrn warmup is activated')
    parser.add_argument('--logpath', type=str, default=None)
    parser.add_argument('--frame_length', type=int, default=10)
    parser.add_argument('--num_frame_queries', type=int, default=30)
    parser.add_argument('--num_video_queries', type=int, default=8)
    parser.add_argument('--enable_memory', action="store_true")
    args = parser.parse_args()
    print(args)

    if args.logpath is None:
        args.logpath = "./exps/%s/%s"%(args.task, datetime.datetime.now().strftime("%m%d_%H%M%S"))
    data_prefix = args.data_prefix

    # ddp backend initialization
    torch.distributed.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    # ==================================================build model==================================================
    device = torch.device("cuda", args.local_rank)
    model = rdvsod(args, device=device).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
                                                find_unused_parameters=True)
    
    batch_size = args.batch_size * args.n_gpu

    if args.local_rank == 0:
        Logger.initialize(args, training=True)
        Logger.info('# available GPUs: %d' % torch.cuda.device_count())

    train_dataset = RGBDTrnVideoDataset(data_prefix, args.task,
                                        img_size=1024,
                                        frame_num=args.frame_length)
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, sampler=Sampler(train_dataset), num_workers=8, pin_memory=True)
    
    for name, param in model.named_parameters():
        if "sam2.image_encoder" in name \
            or "sam2.sam_prompt_encoder" in name \
            or "sam2.memory_attention" in name:
            param.requires_grad = False

    if args.warmup:
        b_lr = args.base_lr / args.warmup_period
        optimizer = optim.AdamW(filter(lambda p : p.requires_grad, model.parameters()), lr=b_lr, weight_decay=0.1)
    else:
        optimizer = optim.AdamW(filter(lambda p : p.requires_grad, model.parameters()), lr=args.base_lr, weight_decay=0.05)

    criterion = get_criterion()

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total_params: {}".format(pytorch_total_params))
    print("learnable_params: {}".format(pytorch_grad_params))

    #  ========================================================================= begin to train the model ============================================================================
    iter_num = 0
    epoch_num = args.iters // len(trainloader) + 1
    max_iterations = epoch_num * len(trainloader)
    
    for epoch in range(epoch_num):
        print(f'epoch {epoch+1}/{epoch_num}')
        # ------------------------------------------- adjust the learning rate when needed-----------------------------------------
        if args.warmup and iter_num < args.warmup_period:
            lr_ = args.base_lr * ((iter_num + 1) / args.warmup_period)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_
        else:
            if args.warmup:
                shift_iter = iter_num - args.warmup_period
                assert shift_iter >= 0, f'Shift iter is {shift_iter}, smaller than zero'
                lr_ = args.base_lr * (1.0 - shift_iter / max_iterations) ** 0.9  # learning rate adjustment depends on the max iterations
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_
        iter_num = iter_num + 1

        #  --------------------------------------------------------- training ---------------------------------------------------------
        trainloader_ = trainloader
        trainloader_.sampler.set_epoch(epoch)
        trn_all_loss, trn_mask_loss, trn_dm_loss = train(epoch, model, trainloader_, optimizer, training=True, criterion=criterion, args=args)

        # --------------------------------------------------------- log and save --------------------------------------------------------
        state_dict = model.state_dict()
        # 过滤掉 sam2 的参数
        filtered_state_dict = {k[7:]: v for k, v in state_dict.items() 
                            if not k.startswith("module.sam2.image_encoder") and not k.startswith("module.sam2.sam_prompt_encoder") and not k.startswith("module.sam2.memory_attention")}
        torch.save(filtered_state_dict, os.path.join(args.logpath, 'latest.pt'))
        
        Logger.tbd_writer.add_scalars('data/loss', {
            'trn_all_loss': trn_all_loss,
            'trn_mask_loss': trn_mask_loss,
            'trn_dm_loss': trn_dm_loss
        }, epoch)
        Logger.tbd_writer.flush()

    if args.local_rank == 0:
        Logger.tbd_writer.close()
        Logger.info('==================== Finished Training ====================')


if __name__ == '__main__':
    main()