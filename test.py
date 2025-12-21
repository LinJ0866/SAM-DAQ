# MIT License
# Copyright (c) 2025 LinJ0866
#
# This file is part of an extended version of the original MIT-licensed project.
# See the root LICENSE file for the original license.

import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader

from models.rdvsod import rdvsod
from utils.tools import to_cuda
from utils.data_us import RGBDTestVideoDataset

def save(mask, save_path, shape):
    mask = F.interpolate(mask, shape, mode='bilinear', align_corners=False)
    mask[mask>=0.5] = 255
    mask[mask<0.5] = 0
    # mask *= 255
    # print(torch.max(mask), torch.min(mask))

    out_img = Image.fromarray((mask[0][0].cpu().numpy()).astype(np.uint8))
    out_img.save(save_path)

def main():

    #  ============================================================================= parameters setting ====================================================================================
    parser = argparse.ArgumentParser(description='SAM-DAQ Inference')
    parser.add_argument('--data_prefix', default='/opt/vsod/datasets', type=str, help='dataset path prefix')
    parser.add_argument('--task', default='RDVS', help='name of dataset: RDVS, ViDSOD-100 or DViSal')
    parser.add_argument('--sam2_config', type=str, default='configs/sam2.1/sam2.1_hiera_b+.yaml', help='Pretrained checkpoint of SAM2')
    parser.add_argument('--sam2_ckpt', type=str, default='./sam2.1_hiera_base_plus.pt', help='Pretrained checkpoint of SAM2')
    parser.add_argument('--load_path', type=str, help='Pretrained checkpoint of SAM-DAQ')
    parser.add_argument('--local-rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--num_frame_queries', type=int, default=30)
    parser.add_argument('--num_video_queries', type=int, default=8)
    parser.add_argument('--enable_memory', action="store_true")
    args = parser.parse_args()
    print(args)

    data_prefix = args.data_prefix
    save_root = os.path.join(os.path.dirname(args.load_path), 'output')

    torch.cuda.set_device(args.local_rank)
    # ==================================================build model==================================================
    device = torch.device("cuda", args.local_rank)
    model = rdvsod(args, device=device).to(device)

    # load checkpoint
    new_state_dict = {}
    checkpoint = torch.load(args.load_path)
    sam2_state = torch.load(args.sam2_ckpt)
    # merge sam2 weights to checkpoint
    for k, v in sam2_state['model'].items():
        new_state_dict["sam2."+k] = v
    for k, v in checkpoint.items():
        new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict, strict=True)
    model.eval()

    meta_dataset = RGBDTestVideoDataset(data_prefix, args.task, img_size=1024)
    torch.autograd.set_grad_enabled(False)
    meta_loader = meta_dataset.get_datasets()
    
    vid_length = meta_dataset.frame_size
    with tqdm(total=vid_length) as loop:
        loop.set_description('Predict')
        for _ in range(vid_length):
            vid_reader = next(meta_loader)
            loader = DataLoader(vid_reader, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

            vid_name = vid_reader.vid_name
            frame_length = len(loader)

            this_out_path = os.path.join(save_root, vid_name)
            os.makedirs(this_out_path, exist_ok=True)

            for ti, data in enumerate(loader):
                with torch.no_grad():
                    datapack = to_cuda(data)

                    img = datapack['image']
                    depth = datapack['depth']
                    label = datapack['label']
                    shape = datapack['shape']
                    out_mask = model(img, depth, is_mem=args.enable_memory, is_training=False, current_ti=ti)
                    save(out_mask, os.path.join(this_out_path, datapack['frame'][0]), shape)

                    loop.set_postfix({
                        'vid': vid_name,
                        'all': '%d/%d' % (ti+1, frame_length)
                    })
            loop.update(1)


if __name__ == '__main__':
    main()