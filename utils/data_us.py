# MIT License
# Copyright (c) 2025 LinJ0866
#
# This file is part of an extended version of the original MIT-licensed project.
# See the root LICENSE file for the original license.

import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
import os

class TransformRGBD:
    def __init__(self):
        self.img_transform = None
        self.gt_transform = None
        self.depth_transform = None

    def __call__(self, images, depths, masks):
        if len(images) != len(masks) or len(depths) != len(masks):
            raise Exception("length are not equal among image, depth and masks.")
        if self.img_transform is None or self.gt_transform is None or self.depth_transform is None:
            return images, depths, masks
        
        new_images = []
        new_depths = []
        new_masks = []

        for image, depth, mask in zip(images, depths, masks):
            new_images.append(self.img_transform(image))
            new_depths.append(self.depth_transform(depth))
            new_masks.append(self.gt_transform(mask))
            
        new_images = torch.stack(new_images)
        new_depths = torch.stack(new_depths)
        new_masks = torch.stack(new_masks)

        return new_images, new_depths, new_masks

class TransformRGBD_train(TransformRGBD):
    def __init__(self,
                 img_size=256):
        self.img_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.gt_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor()])
        self.depth_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor()])

class RGBDDataset(Dataset):
    # split = 'train' or 'test_all'
    def load_dvisal_split(self, split='train'):
        with open(os.path.join(self.data_root, split+'.txt'), mode='r') as f:
            subset = set(f.read().splitlines())
        return subset

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.frame_size

class RGBDTrnVideoDataset(RGBDDataset):
    def __init__(self, data_prefix, task,
                 img_size=256,
                 frame_num=10):

        self.videolists = []
        self.videoframes = {}

        self.task = task
        if task == 'RDVS':
            self.data_root = os.path.join(data_prefix, 'RDVS/train')
            self.im_dir = 'rgb'
            self.d_dir = 'Depth'
            self.gt_dir = 'ground-truth'
        elif task == 'ViDSOD-100':
            self.data_root = os.path.join(data_prefix, 'vidsod_100/train')
            self.im_dir = 'rgb'
            self.d_dir = 'depth'
            self.gt_dir = 'gt'
        elif task == 'DViSal':
            self.data_root = os.path.join(data_prefix, 'DViSal_dataset')
            self.im_dir = 'RGB'
            self.d_dir = 'Depth'
            self.gt_dir = 'GT'
        else:
            raise 'dataset is not support now.'
        
        self.joint_transform = TransformRGBD_train(img_size=img_size)
        self.frame_num = frame_num
        # self.mode = 0 # for VisT300 which tree structure is: trn/val->modal->video->frames

        if task == 'RDVS' or task == 'ViDSOD-100':
            vid_list = sorted(os.listdir(self.data_root))
            for vid in vid_list:
                frames = sorted(os.listdir(os.path.join(self.data_root, vid, self.im_dir)))
                # Filter out paths that are not images
                frames = [frame for frame in frames if frame.endswith('.png') or frame.endswith('.jpg')]

                # Filter out videos with less than frame_num frames
                if len(frames) < frame_num:
                    continue
                self.videoframes[vid] = frames
                self.videolists.append(vid)
        elif task == 'DViSal':
            vid_list = self.load_dvisal_split('train')
            self.data_root = os.path.join(self.data_root, 'data')
            count_frames_all = 0
            count_frames_annotated = 0
            for vid in vid_list:
                frames = sorted(os.listdir(os.path.join(self.data_root, vid, self.im_dir)))
                frames_gt_temp = sorted(os.listdir(os.path.join(self.data_root, vid, self.gt_dir)))
                # Filter out paths that are not images
                frames = [frame for frame in frames if frame.endswith('.png') or frame.endswith('.jpg')]
                count_frames_all += len(frames)
                
                # Filter out frames without GT annotations
                frames = [frame for frame in frames if frame.replace('.jpg', '.png') in frames_gt_temp]
                count_frames_annotated += len(frames)

                # Filter out videos with less than frame_num frames
                if len(frames) < frame_num:
                    continue
                self.videoframes[vid] = frames
                self.videolists.append(vid)
            print('[DViSal] Total frames: %d, annotated frames: %d' % (count_frames_all, count_frames_annotated))

        print('%d out of %d videos accepted in %s.' % (len(self.videolists), len(vid_list), self.data_root))
        self.frame_size = len(self.videolists)

    def __getitem__(self, idx):
        labels = []
        images = []
        depths = []

        video = self.videolists[idx]
        filenames = self.videoframes[video]
        chosen_filenames = []

        vid_im_path = os.path.join(self.data_root, video, self.im_dir)
        vid_d_path = os.path.join(self.data_root, video, self.d_dir)
        vid_gt_path = os.path.join(self.data_root, video, self.gt_dir)

        if self.frame_num != -1:
            frame_idxs = np.random.choice(range(len(filenames)), size=self.frame_num, replace=False)
            frame_idxs.sort()
        else:
            frame_idxs = range(len(filenames))

        for idx in frame_idxs:
            filename = filenames[idx]
            chosen_filenames.append(filename)
            images.append(self.rgb_loader(os.path.join(vid_im_path, filename)))
            depths.append(self.binary_loader(os.path.join(vid_d_path, filename.replace('.jpg', '.png'))))
            labels.append(self.binary_loader(os.path.join(vid_gt_path, filename.replace('.jpg', '.png'))))
        
        shape = (images[0].size[1], images[0].size[0])

        if self.joint_transform:
            images, depths, labels = self.joint_transform(images, depths, labels)
        labels = labels.squeeze(1)
    
        return {
            'image': images,
            'depth': depths,
            'label': labels,
            'file_name': chosen_filenames, # need transpose
            'video_name': video,
            'frames': self.frame_num,
            'ori_size': shape, # need transpose
        }
    
class RGBDTrnVideoFirstStageDataset(RGBDTrnVideoDataset):
    def __init__(self, data_prefix, task,
                 img_size=256,
                 frame_num=10):
        super(RGBDTrnVideoFirstStageDataset, self).__init__(data_prefix, task, img_size, frame_num)

        videolists = {
            'frames': [],
            'frames_vid': [],
        }
        for vid in self.videolists:
            videolists['frames'].extend(self.videoframes[vid])
            videolists['frames_vid'].extend([vid] * len(self.videoframes[vid]))
        
        self.videolists = videolists
    
    def __len__(self):
        return len(self.videolists['frames_vid'])
    
    def __getitem__(self, idx):
        labels = []
        images = []
        depths = []

        video = self.videolists['frames_vid'][idx]
        filenames = self.videoframes[video]
        chosen_filenames = [self.videolists['frames'][idx]]

        vid_im_path = os.path.join(self.data_root, video, self.im_dir)
        vid_d_path = os.path.join(self.data_root, video, self.d_dir)
        vid_gt_path = os.path.join(self.data_root, video, self.gt_dir)
        
        while True:
            idx = np.random.choice(range(1, len(filenames)), 1, replace=False)[0]
            support_name = filenames[idx]
            if support_name not in chosen_filenames:
                chosen_filenames.append(support_name)          
        
            if len(chosen_filenames) >= 2:
                break
            

        for filename in chosen_filenames:
            images.append(self.rgb_loader(os.path.join(vid_im_path, filename)))
            depths.append(self.binary_loader(os.path.join(vid_d_path, filename.replace('.jpg', '.png'))))
            labels.append(self.binary_loader(os.path.join(vid_gt_path, filename.replace('.jpg', '.png'))))
        
        shape = (images[0].size[1], images[0].size[0])

        if self.joint_transform:
            images, depths, labels = self.joint_transform(images, depths, labels)
        labels = labels.squeeze(1)
    
        return {
            'image': images,
            'depth': depths,
            'label': labels,
            'file_name': chosen_filenames, # need transpose
            'video_name': video,
            'frames': self.frame_num,
            'ori_size': shape, # need transpose
        }

    
class RGBDTestVideoDataset(RGBDDataset):
    def __init__(self, data_prefix, task,
                 img_size=256):

        self.videolists = []

        self.task = task
        if task == 'RDVS':
            self.data_root = os.path.join(data_prefix, 'RDVS/test')
            self.im_dir = 'rgb'
            self.d_dir = 'Depth'
            self.gt_dir = 'ground-truth'
        elif task == 'ViDSOD-100':
            self.data_root = os.path.join(data_prefix, 'vidsod_100/test')
            self.im_dir = 'rgb'
            self.d_dir = 'depth'
            self.gt_dir = 'gt'
        elif task == 'DViSal':
            self.data_root = os.path.join(data_prefix, 'DViSal_dataset')
            self.im_dir = 'RGB'
            self.d_dir = 'Depth'
            self.gt_dir = 'GT'
        else:
            raise 'dataset is not support now.'
        
        # self.mode = 0
        self.img_size = img_size

        if task == 'RDVS' or task == 'ViDSOD-100':
            self.videolists = sorted(os.listdir(self.data_root))
        elif task == 'DViSal':
            self.videolists = self.load_dvisal_split('test_all')
            self.data_root = os.path.join(self.data_root, 'data')
        
        print('%d videos accepted in %s.' % (len(self.videolists), self.data_root))
        
        self.frame_size = len(self.videolists)

    def get_datasets(self):
        for video in self.videolists:
            vid_im_path = os.path.join(self.data_root, video, self.im_dir)
            vid_d_path = os.path.join(self.data_root, video, self.d_dir)
            vid_gt_path = os.path.join(self.data_root, video, self.gt_dir)
            
            yield VideoReader(video,
                              vid_im_path,
                              vid_d_path,
                              vid_gt_path,
                              img_size=self.img_size
            )

class VideoReader(RGBDDataset):
    def __init__(self, vid_name, im_root, d_root, gt_root,
                 img_size=256):
        self.vid_name = vid_name

        self.im_root = im_root
        self.d_root = d_root
        self.gt_root = gt_root
        frames = sorted(os.listdir(im_root))
        # Filter out paths that are not images
        frames = [frame for frame in frames if frame.endswith('.png') or frame.endswith('.jpg')]

        # Filter out frames without GT annotations
        frames_gt_temp = os.listdir(self.gt_root)
        frames = [frame for frame in frames if frame.replace('.jpg', '.png') in frames_gt_temp]

        self.frames = frames
        
        self.frame_size = len(self.frames)
        self.joint_transform = TransformRGBD_train(img_size=img_size)
    
    def __getitem__(self, idx):
        png_filename = self.frames[idx].replace('.jpg', '.png')

        image = self.rgb_loader(os.path.join(self.im_root, self.frames[idx]))
        depth = self.binary_loader(os.path.join(self.d_root, png_filename))
        label = self.binary_loader(os.path.join(self.gt_root, png_filename))
        shape = (image.size[1], image.size[0])

        if self.joint_transform:
            image, depth, label = self.joint_transform([image], [depth], [label])
            label = label.squeeze(1)
    
        return {
            'image': image,
            'depth': depth,
            'label': label,
            'shape': shape,
            'video_name': self.vid_name,
            'frame': png_filename,
        }
