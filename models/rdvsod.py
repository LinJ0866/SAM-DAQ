# MIT License
# Copyright (c) 2025 LinJ0866
#
# This file is part of an extended version of the original MIT-licensed project.
# See the root LICENSE file for the original license.

import math
import torch

from torch import nn
from torch.nn import functional as F
from .sam2.build_sam import build_sam2_video_predictor

class rdvsod(nn.Module):
    def __init__(self, args, device):
        super().__init__()

        self.pix_feat = None
        self.device = device
        
        # adapter to project the depth image to proper rgb space
        self.encoder_adapter0 = nn.Sequential(nn.Conv2d(1, 3, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(3, 3, 1))
        
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]
        if 'sam2.1_hiera_l' in args.sam2_config:
            channel_list = [144, 288, 576, 1152]
        elif 'sam2.1_hiera_b+' in args.sam2_config:
            channel_list = [112, 224, 448, 896]
        elif 'sam2.1_hiera_s' in args.sam2_config:
            channel_list = [96, 192, 384, 768]
        elif 'sam2.1_hiera_t' in args.sam2_config:
            channel_list = [96, 192, 384, 768]

        # adapter to finetune the image encoder and fuse the rgb and depth features
        self.encoder_adapter0_ = nn.Sequential(nn.Conv2d(3, 64, (3, 3), padding=(1, 1), bias=True),
                                 nn.ReLU(),
                                 nn.Conv2d(64, channel_list[0], 1),
                                 nn.ReLU())
        self.encoder_adapter1 = bi_modal_parallel_adapter(channel_list[0], channel_list[1])
        self.encoder_adapter2 = bi_modal_parallel_adapter(channel_list[1], channel_list[2])
        self.encoder_adapter3 = bi_modal_parallel_adapter(channel_list[2], channel_list[3])
        self.encoder_adapter4 = bi_modal_parallel_adapter(channel_list[3], channel_list[3])
        
        self.mixer0 = nn.Sequential(nn.Conv2d(32, 32, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(32, 32, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(32, 1, (1, 1), bias=True),
                                      nn.Sigmoid())

        self.mixer1 = nn.Sequential(nn.Conv2d(64, 32, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(32, 32, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(32, 1, (1, 1), bias=True),
                                      nn.Sigmoid())

        self.mixer2 = nn.Sequential(nn.Conv2d(256, 128, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(128, 32, (3, 3), padding=(1, 1), bias=True),
                                      nn.ReLU(),
                                      nn.Conv2d(32, 1, (1, 1), bias=True),
                                      nn.Sigmoid())
        

        self.sam2 = build_sam2_video_predictor(config_file=args.sam2_config, ckpt_path=args.sam2_ckpt,
                               device=device, mode='train')

        self.hidden_dim = 64
        self.num_frame_queries = args.num_frame_queries
        self.frame_queries = nn.Embedding(self.num_frame_queries, self.hidden_dim)

        self.frame_queires_embed = query_embedding(self.hidden_dim)
        if args.enable_memory:
            self.num_video_queries = args.num_video_queries
            self.video_queries = nn.Embedding(self.num_video_queries, self.hidden_dim)

            self.frame2video_embed = nn.Sequential(nn.Conv2d(self.num_frame_queries, 256, 1),
                                        nn.ReLU(),)
            self.video_queires_embed = query_embedding(self.hidden_dim)
            
            self.video_cross_attention = cross_attention(4, 256)

            self.transformer_cross_attention = cross_attention(8, self.hidden_dim)
            self.transformer_self_attention = cross_attention(8, self.hidden_dim)
            self.transformer_ffn = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
            )
            
            self.memory_proj = nn.Sequential(
                nn.Linear(64, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim)
            )

        self.sparse_embed = nn.Linear(self._bb_feat_sizes[-1][0]*self._bb_feat_sizes[-1][1], 256)
    
    def forward(self, imgs, depths, is_mem=True, is_training=True, current_ti=0):
        frames_pred = []

        bsz, frames, _, h, w = imgs.shape
        imgs = imgs.view(-1, 3, h, w).contiguous()
        depths = depths.view(-1, 1, h, w).contiguous()

        d_proj = self.encoder_adapter0(depths)
        # d_proj = depths.repeat(1, 3, 1, 1)
        feats_img, feats_d = self.sam2_image_encoder(imgs, d_proj)

        # auxiliary supervision
        if is_training:
            intermediate_mask_stage2 = self.mixer2(feats_img[2])
            intermediate_mask_stage2 = intermediate_mask_stage2.squeeze(1).view(bsz, frames, self._bb_feat_sizes[2][0], self._bb_feat_sizes[2][1]).contiguous()
            
            intermediate_mask = [intermediate_mask_stage2]
        
        for i in range(3):
            size = self._bb_feat_sizes[i][0]

            feats_img[i] = feats_img[i].view(bsz, frames, -1, size, size).contiguous()
            feats_d[i] = feats_d[i].view(bsz, frames, -1, size, size).contiguous()

        if is_training:            
            if is_mem:
                self.video_query_weight = self.video_queries.weight.unsqueeze(1).repeat(1, bsz, 1).contiguous()
            dense_embeddings = self.sam2.sam_prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bsz, -1, self._bb_feat_sizes[-1][0], self._bb_feat_sizes[-1][1]
            )

            for ti in range(0, frames):
                high_res_features = [
                    feats_img[idx][:, ti]
                    for idx in range(2)
                ]

                frame_query_weight = self.frame_queries.weight.unsqueeze(1).repeat(1, bsz, 1).contiguous()
                frame_query_embeddings = self.frame_queires_embed(frame_query_weight)

                if is_mem:
                    frame_query_feat = torch.einsum("bqc,bchw->bqhw", frame_query_embeddings, feats_img[-1][:, ti]).contiguous()
                    frame_query_feat = self.frame2video_embed(frame_query_feat)
                    video_query_embeddings = self.video_queires_embed(self.video_query_weight)

                    # fuse the frame information to the video query
                    # frame_query_embeddings: [bsz, num_queries, hidden_dim]
                    video_query_embeddings = self.video_cross_attention(
                        video_query_embeddings.transpose(1, 2).contiguous(),
                        frame_query_feat
                    ).transpose(1, 2).contiguous()
                else:
                    video_query_embeddings = frame_query_embeddings

                video_embed_with_current_feat = torch.einsum("bqc,bchw->bqhw", video_query_embeddings, feats_img[-1][:, ti]).contiguous()
                # dense_embeddings = self.dense_embed(video_embed_with_current_feat)
                if is_mem:
                    sparse_embeddings = self.sparse_embed(video_embed_with_current_feat.view(bsz, self.num_video_queries, -1).contiguous())
                else:
                    sparse_embeddings = self.sparse_embed(video_embed_with_current_feat.view(bsz, self.num_frame_queries, -1).contiguous())


                # sparse_embeddings: [bsz, num, 256]
                # dense_embeddings: [bsz, 256, 28, 28]
                low_res_multimasks, _, _, _ = self.sam2.sam_mask_decoder(
                    image_embeddings=feats_img[-1][:, ti],
                    image_pe=self.sam2.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    repeat_image=False,
                    high_res_features=high_res_features,
                )
                low_res_multimasks = low_res_multimasks.float()
                sam_high_mask = F.interpolate(
                    low_res_multimasks,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )
                sam_high_mask = torch.sigmoid(sam_high_mask)
                frames_pred.append(sam_high_mask)

                # 更新frame_query(_feat)
                if is_mem:
                    self.update_video_queries(feats_img[-1][:, ti], sam_high_mask)

            return intermediate_mask, torch.stack(frames_pred, dim=0).squeeze(2).transpose(0, 1).contiguous()
        else:
            if is_mem and current_ti == 0:
                self.video_query_weight = self.video_queries.weight.unsqueeze(1).repeat(1, bsz, 1).contiguous()
            
            # sparse_embeddings = torch.empty(
            #     (bsz, 0, 256), device=self.device
            # )
            dense_embeddings = self.sam2.sam_prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bsz, -1, self._bb_feat_sizes[-1][0], self._bb_feat_sizes[-1][1]
            )
            high_res_features = [
                feats_img[idx][:, 0]
                for idx in range(2)
            ]

            frame_query_weight = self.frame_queries.weight.unsqueeze(1).repeat(1, bsz, 1).contiguous()
            frame_query_embeddings = self.frame_queires_embed(frame_query_weight)

            if is_mem:
                frame_query_feat = torch.einsum("bqc,bchw->bqhw", frame_query_embeddings, feats_img[-1][:, 0]).contiguous()
                frame_query_feat = self.frame2video_embed(frame_query_feat)
                video_query_embeddings = self.video_queires_embed(self.video_query_weight)

                # fuse the frame information to the video query
                # frame_query_embeddings: [bsz, num_queries, hidden_dim]
                video_query_embeddings = self.video_cross_attention(
                    video_query_embeddings.transpose(1, 2).contiguous(),
                    frame_query_feat
                ).transpose(1, 2).contiguous()
            else:
                video_query_embeddings = frame_query_embeddings

            video_embed_with_current_feat = torch.einsum("bqc,bchw->bqhw", video_query_embeddings, feats_img[-1][:, 0]).contiguous()
            # dense_embeddings = self.dense_embed(video_embed_with_current_feat)
            if is_mem:
                sparse_embeddings = self.sparse_embed(video_embed_with_current_feat.view(bsz, self.num_video_queries, -1).contiguous())
            else:
                sparse_embeddings = self.sparse_embed(video_embed_with_current_feat.view(bsz, self.num_frame_queries, -1).contiguous())

            low_res_multimasks, _, _, _ = self.sam2.sam_mask_decoder(
                image_embeddings=feats_img[-1][:, 0],
                image_pe=self.sam2.sam_prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                repeat_image=False,
                high_res_features=high_res_features,
            )

            low_res_multimasks = low_res_multimasks.float()
            sam_high_mask = F.interpolate(
                low_res_multimasks,
                size=(h, w),
                mode="bilinear",
                align_corners=False,
            )
            sam_high_mask = torch.sigmoid(sam_high_mask)

            if is_mem:
                self.update_video_queries(feats_img[-1][:, 0], sam_high_mask)

            return sam_high_mask

    
    def update_video_queries(self, pix_feat, pred_masks_high_res, alpha=0.1):
        maskmem_out = self.sam2.memory_encoder(
            pix_feat, pred_masks_high_res, skip_mask_sigmoid=True  # sigmoid already applied
        )
        maskmem_feature = maskmem_out["vision_features"] # bsz, 64, 28, 28
        maskmem_feature = self.memory_proj(maskmem_feature.flatten(2).permute(0, 2, 1).contiguous())

        # self.video_query_weight [num, bsz, hidden_dim]
        query_feat = self.transformer_cross_attention(self.video_query_weight.permute(1, 2, 0).contiguous(), maskmem_feature.permute(0, 2, 1).contiguous())
        query_feat = self.transformer_self_attention(query_feat, query_feat)

        query_feat = self.transformer_ffn(query_feat.permute(0, 2, 1).contiguous())
        self.video_query_weight = self.video_query_weight * alpha + query_feat.permute(1, 0, 2).contiguous()
        
        return

    '''
    3-level features
    - [B, 32, 112, 112]
    - [B, 64, 56, 56]
    - [B, 256, 28, 28]
    ''' 
    def sam2_image_encoder(self, rgb, d):
        batch_size = rgb.shape[0]

        assert (
            self.sam2.image_encoder.trunk.channel_list == self.sam2.image_encoder.neck.backbone_channel_list
        ), f"Channel dims of trunk and neck do not match. Trunk: {self.sam2.image_encoder.trunk.channel_list}, neck: {self.sam2.image_encoder.neck.backbone_channel_list}"

        """
        trunk in image encoder has 4 levels of features:
        - feat torch.Size([B, 112, 112, 112])
        - feat torch.Size([B, 224, 56, 56])
        - feat torch.Size([B, 448, 28, 28])
        - feat torch.Size([B, 896, 14, 14])
        """
        # add encoder adapter to the last block of each level of features
        with torch.no_grad():
            feat_rgb_0 = self.sam2.image_encoder.trunk(rgb, 0)
            feat_d_0 = self.sam2.image_encoder.trunk(d, 0)
        
        feat_d_0_skip = self.encoder_adapter0_(F.interpolate(d, size=feat_d_0.shape[-2:], mode='bilinear'))
        feat_d_0 = feat_d_0 + feat_d_0_skip

        with torch.no_grad():
            feat_rgb_1 = self.sam2.image_encoder.trunk(feat_rgb_0, 1)
            feat_d_1 = self.sam2.image_encoder.trunk(feat_d_0, 1)

        # parallel adapter
        feat_rgb_0_skip, feat_d_0_skip = self.encoder_adapter1(feat_rgb_0, feat_d_0) # parallel adapter
        feat_rgb_0_skip = F.interpolate(feat_rgb_0_skip, size=feat_rgb_1.shape[-2:], mode='bilinear')
        feat_d_0_skip = F.interpolate(feat_d_0_skip, size=feat_rgb_1.shape[-2:], mode='bilinear')
        feat_rgb_1 = feat_rgb_1 + feat_rgb_0_skip
        feat_d_1 = feat_d_1 + feat_d_0_skip

        with torch.no_grad():
            feat_rgb_2 = self.sam2.image_encoder.trunk(feat_rgb_1, 2)
            feat_d_2 = self.sam2.image_encoder.trunk(feat_d_1, 2)

        # parallel adapter
        feat_rgb_1_skip, feat_d_1_skip = self.encoder_adapter2(feat_rgb_1, feat_d_1) # parallel adapter
        feat_rgb_1_skip = F.interpolate(feat_rgb_1_skip, size=feat_rgb_2.shape[-2:], mode='bilinear')
        feat_d_1_skip = F.interpolate(feat_d_1_skip, size=feat_rgb_2.shape[-2:], mode='bilinear')
        feat_rgb_2 = feat_rgb_2 + feat_rgb_1_skip
        feat_d_2 = feat_d_2 + feat_d_1_skip

        with torch.no_grad():
            feat_rgb_3 = self.sam2.image_encoder.trunk(feat_rgb_2, 3)
            feat_d_3 = self.sam2.image_encoder.trunk(feat_d_2, 3)

        # parallel adapter
        feat_rgb_2_skip, feat_d_2_skip = self.encoder_adapter3(feat_rgb_2, feat_d_2) # parallel adapter
        feat_rgb_2_skip = F.interpolate(feat_rgb_2_skip, size=feat_rgb_3.shape[-2:], mode='bilinear')
        feat_d_2_skip = F.interpolate(feat_d_2_skip, size=feat_rgb_3.shape[-2:], mode='bilinear')
        feat_rgb_3 = feat_rgb_3 + feat_rgb_2_skip
        feat_d_3 = feat_d_3 + feat_d_2_skip

        # adapter
        feat_rgb_3_skip, _ = self.encoder_adapter4(feat_rgb_3, feat_d_3)
        feat_rgb_3 = feat_rgb_3 + feat_rgb_3_skip
        
        feats_rgb = [feat_rgb_0, feat_rgb_1, feat_rgb_2, feat_rgb_3]
        feats_d = [feat_d_0, feat_d_1, feat_d_2, feat_d_3]

        feats_rgb = self.prepare_backbone_features(feats_rgb, batch_size)
        feats_d = self.prepare_backbone_features(feats_d, batch_size)

        return feats_rgb, feats_d

    def prepare_backbone_features(self, feats, batch_size):
        # sam2.modeling.backbones.image_encoder.py ImageEncoder
        features, pos = self.sam2.image_encoder.neck(feats)
        if self.sam2.image_encoder.scalp > 0:
            # Discard the lowest resolution features
            features, pos = features[: -self.sam2.image_encoder.scalp], pos[: -self.sam2.image_encoder.scalp]

        src = features[-1]
        backbone_out = {
            "vision_features": src,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }

        # sam2.modeling.sam2_base.py SAM2Base.forward_image
        if self.sam2.use_high_res_features_in_sam:
            # precompute projected level 0 and level 1 features in SAM decoder
            # to avoid running it again on every SAM click
            backbone_out["backbone_fpn"][0] = self.sam2.sam_mask_decoder.conv_s0(
                backbone_out["backbone_fpn"][0]
            )
            backbone_out["backbone_fpn"][1] = self.sam2.sam_mask_decoder.conv_s1(
                backbone_out["backbone_fpn"][1]
            )
        _, vision_feats, _, _ = self.sam2._prepare_backbone_features(backbone_out)
        
        # Add no_mem_embed, which is added to the lowest rest feat. map during training on videos
        if self.sam2.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.sam2.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size).contiguous()
            for feat, feat_size in zip(vision_feats, self._bb_feat_sizes)
        ]

        return feats

class query_embedding(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.mask_embed = MLP(hidden_dim, hidden_dim, 256, 3)
    
    def forward(self, query):
        query = self.decoder_norm(query)
        query = query.transpose(0, 1).contiguous()
        query_embeddings = self.mask_embed(query)

        return query_embeddings

class bi_modal_parallel_adapter(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.fc_up = nn.Conv2d(self.in_channel * 2, 64, (3, 3), padding=(1, 1), bias=True)
        self.activate = nn.GELU()
        self.fc_down = nn.Conv2d(64, self.out_channel * 2, 1)
        self.skip = nn.Conv2d(self.in_channel * 2, self.out_channel * 2, 1)
    
    def forward(self, x1, x2):
        x = torch.concat([x1, x2], dim=1)
        out = self.fc_up(x)
        out = self.activate(out)
        out = self.fc_down(out)

        out1, out2 = torch.split(out, split_size_or_sections=self.out_channel, dim=1)
        return out1, out2

class cross_attention(nn.Module):
    def __init__(self, h, d_model):
        super().__init__()

        self.h = h
        self.d_model = d_model//h

        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)

    # dim = 3: [bsz, ch, l]
    # dim = 4: [bsz, ch, h, w]
    def forward(self, x1, x2):
        dim = 3
        if x1.dim() == 4:
            bsz, _, h, w = x1.shape
            dim = 4
        elif x1.dim() == 3:
            bsz, ch, _ = x1.shape
        x1 = x1.flatten(2).transpose(1, 2).contiguous()
        x2 = x2.flatten(2).transpose(1, 2).contiguous()
        # print(x1.shape, x2.shape)

        query = self.linear_q(x1).view(bsz, -1, self.h, self.d_model).transpose(1, 2).contiguous()
        key = self.linear_k(x2).view(bsz, -1, self.h, self.d_model).transpose(1, 2).contiguous()
        value = self.linear_v(x2).view(bsz, -1, self.h, self.d_model).transpose(1, 2).contiguous()

        scores = torch.matmul(
            query, key.transpose(-2, -1)
        ) / math.sqrt(query.shape[-1])

        output = torch.matmul(F.softmax(scores, dim=-1), value)
        # print(output.shape)
        # exit(0)
        if dim == 4:
            output = output.view(bsz, h, w, -1).permute(0, 3, 1, 2).contiguous()
        elif dim ==3:
            output = output.transpose(1, 2).contiguous().view(bsz, -1, ch).transpose(1, 2).contiguous()
        
        return output

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
