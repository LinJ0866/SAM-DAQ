CUDA_VISIBLE_DEVICES=0 torchrun ./test.py \
    --data_prefix /home/linj/workspace/vsod/datasets \
    --sam2_config configs/sam2.1/sam2.1_hiera_l.yaml \
    --sam2_ckpt ./sam2.1_hiera_large.pt \
    --load ./exps/RDVs_0322_140302/latest.pt \
    --task RDVS --num_frame_queries=30 --num_video_queries=8 --enable_memory

CUDA_VISIBLE_DEVICES=0 torchrun ./test.py \
    --data_prefix /home/linj/workspace/vsod/datasets \
    --sam2_config configs/sam2.1/sam2.1_hiera_l.yaml \
    --sam2_ckpt ./sam2.1_hiera_large.pt \
    --load ./exps/ViDSOD_0323_140914/latest.pt \
    --task ViDSOD-100 --num_frame_queries=30 --num_video_queries=8 --enable_memory

CUDA_VISIBLE_DEVICES=0 torchrun ./test.py \
    --data_prefix /home/linj/workspace/vsod/datasets \
    --sam2_config configs/sam2.1/sam2.1_hiera_l.yaml \
    --sam2_ckpt ./sam2.1_hiera_large.pt \
    --load ./exps/DViSal_1120_091836/latest.pt \
    --task DViSal --num_frame_queries=30 --num_video_queries=8 --enable_memory
