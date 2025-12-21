CUDA_VISIBLE_DEVICES=0 torchrun --master_port 34561 --nproc_per_node=1 ./train.py \
    --data_prefix /home/linj/workspace/vsod/datasets \
    --sam2_config configs/sam2.1/sam2.1_hiera_l.yaml  --sam2_ckpt ./sam2.1_hiera_large.pt \
    --task RDVS --frame_length=10 --num_frame_queries=30 --num_video_queries=8 \
    --iters 2000 --batch_size=1 --base_lr=0.0001 --enable_memory # --logpath exps/test

CUDA_VISIBLE_DEVICES=0 torchrun --master_port 34562 --nproc_per_node=1 ./train.py \
    --data_prefix /home/linj/workspace/vsod/datasets \
    --sam2_config configs/sam2.1/sam2.1_hiera_l.yaml  --sam2_ckpt ./sam2.1_hiera_large.pt \
    --task ViDSOD --frame_length=10 --num_frame_queries=30 --num_video_queries=8 \
    --iters 2000 --batch_size=1 --base_lr=0.0001 --enable_memory # --logpath exps/test

CUDA_VISIBLE_DEVICES=0 torchrun --master_port 34563 --nproc_per_node=1 ./train.py \
    --data_prefix /home/linj/workspace/vsod/datasets \
    --sam2_config configs/sam2.1/sam2.1_hiera_l.yaml  --sam2_ckpt ./sam2.1_hiera_large.pt \
    --task DViSal --frame_length=10 --num_frame_queries=30 --num_video_queries=8 \
    --iters 2000 --batch_size=1 --base_lr=0.0001 --enable_memory # --logpath exps/test
