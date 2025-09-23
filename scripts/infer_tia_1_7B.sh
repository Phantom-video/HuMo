#!/bin/bash

# Run on a single GPU
CUDA_VISIBLE_DEVICES=0 torchrun --node_rank=0 --nproc_per_node=1 --nnodes=1 \
    --rdzv_endpoint=127.0.0.1:12345 \
    --rdzv_conf=timeout=900,join_timeout=900,read_timeout=900 \
    main.py humo/configs/inference/generate_1_7B.yaml \
    dit.sp_size=1 \
    generation.frames=97 \
    generation.scale_t=7.0 \
    generation.scale_i=4.0 \
    generation.scale_a=7.5 \
    generation.mode=TIA \
    generation.height=480 \
    generation.width=832 \
    diffusion.timesteps.sampling.steps=50 \
    generation.positive_prompt=./examples/test_case.json \
    generation.output.dir=./output

# # Run on 2 GPUs
# CUDA_VISIBLE_DEVICES=0,1 torchrun --node_rank=0 --nproc_per_node=2 --nnodes=1 \
#     --rdzv_endpoint=127.0.0.1:12345 \
#         --rdzv_conf=timeout=900,join_timeout=900,read_timeout=900 \
#         main.py humo/configs/inference/generate_1_7B.yaml \
#         dit.sp_size=2 \
#         generation.frames=97 \
#         generation.scale_t=7.0 \
#         generation.scale_i=4.0 \
#         generation.scale_a=7.5 \
#         generation.mode=TIA \
#         generation.height=480 \
#         generation.width=832 \
#         diffusion.timesteps.sampling.steps=50 \
#         generation.positive_prompt=./examples/test_case.json \
#         generation.output.dir=./output

# # Run on 4 GPUs
# CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --node_rank=0 --nproc_per_node=4 --nnodes=1 \
#     --rdzv_endpoint=127.0.0.1:12345 \
#         --rdzv_conf=timeout=900,join_timeout=900,read_timeout=900 \
#         main.py humo/configs/inference/generate_1_7B.yaml \
#         dit.sp_size=4 \
#         generation.frames=97 \
#         generation.scale_t=7.0 \
#         generation.scale_i=4.0 \
#         generation.scale_a=7.5 \
#         generation.mode=TIA \
#         generation.height=480 \
#         generation.width=832 \
#         diffusion.timesteps.sampling.steps=50 \
#         generation.positive_prompt=./examples/test_case.json \
#         generation.output.dir=./output
