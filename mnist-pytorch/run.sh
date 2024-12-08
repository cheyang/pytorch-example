#!/bin/bash

# 从环境变量中获取所需的值
world_size=${WORLD_SIZE}
master_addr=${MASTER_ADDR}
master_port=${MASTER_PORT}
nproc_per_node=${NPROC_PER_NODE}
rank=${RANK:-0}  # 如果RANK没有设置，缺省值为0

# 检查环境变量是否已经设置
if [ -z "$world_size" ]; then
    echo "Error: WORLD_SIZE environment variable is not set."
    exit 1
fi

if [ -z "$master_addr" ]; then
    echo "Error: MASTER_ADDR environment variable is not set."
    exit 1
fi

if [ -z "$master_port" ]; then
    echo "Error: MASTER_PORT environment variable is not set."
    exit 1
fi

if [ -z "$nproc_per_node" ]; then
    echo "Error: NPROC_PER_NODE environment variable is not set."
    exit 1
fi

# 计算 NNODES
nnodes=$((world_size / nproc_per_node))
if [ $((world_size % nproc_per_node)) -ne 0 ]; then
    echo "Warning: WORLD_SIZE is not an exact multiple of NPROC_PER_NODE, rounding down."
fi

# 构建完整的 torchrun 命令
torchrun_cmd="torchrun --master_port ${master_port} \
                       --nproc_per_node=${nproc_per_node} \
                       --nnodes=${nnodes} \
                       --node_rank=${rank} \
                       --master_addr=${master_addr} \
                       /root/mnist-pytorch/mnist.py $@"

# 打印 torchrun 命令
echo "Running command: $torchrun_cmd"

# 执行 torchrun 命令
eval $torchrun_cmd

