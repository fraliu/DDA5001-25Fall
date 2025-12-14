#!/bin/bash

# 定义日志目录
mkdir -p logs
mkdir -p results

# 定义要运行的任务列表
# 格式：GPU_ID  MODEL_PATH  OUTPUT_FILE
# 请在下方修改你需要跑的模型路径

# 任务 1：在 GPU 5 上跑 Model A
nohup python rollout.py \
    --model "experiments/adam_lr2e-5_epoch2" \
    --output_file "model_adam_math500.jsonl" \
    --gpu_id 5 \
    --is_local_model \
    > gpu_5_model_A.log 2>&1 &

echo "Task 1 started on GPU 5 (PID: $!)"

# 任务 2：在 GPU 6 上跑 Model B
nohup python rollout.py \
    --model " experiments/lora_lr5e-5_epoch3_rank16" \
    --output_file "model_lora_math500.jsonl" \
    --gpu_id 6 \
    --is_local_model \
    > gpu_6_model_B.log 2>&1 &

echo "Task 2 started on GPU 6 (PID: $!)"

# 任务 3：在 GPU 7 上跑 Model C
nohup python rollout.py \
    --model "experiments/sgd_lr5e-5_epoch3" \
    --output_file "model_sgd_math500.jsonl" \
    --gpu_id 7 \
    --is_local_model \
    > gpu_7_model_C.log 2>&1 &

echo "Task 3 started on GPU 7 (PID: $!)"

echo "Task 4 started on GPU 0 (PID: $!)"

echo "---------------------------------------"
echo "All tasks submitted to background."
echo "Check progress via: tail -f logs/*.jsonl"