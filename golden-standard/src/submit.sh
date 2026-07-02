#!/bin/bash
#SBATCH --job-name=gcdqn_interp
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --partition=<partition>

mkdir -p logs

cd /home/rg9360/golden-standard/src

python train.py env:box-moving \
    --agent.agent_name gcdqn_interp \
    --exp.name todo \
    --env.number_of_boxes_max 4 \
    --env.number_of_boxes_min 4 \
    --env.number_of_moving_boxes_max 1 \
    --env.grid_size 6 \
    --exp.gamma 0.99 \
    --env.episode_length 100 \
    --exp.seed ${SLURM_ARRAY_TASK_ID} \
    --exp.project camera-ready \
    --exp.epochs 50 \
    --exp.gif_every 10 \
    --agent.alpha 0.1 \
    --agent.batch_size 1024 \
    --agent.lr 3e-4 \
    --exp.max_replay_size 10000 \
    --exp.batch_size 1024 \
    --exp.lr 3e-4 \
    --exp.use_future_and_random_goals \
    --exp.eval_different_box_numbers \
    --agent.thinking_steps 5 \
    --wandb_mode offline



python train.py env:box-moving \
    --agent.agent_name gcdqn_interp \
    --exp.name todo \
    --env.number_of_boxes_max 4 \
    --env.number_of_boxes_min 4 \
    --env.number_of_moving_boxes_max 1 \
    --env.grid_size 6 \
    --exp.gamma 0.99 \
    --env.episode_length 100 \
    --exp.seed 0 \
    --exp.project camera-ready \
    --exp.epochs 50 \
    --exp.gif_every 10 \
    --agent.alpha 0.1 \
    --agent.batch_size 1024 \
    --agent.lr 3e-4 \
    --exp.max_replay_size 10000 \
    --exp.batch_size 1024 \
    --exp.lr 3e-4 \
    --exp.use_future_and_random_goals \
    --exp.eval_different_box_numbers \
    --agent.thinking_steps 5 \
    --wandb_mode offline
