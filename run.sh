#!/usr/bin/env bash

logdir="data"
algo=$1
envs=("SuperMarioBros-1-2-v0" "SuperMarioBros-1-3-v0")

for ((i=0;i<1;i+=1)); do
    for env in ${envs[@]}; do
        python $1.py \
            --env $env \
            --seed $i \
            --policy_type 'cnn' \
            --epochs 1500 \
            --max_episode_len 250 \
            --steps_per_epoch 2000 \
            --pi_lr 1e-4 \
            --vf_lr 1e-4 \
            --cpu 8 \
            --gamma 0.992 \
            --lam 1.00 \
            --device 'cuda:0'
    done
done

# for env in ${envs[@]}; do
#     python utils/plot.py \
#         --logdir $logdir'/'$algo'_'$env'/' \
#         -y 'Performance' \
#         -s 1
# done
