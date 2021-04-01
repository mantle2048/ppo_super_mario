#!/usr/bin/env bash

logdir="data"
algo=$1
envs=("SuperMarioBros-1-2-v0" "SuperMarioBros-1-3-v0" "SuperMarioBros-2-3-v0" "SuperMarioBros-3-1-v0" "SuperMarioBros-3-3-v0" "SuperMarioBros-4-2-v0" "SuperMarioBros-4-3-v0" "SuperMarioBros-5-2-v0" "SuperMarioBros-5-3-v0")

for ((i=0;i<1;i+=1)); do
    for env in ${envs[@]}; do
        python $1.py \
            --env $env \
            --seed $i \
            --policy_type 'cnn' \
            --epochs 250 \
            --max_episode_len 1000 \
            --steps_per_epoch 4000 \
            --datestamp \
            --pi_lr 1e-4 \
            --cpu 4 \
            --gamma 0.99 \
            --device 'cuda:2'
    done
done

# for env in ${envs[@]}; do
#     python utils/plot.py \
#         --logdir $logdir'/'$algo'_'$env'/' \
#         -y 'Performance' \
#         -s 1
# done
