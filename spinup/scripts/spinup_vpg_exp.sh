#!/bin/bash

set -x

declare -a envs=("CartPole-v1" "MountainCarContinuous-v0" "Pendulum-v0" "LunarLander-v2")

for env in "${envs[@]}"
do 
	python -m spinup.run vpg --env $env --data_dir ~/spinningup/data/vpg_exp/$env --hid\[hid\] \[128,128,128\] --seed 0 10 20 --epochs 200 --dt --act\[act\] torch.nn.ReLU torch.nn.Tanh --num_cpu 4
done
