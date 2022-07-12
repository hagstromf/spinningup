#!/bin/bash

set -x

declare -a envs="CartPole-v1 MountainCarContinuous-v0 Pendulum-v0 LunarLander-v2"
declare -a algos=("my_vpg" "vpg")

for algo in "${algos[@]}"
do 
	python -m spinup.run $algo --env $envs --data_dir ~/spinningup/data/vpg_exp/$envs --hid\[hid\] \[128,128,128\] --seed 0 10 20 --epochs 150 --dt
done
