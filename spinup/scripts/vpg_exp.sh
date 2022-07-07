#!/bin/bash

set -x

declare -a envs="CartPole-v1 MountainCarContinuous-v0 BipedalWalker-v3 LunarLander-v2"
declare -a algos=("my_vpg" "vpg")

for algo in "${algos[@]}"
do 
	python -m spinup.run "$algo" --env $envs --data_dir ~/spinningup/data/vpg_exp --exp_name "$algo" --seed 0 10 20 --epochs 150 --dt
done
