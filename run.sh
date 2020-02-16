#!/usr/bin/env bash
declare -i num_runs=$(python config/$1.py)

for ((i=0; i<num_runs; i++))
do
   python main.py --exp $1 --run $i
done