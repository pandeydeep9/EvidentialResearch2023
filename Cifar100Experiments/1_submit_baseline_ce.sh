#!/bin/bash
identifier_name="Baseline."

for sd in 1 2 3; do
	for model in 'resnet18' ; do
	    python3 train_baseline_ce.py -net $model -gpu --exp_id $identifier_name --seed $sd
	done
done
