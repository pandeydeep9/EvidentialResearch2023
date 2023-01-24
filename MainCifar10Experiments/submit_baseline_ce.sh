#!/bin/bash
identifier_name="Standard."
for sd in 1 2 3; do
	for model in 'resnet50'; do
	    python3 main.py -net $model -gpu --exp_id $identifier_name --seed $sd
	done
done
