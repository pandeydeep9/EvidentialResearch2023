#!/bin/bash
identifier_name="BASELINE."
for sd in 1 2 3; do
	for model in 'resnet50'; do
	    python3 train_classifier.py --config configs/train_classifier_mini.yaml --seed $sd --identifier $identifier_name
	done
done
