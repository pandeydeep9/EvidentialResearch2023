#!/bin/bash
identifier_name="Evidential."
let count=1

for sd in 1 2 3; do
	for kl in 0 1 10 100 1000 10000 50000; do

		for unc_type in "log" "mse" "digamma"; do
			for act in 'exp'; do
			    for uncertainty in "true"; do
				for evcor in "true" "false"; do

				    for model in 'resnet18'; do
					count=$((seed + 1))
					kl_str=$(bc <<<"scale=3;$kl/1000")
					echo $count

					python3 main_evidential.py -net $model -gpu --exp_id $identifier_name \
							--unc_type $unc_type --unc_act $act --kl_strength $kl_str \
							--uncertainty $uncertainty --use_vac_reg $evcor --seed $sd
				    done
				done
			    done
			done
		done
	done
done
