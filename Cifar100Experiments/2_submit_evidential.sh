#!/bin/bash
identifier_name="Evidential."

for sd in 1 2 3; do
	for unc_type in "log" "mse" "digamma"; do
	    for act in 'relu' 'softplus' 'exp'; do
		for kl in 0 1 10 100 200 500 1000 2000 5000 10000; do
		    for uncertainty in "true"; do
		        for evcor in "false"; do
		            for model in 'resnet18'; do
		                kl_str=$(bc <<<"scale=3;$kl/1000")
		                python3 train_evidential.py -net $model -gpu --exp_id $identifier_name \
					 --unc_type $unc_type --unc_act $act --kl_strength $kl_str \
					 --uncertainty $uncertainty --use_vac_reg $evcor --seed $sd 
		            done

		        done
		    done
		done
	    done
	done
done

for sd in 1 2 3; do
	for unc_type in "log" "mse" "digamma"; do
	    for act in 'exp'; do
		for kl in 0 1 10 100 200 500 1000 2000 5000 10000; do
		    for uncertainty in "true"; do
		        for evcor in "true" ; do
		            for model in 'resnet18'; do
		                kl_str=$(bc <<<"scale=3;$kl/1000")
		                python3 train_evidential.py -net $model -gpu --exp_id $identifier_name \
					 --unc_type $unc_type --unc_act $act --kl_strength $kl_str \
					 --uncertainty $uncertainty --use_vac_reg $evcor --seed $sd 
		            done

		        done
		    done
		done
	    done
	done
done


