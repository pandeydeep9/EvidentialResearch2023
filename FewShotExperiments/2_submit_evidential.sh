#!/bin/bash
identifier_name="EVID."
for sd in 1 2 3; do
	for kl in 0 1 10 ; do
	for unc_type in "log" ; do
	    for act in 'exp'; do
		    for uncertainty in "true"; do
		        for evcor in "true" "false"; do
		            for model in 'resnet18'; do
		                kl_str=$(bc <<<"scale=3;$kl/10")

		                python3 train_classifier_evidential.py --config configs/train_classifier_mini_evid.yaml \
					 --exp_id $identifier_name \
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
	for kl in 0 1 10 ; do
		for unc_type in "log" ; do
		    	for act in 'softplus' 'relu'; do
			    for uncertainty in "true"; do
				for evcor in "false"; do
				    for model in 'resnet18'; do
				        kl_str=$(bc <<<"scale=3;$kl/10")
				        python3 train_classifier_evidential.py --config configs/train_classifier_mini_evid.yaml \
							 --exp_id $identifier_name \
							 --unc_type $unc_type --unc_act $act --kl_strength $kl_str \
							 --uncertainty $uncertainty --use_vac_reg $evcor --seed $sd
				    done

				done
			    done
			done
		done
	done
done
