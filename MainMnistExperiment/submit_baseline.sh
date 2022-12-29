#!/bin/bash
identifier_name="MNIST.CE.BASELINE."
let sd=1
for seed in 0 1 2; do
    for unc_type in "log"; do
        for act in 'none'; do
            for kl_str in 0; do
                for dropout in "true"; do
                    for uncertainty in "false"; do
                        echo $sd
                        ((sd=sd+1))

                        python3 main.py --exp_id $identifier_name \
                            --unc_type $unc_type --unc_act $act --kl_strength $kl_str --dropout $dropout \
                            --uncertainty $uncertainty --seed $seed --use_vac_reg "False"
                    done
                done
            done
        done
    done
done
