#!/bin/bash

python3 main_demo.py --use_vac_reg "True" --demo_ce_model "False" #Evidential model with regularization
python3 main_demo.py --use_vac_reg "False" --demo_ce_model "False" #Evidential model without regularization 
python3 main_demo.py --use_vac_reg "True" --demo_ce_model "True" #Standard cross Entropy model


