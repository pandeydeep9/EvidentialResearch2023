

Runnning Experiments requires 
Python
pytorch
Torchvision
Additional details of the requirements can be found in the file "requirements.txt"

1) For The ToyMnist Experiments and results, Consider the folder ToyMnistExperiment
Run bash submit_demo.sh

2) For all remaining MNIST experiments and results, consider the folder MainMnistExperiment 
Run bash submit_baseline.sh to obtain Cross-Entropy based standard classifier model's results
Run bash submit_job.sh to obtain all Evidential model results (All hyperparameters and settings)

3) For all Cifar10 experiments and results, consider the folder MainCifar10Experiments
Run bash submit_baseline_ce.sh to obtain Cross-Entropy based standard classifier model's results
Run bash submit_evidential.sh to obtain all Evidential model results (All hyperparameters and settings)

4) For all Cifar100 experiments and results, consider the folder Cifar100Experiments
Run bash 1_submit_baseline_ce.sh to obtain Cross-Entropy based standard classifier model's results
Run bash 2_submit_evidential.sh to obtain all Evidential model results (All hyperparameters and settings)

5) For all Few-Shot Classification experiments with mini-ImageNet, consider the folder FewShotExperiments
Run bash 1_submit_baseline_ce.sh to obtain Cross-Entropy based standard classifier model's results
Run bash 2_submit_evidential.sh to obtain all Evidential model results (All hyperparameters and settings)

Datasets:
MNIST dataset, Cifar10 dataset, and Cifar100 dataset are automatically downloaded from the script. 
miniImagenet dataset needs to be downloaded and placed in the materials folder. Follow instructions in 0_mini_imagenet_instructions.txt for this dataset.

References:
- MNIST experiments are based on the github repository: https://github.com/dougbrion/pytorch-classification-uncertainty
- Cifar10 experiments are based on the github repostiory: https://github.com/kuangliu/pytorch-cifar.git
- Cifar100 experiments are based on the github repository: https://github.com/weiaicunzai/pytorch-cifar100.git
- miniImageNet experiments are based on the github repository: https://github.com/yinboc/few-shot-meta-baseline.git


