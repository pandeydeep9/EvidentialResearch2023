MNIST experiments are based on the github repository :https://github.com/dougbrion/pytorch-classification-uncertainty

MNIST experiments require 
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
