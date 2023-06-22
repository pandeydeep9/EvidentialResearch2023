<H1> Regularized Evidential Model Code (RED) </H1>

Paper Title: Learn to Accumulate Evidence from All Training Samples: Theory and Practice </br>
Paper link: https://openreview.net/forum?id=2MaUpKBSju </br>


<p> This repository contains source code for the <b>Regularized Evidential Model Code (RED) </b> model. 



<H3> Requirements </H3>
<p>Runnning Experiments in this repository requires following : </p>
<ul>
<li>Python
<li>pytorch
<li>Torchvision
</ul>
<p> Note: Additional details of the requirements can be found in the file "requirements.txt" and corresponding code folders</p>

<H3> Running Experiments</H3>
<p>There are 5 sets of experiments</p>
<ol>
<li> <b>Toy MNIST Experiments</b>
<p> For The ToyMnist Experiments and results, Consider the folder ToyMnistExperiment. To run all experiments, run ``` bash submit_demo.sh ``` </p>
</li>
<li> <b>All Remaining MNIST Experiments</b>
<p> For all remaining MNIST experiments and results, consider the folder MainMnistExperiment </p>
Run ``` bash submit_baseline.sh ``` to obtain Cross-Entropy based standard classifier model's results<br>
Run ``` bash submit_job.sh ``` to obtain all Evidential model results (All hyperparameters and settings) <br>

<li> <b>All Cifar10 Experiments</b>
<p> For all Cifar10 experiments and results, consider the folder MainCifar10Experiments </p>
Run ``` bash submit_baseline_ce.sh ``` to obtain Cross-Entropy based standard classifier model's results <br>
Run ``` bash submit_evidential.sh ``` to obtain all Evidential model results (All hyperparameters and settings) <br>

<li> <b>All Cifar100 Experiments</b>
<p> For all Cifar100 experiments and results, consider the folder Cifar100Experiments </p> 
Run ``` bash 1_submit_baseline_ce.sh ``` to obtain Cross-Entropy based standard classifier model's results<br>
Run ``` bash 2_submit_evidential.sh ``` to obtain all Evidential model results (All hyperparameters and settings)<br>

<li> <b>All Few Shot Classification Experiments</b>
<p> For all Few-Shot Classification experiments with mini-ImageNet, consider the folder FewShotExperiments</p>
Run ``` bash 1_submit_baseline_ce.sh ``` to obtain Cross-Entropy based standard classifier model's results<br>
Run ``` bash 2_submit_evidential.sh ``` to obtain all Evidential model results (All hyperparameters and settings)<br>

</ol>
<H3> Datasets:</H3>

MNIST dataset, Cifar10 dataset, and Cifar100 dataset are automatically downloaded from the script. <br>
mini-Imagenet dataset needs to be downloaded and placed in the materials folder. Follow instructions in 0_mini_imagenet_instructions.txt for mini-ImageNet.

<H3>References:</H3>
<ul>
<li>MNIST experiments are based on the github repository: https://github.com/dougbrion/pytorch-classification-uncertainty
<li>Cifar10 experiments are based on the github repostiory: https://github.com/kuangliu/pytorch-cifar.git
<li>Cifar100 experiments are based on the github repository: https://github.com/weiaicunzai/pytorch-cifar100.git
<li>mini-ImageNet experiments are based on the github repository: https://github.com/yinboc/few-shot-meta-baseline.git


