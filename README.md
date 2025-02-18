# GPDP_IEEE_SSP_2025 
This repository is part of the paper "Overcoming Overfitting in Reinforcement Learning via Gaussian Process Diffusion Policy". <br />
Authors: Amornyos Horprasert, Esa Apriaskar, Xingyu Liu, Lanlan Su, Lyudmila S. Mihaylova. <br />
(Link: TBD) <br />

This repository provides <br />
1. An official PyTorch implementation of Gaussian Process Diffusion Policy (GPDP).  <br />
2. More detail on experiment and comparing algorithms. <br />

## 1. A PyTorch Implementation
### Requirements
PyTorch, MuJoCo, Gymnasium are strictly required packages. Other supporting packages can be found in requirements.txt. <br />

### Arguments
```
--alg [...]            'Algorithm: Replace [...] with "SAC-S", "SAC-D", "D-QL" or "GPDP", default: "GPDP"'
--task [...]           'Task: Replace [...] with "training", "testing", default: "training"'
--gradient_step N      'Number of gradient step: Replace N with integer, default: 2e+06'
--eval_mode [...]      'Evaluation mode: Replace [...] with "standard", "shifted", default: "standard"'
--rendering [...]      'Rendering: Replace [...] with "render", "no-render", default: "no-render"'
```
### Training
To run the training, please refer to this command...
```
python main.py --alg GPDP --task training --gradient_step 2e+06
```
### Evaluation (Standard)
To run the evaluation in standard setting (no uncertainty, no distribution shifts), please refer to this command...
```
python main.py --alg GPDP --task testing --eval_mode standard --rendering [...]
```
![gpdp_standard](https://github.com/AmornyosH/GPDP_IEEE_SSP_2025/blob/main/GPDP_standard.gif)
### Evaluation (Shifted)
To run the evaluation in distribution shifts setting, please refer to this command...
```
python main.py --alg GPDP --task testing --eval_mode shifted --rendering [...]
```
![gpdp_shifted](https://github.com/AmornyosH/GPDP_IEEE_SSP_2025/blob/main/GPDP_shifted.gif)

## 2. More detail on experiment and comparing algorithms
### Our Implementation of Soft Actor-Critic algorithm
We implement Soft Actor-Critic (SAC) based on the source code from [pytorch-soft-actor-critic](https://github.com/pranz24/pytorch-soft-actor-critic).
The code was slightly modified to suit our code structure and training/testing conditions.  

### Our Implementation of Diffusion Q-Learning algorithm
We implement Diffusion Q-Learning (D-QL) algorithm, proposed in ..., by implementing our own source codes in PyTorch.
All function approximators in D-QL share the same architecture with our apporach (GPDP), which are declared in the paper.
Here, we would like to share the hyperparameters setting for D-QL as the following:
- Number of gradient step: 1e+06 (Early stop)
- eta = 1.00 

The results of D-QL quoted in the paper are from the best model selected by online selection method.
We performed online evaluation for every 10 epoch (~39k steps), then selects the checkpoint models that provide best non-discounted return.
We first set the same gradient step as our approach (~2M steps) but we spot degradation in the performance, so we early stop the training by ~1M steps. 





