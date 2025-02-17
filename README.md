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
python main.py --alg [...] --task training --gradient_step N
```
### Evaluation (Standard)
To run the evaluation in standard setting (no uncertainty, no distribution shifts), please refer to this command...
```
python main.py --alg [...] --task testing --eval_mode standard --rendering [...]
```
![gpdp_standard](https://github.com/AmornyosH/GPDP_IEEE_SSP_2025/blob/main/GPDP_standard.gif)
### Evaluation (Shifted)
To run the evaluation in distribution shifts setting, please refer to this command...
```
python main.py --alg [...] --task testing --eval_mode shifted --rendering [...]
```

## 2. More detail on experiment and comparing algorithms
### Reproduction of Soft Actor-Critic algorithm
We reproduce Soft Actor-Critic (SAC) by using the source code from [pytorch-soft-actor-critic](https://github.com/pranz24/pytorch-soft-actor-critic).
The code was slightly modified to suit our code structure and training/testing conditions.  




