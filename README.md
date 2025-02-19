# GPDP_IEEE_SSP_2025 
This repository is part of the paper "Overcoming Overfitting in Reinforcement Learning via Gaussian Process Diffusion Policy". <br />
Authors: Amornyos Horprasert, Esa Apriaskar, Xingyu Liu, Lanlan Su, Lyudmila S. Mihaylova. <br />
(Link: TBD) <br />

This repository provides <br />
1. An official PyTorch implementation of Gaussian Process Diffusion Policy (GPDP).  <br />
2. More detail on experiment and comparing algorithms. <br />

## 1. A PyTorch Implementation
This project was conducted in Python language. If you prefer other kind of programming language, for such as, C++, R and etc...
Feel free to use this repository as a guideline for your own framework. 
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

### Evaluation (Standard)
To run the evaluation in standard setting (no uncertainty, no distribution shifts), please refer to this command...
```
python main.py --alg GPDP --task testing --eval_mode standard --rendering [...]
```

### Evaluation (Shifted)
To run the evaluation in distribution shifts setting, please refer to this command...
```
python main.py --alg GPDP --task testing --eval_mode shifted --rendering [...]
```

### Training
In order to make the git push possible, we drop the dataset created from SAC and SAC'buffers from the repository. 
Therefore, All the trained agents are already provided. 
However, if the new training is required, please create the dataset first by referring to the following command:
```
python dataset_creation_walker2d.py
```
To run the training of GPDP, please refer to this command...
```
python main.py --alg GPDP --task training --gradient_step 2e+06
```

## 2. More detail on experiment and comparing algorithms
### Our Implementation of Soft Actor-Critic algorithm
We implement Soft Actor-Critic (SAC) based on the source code from [pytorch-soft-actor-critic](https://github.com/pranz24/pytorch-soft-actor-critic).
The code was modified to suit our code structure and training/testing conditions.  

### Our Implementation of Diffusion Q-Learning algorithm
We implement Diffusion Q-Learning (D-QL) algorithm, proposed in ..., by implementing our own source codes in PyTorch.
All function approximators in D-QL share the same architecture with our apporach (GPDP), which are declared in the paper.
Here, we would like to share the hyperparameters setting for D-QL as the following:
- Number of gradient step: 1e+06 (Early stop)
- $\alpha$ = 1.00 (Normalise constant for Q-values term in the policy objective function.) 

The results of D-QL quoted in the paper are from the best model selected by online selection method.
We performed online evaluation for every 10 epoch (~39k steps), then selects the checkpoint models that provide best non-discounted return.
We first set the same gradient step as our approach (~2M steps) but we spot degradation in the performance, so we early stop the training by ~1M steps. 

### Additional Details on The Standard Evaluation
We would like to present the video of the standard evaluation for Walker2d environment as the following.
The presented video is created by running each algorithms in multiple random seed generators then select the run that gives the best non-discounted cumulative reward to present here.
There are 4 panels, which represents SAC-S, SAC-D, D-QL and GPDP(ours) respectively. 

### Additional Details on The Distribution Shifted Evaluation
We would like to present the video of distribution shifted scenario illustrated in Fig.1. of the paper.
The presented video is created by the same procedure as in the standard evaluation. 
It can be seen in SAC-D algorithm (2nd panel) that the agent seems to success fully recover itself in the end of the video. 
We have extended the horizon and found that the agent cannot recover from falling state. However, it still shows the impressive exploration of the policy from a powerful SAC algorithm. 

<img src="https://github.com/AmornyosH/GPDP_IEEE_SSP_2025/blob/main/my_utilities/evaluation_shifted_best_traj.gif" alt="shifted" width="75%" height="75%">

### Simulated Hardware and Stochasticity
We are aware that apart from the robustness of the algorithm, the hardware plays a crucial role in the computational related to matrix, tensor and etc, which can lead to a different results on different machine. 
Here, we would like to declare the machine's specification used for this research. 

| Hardware | Specification |
| -------- | ------------- |
|   CPU   | Intel Core i5-12400F |
|   GPU   | Geforce RTX 3060 12GB |
| Operating System | Ubuntu 20.04 |
| GPU Acceleration Software | CUDA (ver.11.08) |

The seed generator configurations can be found in the ```main.py```.

