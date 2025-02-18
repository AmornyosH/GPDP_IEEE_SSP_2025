import numpy as np
from time import time, sleep

# ========================================= My Modules ===========================================
from GPDP.GPDP import GaussianProcessDiffusionPolicy
# from GPDP.GPDP_2 import GaussianProcessDiffusionPolicy
from Diffusion_QL.Diffusion_QL import DiffusionQL
from SAC.sac import SAC

# ======================================== Other Modules =========================================
import os
import argparse
import gymnasium as gym
import random
import torch
CUDA = torch.cuda.is_available()

# ======================================= Local Functions ========================================
def addArguments(parser):
    parser.add_argument('--alg', default="GPDP", help='Algorithm: ["SAC-S", "SAC-D", "D-QL", "GPDP"], default:GPDP')
    parser.add_argument('--task', default='training', help='Task: ["training", "testing"], default:training')
    parser.add_argument('--eval_mode', default='standard', help='Evaluation mode: [standard, shifted], default:standard')
    parser.add_argument('--rendering', default='no-render', help='Rendering: [render, no-render], default:no-render')

def setGlobalSeed(seed:int):
    random.seed(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed)

def getParamsDict(env):
    return {'environment': env,
            'epoch': 256,
            'num_batches': 10,
            'expert_num_batches': 10,
            'horizon': 1000,
            'action_config': 'continous',
            'state_dim': 17,
            'action_dim': 6}

def interaction(agent, seed, random:bool=False) -> float:
    
    # `````````` Initialise/Reset observation parameters
    step = 0  # Initialise step counter
    state = env.reset(seed=seed)[0]
    truncated = False
    terminated = False
    trajectory_reward = 0
    state_exps = []
    action_exps = []
    reward_exps = []
    next_state_exps = []
    noise_start_time = 300

    while not terminated and not truncated if not UNCERTAINTY else not truncated:
        if not random:
            action = agent.predict(state=state, size=1)
            action = action.detach().cpu().tolist()  # Change to cpu memories and make it a list
        else:
            action = np.random.randn(6)
        
        if UNCERTAINTY:
            if step > noise_start_time and step < (noise_start_time+100):
                action[3:6] = [0., 0., 0.]

        next_state, reward, terminated, truncated, _ = env.step(action)
        trajectory_reward += reward
        
        # Store experiences
        state_exps.append(state)
        action_exps.append(action)
        reward_exps.append(reward)
        next_state_exps.append(next_state)
        
        # Increment
        step += 1
        state = next_state
        # env.render()

    np.stack(state_exps, axis=0)
    np.stack(action_exps, axis=0)
    np.stack(reward_exps, axis=0)  
    reward_exps = np.reshape(reward_exps, [-1, 1])
    np.stack(next_state_exps, axis=0)

    return state_exps, action_exps, reward_exps, next_state_exps

def evaluating(eval_times:int=10):
    _TEST_TIME = eval_times
    _TEST_SEEDS = [2203+(n*(7**n)) for n in range(eval_times)]
    _EXPERT_REWARD = 4580.59  # Raw reward of SAC with Stochastic Policy
    norm_reward_accum = 0
    norm_reward_append = []
    print('********************************* Evaluation Start *********************************')
    for e in range(_TEST_TIME):
        setGlobalSeed(_TEST_SEEDS[e])
        # Testing only (No fine-tuning)
        # ========== Start the program
        # Check for the rendering
        # Run evaluated policy
        _, _, reward_exps, _ = interaction(agent, seed, random=False)
        trajectory_reward = np.sum(reward_exps)
        norm_reward_append.append(reward_exps)
        norm_reward_accum += trajectory_reward
        
        print('Test: ', e+1, '/', _TEST_TIME, 
              ', Accum_Reward: ', round(trajectory_reward, 2), 
              ', Norm_Accum_Reward: ', round(100*trajectory_reward/_EXPERT_REWARD, 2),
              ', Average_Reward: ', round(norm_reward_accum/(e+1), 2),
              ', Average Norm_Reward: ', round(100*(norm_reward_accum/(e+1))/_EXPERT_REWARD, 2),
              ', seed: ', _TEST_SEEDS[e])
        
        env.reset()
    print('********************************** Evaluation End **********************************')
    norm_reward_append = np.reshape(np.vstack(norm_reward_append), (-1, 1000))
    # norm_reward_append = np.vstack(norm_reward_append)
    return norm_reward_accum/_TEST_TIME, norm_reward_append

# ====================================== Main Program Start ======================================
seed = 2203
setGlobalSeed(seed)

parser = argparse.ArgumentParser(description='GPDP (IEEE SSP 2025) PyTorch Args')
addArguments(parser)
args = parser.parse_args()

# ========== Global constants
ENV_NAME = 'Walker2d-v5'
TASK = args.task
EXPERT_PATH = 'datasets/dataset_{}_SAC_1000000.npz'.format(ENV_NAME)
RENDER = True if args.rendering == 'render' else False
UNCERTAINTY = True if args.eval_mode == 'shifted' else False

# Get parameters dictionary of the environment.
params_dict = getParamsDict(env=ENV_NAME)
dataset = np.load(EXPERT_PATH)  # ========== Load the dataset

# ========== Create the agent
if args.alg == 'GPDP':
    agent = GaussianProcessDiffusionPolicy(params_dict=params_dict, dataset=dataset, ft=False)
elif args.alg == 'D-QL':
    agent = DiffusionQL(params_dict=params_dict, dataset=dataset, ft=False)
elif args.alg == 'SAC-D':
    agent = SAC(load=True, params_dict=params_dict, eval_policy=True, cuda=CUDA)
elif args.alg == 'SAC-S':
    agent = SAC(load=True, params_dict=params_dict, eval_policy=False, cuda=CUDA)

# ========== Training (Offline)
EPOCH = 512
if args.task == 'training':
    # Train Diffusion Policy and Value functions.
    while agent.training_record < EPOCH:
        print('Start Offline Training...')

        # Evaluated Policy for every 10 epoch.
        if agent.training_record % 10 == 0:
            # Start environment
            if RENDER: 
                env = gym.make(ENV_NAME, render_mode='human', width=1280, height=720)
            else:
                env = gym.make(ENV_NAME)  
            norm_reward, norm_reward_append = evaluating(eval_times=10)
            agent.norm_reward_training_append.append(norm_reward)
            if norm_reward > agent.best_norm_reward_training:
                # Save checkpoint best model
                agent.recordSaving(path=agent.training_checkpoint_path)  # Save DP and V, Q.
                agent.gp_model.recordSaving(path=agent.gp_model.training_checkpoint_path)  # Save GP.
                agent.best_norm_reward_training = norm_reward
            env.close()

        # Train Diffusion Policy
        agent.training(total_epoch=EPOCH, eval=True)

        # Check if the agent has gp ?
        if hasattr(agent, 'gp_model'):
            # Train GP (after)
            # agent.gp_model.y_train = agent.getAlteredObservation(agent.gp_model.x_train)
            agent.gp_model.myTraining(total_epoch=2000, ft=False)
        print('Training is done!')


# ========== Testing
if args.task == 'testing':
    # Start environment
    if RENDER is True: 
        env = gym.make(ENV_NAME, render_mode='human', width=1280, height=720)
    else:
        env = gym.make(ENV_NAME)  
    _, norm_reward_append = evaluating(eval_times=10)
    np.save(agent.evaluation_path, norm_reward_append) # save the evaluation rewards.
    env.close()
