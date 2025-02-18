import numpy as np
from time import time

# ========================================= My Modules ===========================================
from SAC.sac import SAC
from SAC.replay_memory import ReplayMemory

# ======================================== Other Modules =========================================
import os
import gymnasium as gym
import random
import torch
CUDA = torch.cuda.is_available()

# ======================================= Local Functions ========================================
def getParamsDict(env, training, testing):
    return {'environment': env,
            'epoch': 256,
            'num_batches': 10,
            'expert_num_batches': 10,
            'batch_size': 256,
            'horizon': 1000,
            'action_config': 'continous',
            'state_dim': 17,
            'action_dim': 6,
            'training_flag': training,
            'test_flag': testing}

def interaction(agent, seed, train:bool=False):
    
    # `````````` Initialise/Reset observation parameters
    step = 0  # Initialise step counter
    state = env.reset(seed=seed)[0]
    truncated = False
    terminated = False
    noise_finish = False
    trajectory_reward = 0
    noise_start_time = np.random.randint(100, 800)

    while not terminated and not truncated if not UNCERTAINTY else not truncated:

        _num_samples = len(reward_buffers)
        if _num_samples > 0.50*NUM_SAMPLES:
            # Partial Expert Policy (Half Expert and Random the other half.) 
            if step > noise_start_time:
                action = env.action_space.sample()
            else:
                # We employ deterministic policy here, 
                # since we will inject stochasticity into the agent anyway in this condition.
                action = agent.select_action(state, evaluate=True).detach().cpu().tolist() 
            cond = 1
        else:
            # Expert Policy (Gaussian Policy)
            # We use stochastic policy here to introduce stochasticity in the dataset.
            action = agent.select_action(state, evaluate=False).detach().cpu().tolist()  
            cond = 2
        
        # if UNCERTAINTY:
        #     if step > noise_start_time and not noise_finish:
        #         while state[0] > 0.5:
        #             action[3:6] = [0., 0., 0.]
        #             next_state, _, _, _, _ = env.step(action) # Step
        #             state = next_state
        #         noise_finish = True
        #     # if step > noise_start_time and step < (noise_start_time+100):
        #     #     action[3:6] = [0., 0., 0.]

        next_state, reward, terminated, truncated, _ = env.step(action)
        trajectory_reward += reward

        # Add to buffers
        state_buffers.append(state)
        action_buffers.append(action)
        reward_buffers.append(reward)
        next_state_buffers.append(next_state)
        truncrated_buffers.append(truncated)
        termination_buffers.append(terminated)

        # Increment
        step += 1
        state = next_state
    
    return step, trajectory_reward, cond

def setGlobalSeed(seed:int):
    random.seed(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed)

def evaluating(eval_times:int=3, s_print:bool=False):
    _TEST_TIME = eval_times
    # _TEST_SEEDS = [2203, 210388, 21222122]
    # _TEST_SEEDS = np.random.randint(22, 10000000, size=_TEST_TIME)
    _TEST_SEEDS = [2203+(n*(8**n)) for n in range(eval_times)]
    # _EXPERT_REWARD = 4592.3  # Raw reward of SAC (from D4RL paper)
    norm_reward_accum = 0

    # print('********************************* Evaluation Start *********************************')
    for e in range(_TEST_TIME):
        setGlobalSeed(_TEST_SEEDS[e])
        # Testing only (No fine-tuning)
        # ========== Start the program
        # Check for the rendering
        # Run evaluated policy
        _, reward_accum = interaction(agent, seed, train=False)
        # trajectory_reward = np.sum(reward_accum)
        # # Run random policy
        # state_exps, action_exps, reward_exps, next_state_exps = Interaction(agent, seed, random=True)
        # random_trajectory_reward = np.sum(reward_exps)
        # Normalised reward
        # norm_trajectory_reward = (trajectory_reward-random_trajectory_reward) / (_EXPERT_REWARD-random_trajectory_reward)
        # norm_trajectory_reward *= 100
        # norm_reward_accum += norm_trajectory_reward
        norm_reward_accum += reward_accum

        if s_print:
            print('Training_Epoch: ', agent.gradient_steps,
                    ', Test: ', e+1, '/', _TEST_TIME, 
                    ', Evaluation_Reward: ', round(reward_accum, 2), 
                    ', Average_Reward: ', round(norm_reward_accum/(e+1), 2),
                    ', seed: ', _TEST_SEEDS[e])
        
    # print('********************************** Evaluation End **********************************')
    return norm_reward_accum/_TEST_TIME

# ====================================== Main Program Start ======================================
seed = 2203
setGlobalSeed(seed)

# ========== Global constants
ENV_NAME = 'Walker2d-v5'
LOADING = True
TRAINING = False
FINE_TUNING = False
TESTING = True
GP_TRAINING = False
EXPERT_PATH = 'resources/walker2d_d4rl_dataset_medium_expert_1m.npz'
RENDER = False
UNCERTAINTY = False
HALF_SAVE = False
NUM_SAMPLES = 1000000

# Start environment
if RENDER: 
    env = gym.make(ENV_NAME, render_mode='human', width=1280, height=720)
else:
    env = gym.make(ENV_NAME)  

# ========== Get parameters dictionary of the environment.
params_dict = getParamsDict(env=ENV_NAME, training=TRAINING, testing=TESTING)

# ========== Create the agent
agent = SAC(load=LOADING, params_dict=params_dict, cuda=CUDA)
memory = ReplayMemory(load=LOADING, capacity=int(1e+06), seed=seed)
if LOADING:
    agent.load_checkpoint("SAC/checkpoints/sac_checkpoint_{}_".format(ENV_NAME), evaluate=True)
    memory.load_buffer("SAC/checkpoints/sac_buffer_{}_".format(ENV_NAME))

DATASET_PATH = 'datasets/dataset_{}_{}_{}'.format(ENV_NAME, agent.alg, NUM_SAMPLES)

if not os.path.isfile(DATASET_PATH):
    print('********************************* Create the dataset *********************************')
    state_buffers = []
    action_buffers = []
    reward_buffers = []
    next_state_buffers = []
    truncrated_buffers = []
    termination_buffers = []

    while len(reward_buffers) < NUM_SAMPLES:
        _, trajectory_reward, cond = interaction(agent, seed=seed, train=False)
        print('num_samples: ', len(reward_buffers), ', condition: ', cond, ', rewards: ', trajectory_reward, ', termination: ', termination_buffers[-1])

    state_buffers = np.vstack(state_buffers)
    action_buffers = np.vstack(action_buffers)
    reward_buffers = np.vstack(reward_buffers)
    next_state_buffers = np.vstack(next_state_buffers)
    truncrated_buffers = np.vstack(truncrated_buffers)
    termination_buffers = np.vstack(termination_buffers)

    np.savez(DATASET_PATH, 
             len(state_buffers),    # arr_0
             state_buffers,         # arr_1
             action_buffers,        # arr_2
             reward_buffers,        # arr_3
             next_state_buffers,    # arr_4
             truncrated_buffers,    # arr_5
             termination_buffers)   # arr_6
else:
    print('Dataset exists, do nothing')

