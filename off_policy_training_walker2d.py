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
    state_exps = []
    action_exps = []
    reward_exps = []
    next_state_exps = []
    noise_start_time = 300

    while not terminated and not truncated if not UNCERTAINTY else not truncated:
        
        if agent.start_steps > agent.gradient_steps and train:
            action = env.action_space.sample()
        else:
            action = agent.select_action(state, evaluate=False if train else True).detach().cpu().tolist()  # Change to cpu memories and make it a list
        
        if UNCERTAINTY:
            # if step > noise_start_time and not noise_finish:
            #     while state[0] > 0.5:
            #         action[3:6] = [0., 0., 0.]
            #         # action[2] = 0.
            #         next_state, _, _, _, _ = env.step(action) # Step
            #         state = next_state
            #     noise_finish = True

            if step > noise_start_time and step < (noise_start_time+100):
                action[3:6] = [0., 0., 0.]
                # action[:] = [0., 0., 0., 0., 0., 0.]


        next_state, reward, terminated, truncated, _ = env.step(action)
        trajectory_reward += reward

        # if train:
        #     mask = 1 if step == env._max_episode_steps else float(not terminated)
        #     memory.push(state, action, reward, next_state, mask) # Append transition to memory
            
        #     if len(memory) > params_dict['batch_size']:
        #         # Number of updates per step in environment
        #         for _ in range(agent.target_update_interval):
        #             # Update parameters of all the networks
        #             _, _, _, _, _ = agent.update_parameters(memory, params_dict['batch_size'])

        # Increment
        step += 1
        state = next_state
    
    return step, trajectory_reward

def setGlobalSeed(seed:int):
    random.seed(seed)
    np.random.seed(seed=seed)
    torch.manual_seed(seed)

def evaluating(eval_times:int=3, s_print:bool=False):
    _TEST_TIME = eval_times
    # _TEST_SEEDS = [2203, 210388, 21222122]
    # _TEST_SEEDS = np.random.randint(22, 10000000, size=_TEST_TIME)
    _TEST_SEEDS = [2203+(n*(9**n)) for n in range(eval_times)]
    # _TEST_SEEDS = [2202+((10**n)) for n in range(eval_times)]
    # _EXPERT_REWARD = 4592.3  # Raw reward of SAC (from D4RL paper)
    norm_reward_accum = 0

    # print('********************************* Evaluation Start *********************************')
    for e in range(_TEST_TIME):
        setGlobalSeed(_TEST_SEEDS[e])
        # Testing only (No fine-tuning)
        # ========== Start the program
        # Check for the rendering
        # Run evaluated policy
        _, reward_accum = interaction(agent, seed, train=True)
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
UNCERTAINTY = True
HALF_SAVE = False

# Start environment
if RENDER: 
    env = gym.make(ENV_NAME, render_mode='human', width=1280, height=720)
else:
    env = gym.make(ENV_NAME)  
    # env = gym.make('Walker2d-v3') 

# ========== Get parameters dictionary of the environment.
params_dict = getParamsDict(env=ENV_NAME, training=TRAINING, testing=TESTING)

# ========== Create the agent
agent = SAC(load=LOADING, params_dict=params_dict, cuda=CUDA)
memory = ReplayMemory(load=LOADING, capacity=int(1e+06), seed=seed)
if LOADING:
    agent.load_checkpoint("SAC/checkpoints/sac_checkpoint_{}_".format(ENV_NAME))
    memory.load_buffer("SAC/checkpoints/sac_buffer_{}_".format(ENV_NAME))

if TRAINING:
    TOTAL_GRADIENT_STEPS = int(4e+06)  # 1M steps of training.
    episode = 0
    evaluation_time = 10
    print('Start Off-Policy Training...')
    # Train DP and Value functions.
    while agent.gradient_steps < TOTAL_GRADIENT_STEPS:

        if agent.gradient_steps > TOTAL_GRADIENT_STEPS/2:
            UNCERTAINTY = True

        # Evaluated Policy for every 10 epoch.
        if episode % evaluation_time == 0:
            print('********************************* Evaluation Start *********************************')
            norm_reward = evaluating(eval_times=10)
            agent.training_reward_append.append(norm_reward)

            if norm_reward > agent.best_reward_training:
                # Save checkpoint best model
                agent.save_checkpoint(env_name=ENV_NAME, suffix='checkpoint')
                print('Save checkpoint.')
                agent.best_reward_training = norm_reward

            print('Training_Epoch: ', agent.gradient_steps,
                    ', Evaluation_Reward: ', round(norm_reward, 2), 
                    ', Average_Reward: ', round(np.mean(agent.training_reward_append), 2))
            
            print('********************************** Evaluation End **********************************')

        step, reward_accum = interaction(agent=agent, seed=seed, train=True)
        # Save the model
        agent.save_checkpoint(env_name=ENV_NAME, suffix='uncer' if UNCERTAINTY else '')
        # Save the model at 1M
        # if agent.gradient_steps >= TOTAL_GRADIENT_STEPS//2 and not HALF_SAVE:
        #     agent.save_checkpoint(env_name=ENV_NAME, suffix='half')
        episode += 1

        print('Episode: ', episode, ', Steps: ', step,
              ', Trajectory_Reward: ', reward_accum, ', Gradient_step: ', agent.gradient_steps, ', Uncertainty: ', UNCERTAINTY)

    # Save buffers
    memory.save_buffer(env_name=ENV_NAME)

# ========== Evaluation only
if TESTING:
    # Start environment
    if RENDER: 
        env = gym.make(ENV_NAME, render_mode='human')
        # env = gym.make('Walker2d-v3', render_mode='human')
    else:
        env = gym.make(ENV_NAME) 
    evaluating(eval_times=10, s_print=True)
    env.close()