'''
Guided DDPM module
Revision: 1
Remark: PyTorch version
'''
# ============================ Pytorch Related ============================
import torch
import gpytorch
CUDA = torch.cuda.is_available()
if CUDA:
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
# gpytorch.settings.lazily_evaluate_kernels._set_state(False)
# ============================ Pytorch Related ============================

# ================================== Others ==================================
from my_utilities import my_utils, my_NN
from time import time, sleep
import numpy as np
import os

# Declare Global Constants
GAMMA = 0.99  # Discount factor.
TAU = 0.005  # Tau for soft updating of target model's weights.

class DiffusionQL:
    def __init__(self, params_dict:dict, dataset:dict, ft:bool=False):
        self.ALG = 'Diffusion-QL'
        self.ENV_CONFIG = params_dict['environment']
        self.NUM_SAMPLE = dataset['arr_0']
        self.STATE_DIM = int(params_dict['state_dim'])
        self.ACTION_DIM = int(params_dict['action_dim'])

        # Initialise replay buffers
        self.state_buffer = torch.tensor(dataset['arr_1'], dtype=torch.float32)            # State buffer (unnormalised)
        self.next_state_buffer = torch.tensor(dataset['arr_4'], dtype=torch.float32)       # Next state buffer (unnormalised)
        self.action_buffer = torch.tensor(dataset['arr_2'], dtype=torch.float32)           # Action buffer
        self.reward_buffer = torch.tensor(dataset['arr_3'], dtype=torch.float32)           # Reward buffer
        self.MINIBATCH_SIZE = 256

        # Initialise Diffusion Model's Parameters
        self.initialiseDiffusionParams(schedule='vp', beta_min=1, beta_max=10, num_step=5, dec_step=5)

        # Initialise Paths
        self.training_record_path = 'Diffusion_QL/training_records/{:s}_{:s}_training_record'.format(self.ALG, self.ENV_CONFIG)
        self.training_record_2m_path = 'Diffusion_QL/training_records/{:s}_{:s}_training_record_2m'.format(self.ALG, self.ENV_CONFIG)
        self.training_checkpoint_path = 'Diffusion_QL/training_records/{:s}_{:s}_checkpoint'.format(self.ALG, self.ENV_CONFIG)
        self.ft_training_record_path = 'Diffusion_QL/training_records/{:s}_{:s}_ft_training_record'.format(self.ALG, self.ENV_CONFIG)
        self.ft_checkpoint_path = 'Diffusion_QL/training_records/{:s}_{:s}_ft_checkpoints.pth'.format(self.ALG, self.ENV_CONFIG)  # Online Saving
        self.evaluation_path = 'Diffusion_QL/norm_eval_rewards_append'

        # Initialise neural networks
        self.EPSILON_INPUT_DIM = self.STATE_DIM + self.ACTION_DIM + self.POS_DIM
        self.EPSILON_BEH_INPUT_DIM = self.ACTION_DIM + self.POS_DIM
        self.Q_INPUT_DIM = self.STATE_DIM + self.ACTION_DIM
        self.V_INPUT_DIM = self.STATE_DIM

        # Check for training record (offline)...
        # Check for the training record file and the response from user...
        if os.path.isfile(self.training_record_path) is True:
            print('========== ({:s}) There exists a training record for this agent. Do you wish to load the exist one ?'.format(self.ALG))
            _ans_1 = input('========== ({:s}) Press [y/n] and enter: '.format(self.ALG))
        else:
            _ans_1 = 'n'
        # Check for the answer...
        if _ans_1 == 'n' or _ans_1=='N' or _ans_1=='No' or _ans_1=='NO':
            print('========== ({:s}) Create new record and models!'.format(self.ALG))
            self.training_record = 0
            self.norm_reward_training_append = []
            self.best_norm_reward_training = 0
            self.epsilon_loss_append = []
            self.q_1_loss_append = []
            self.q_2_loss_append = []
            self.epsilon = my_NN.MLP(input_dim=self.EPSILON_INPUT_DIM, output_dim=self.ACTION_DIM)
            self.q_1 = my_NN.MLP(input_dim=self.Q_INPUT_DIM, output_dim=1)
            self.q_2 = my_NN.MLP(input_dim=self.Q_INPUT_DIM, output_dim=1)
            # Initialise Target models
            self.epsilon_tar = self.epsilon
            self.q_1_tar = self.q_1
            self.q_2_tar = self.q_2
        elif _ans_1 == 'y' or _ans_1=='Y' or _ans_1=='Yes' or _ans_1=='YES':
            print('========== ({:s}) Load training record.'.format(self.ALG))
            self.loadTrainingRecord()  # Load from the method here <--------- ****
        else:
            print('========== ({:s}) Try another answer. ("y" for yes (create new) or "n" for no (load existing one)).'.format(self.ALG))
            exit()

        # Declare optimizer for the networks (offline training)
        self.epsilon_optimizer = torch.optim.Adam(self.epsilon.parameters(), lr=3e-04)
        self.q_1_optimizer = torch.optim.Adam(self.q_1.parameters(), lr=3e-04)
        self.q_2_optimizer = torch.optim.Adam(self.q_2.parameters(), lr=3e-04)
        # Set to cuda if GPU is available.
        if CUDA:
            self.epsilon.cuda()
            self.q_1.cuda()
            self.q_2.cuda()
            self.epsilon_tar.cuda()
            self.q_1_tar.cuda()
            self.q_2_tar.cuda()

    # Training Record Loading Method
    def loadTrainingRecord(self):
        # _loaded_training_record = torch.load(self.training_record_path, 
        #                                      map_location=torch.device('cpu' if not CUDA else 'cuda'))
        _loaded_training_record = torch.load(self.training_checkpoint_path, 
                                             map_location=torch.device('cpu' if not CUDA else 'cuda'))
        self.training_record = _loaded_training_record['training_record']
        self.norm_reward_training_append = _loaded_training_record['norm_reward_training_append']
        self.epsilon = _loaded_training_record['epsilon']
        self.q_1 = _loaded_training_record['q_1']
        self.q_2 = _loaded_training_record['q_2']
        self.epsilon_tar = _loaded_training_record['epsilon_tar']
        self.q_1_tar = _loaded_training_record['q_1_tar']
        self.q_2_tar = _loaded_training_record['q_2_tar']
        self.epsilon_loss_append = _loaded_training_record['epsilon_beh_loss_append']
        self.q_1_loss_append = _loaded_training_record['q_1_loss_append']
        self.q_2_loss_append = _loaded_training_record['q_2_loss_append']
        print('========== ({:s}) Training Record: '.format(self.ALG), self.training_record, ' epoch.', 
              ', Gradient steps: ', self.training_record*(self.NUM_SAMPLE//self.MINIBATCH_SIZE))

    # Diffusion model's parameters initialisation method
    def initialiseDiffusionParams(self, schedule='vp', beta_min=0.1, beta_max=10, num_step=50, dec_step=10):
        # Intialise Diffusion Model Parameters
        self.DIFFU_STEPS = num_step
        self.DEC_DIFFU_STEPS = dec_step
        self.BETA_MIN = beta_min
        self.BETA_MAX = beta_max
        self.MIN_DIFFU_SPACE = torch.tensor(-1., dtype=torch.float32)
        self.MAX_DIFFU_SPACE = torch.tensor(1., dtype=torch.float32)
        self.POS_DIM = 4
        self.beta = np.zeros([self.DIFFU_STEPS, 1], dtype=float)
        self.alpha = np.zeros([self.DIFFU_STEPS, 1], dtype=float)
        self.alpha_bar = np.zeros([self.DIFFU_STEPS, 1], dtype=float)
        self.DIFFU_MEAN = torch.tensor(0., dtype=torch.float32)
        self.DIFFU_VAR = torch.tensor(1., dtype=torch.float32)
        self.DIFFU_STD = torch.sqrt(self.DIFFU_VAR)
        self.reverse_mean = 0.
        self.reverse_cov = 0.

        # Create sinusoidal position encoding array. (Pre-defined)
        def _sinPositionEncoding(seq_len, dim, N=10000):
            output = np.zeros([seq_len, dim], dtype=float)
            for k in range(seq_len):
                for i in np.arange(int(dim/2)):
                    denominator = np.power(N, 2*i/dim)
                    output[k, 2*i] = np.sin(k/denominator)
                    output[k, 2*i+1] = np.cos(k/denominator)
            return output
        self.POS_EMB = _sinPositionEncoding(seq_len=self.DIFFU_STEPS, dim=self.POS_DIM)  

        # ----- Derive the diffusion schedule
        # For the sake of simplicity, the order of the element in the array represents the forward process order.
        # For example, B_{fp} = [1, 2, 3, ..., N-1, N]  (the elements represent indices.)
        # In order to use for the reverse process, The elements have to be flip the other way around.
        # For example, B_{rp} = [N, N-1, N-2, ..., 2, 1] (the elements represent indices.)
        # Always remember that the diffusion process has the minimum time step is 1 and maximum at N. B = [1, 2, 3, ..., N]
        # ----- Variance Preserving Schedule (DQL)
        if schedule == 'vp':
            # Derive the diffusion rate (noise schedule) 
            # which means, if it is the reverse, we have to reverse the order of the array.
            for i in range(self.DIFFU_STEPS):
                # Formular for the alpha: np.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
                self.alpha[i] = np.exp((-self.BETA_MIN * 1/self.DIFFU_STEPS) - (0.5 * (self.BETA_MAX-self.BETA_MIN) * (2*(i+1)-1)/(np.square(self.DIFFU_STEPS))))
                self.beta[i] = 1 - self.alpha[i]
                self.alpha_bar[i] = np.prod(self.alpha[0:i+1])

        # ----- Linear Schedule (Same as original DDPM)
        elif schedule == 'linear':        
            self.beta = np.linspace(start=self.BETA_MIN, stop=self.BETA_MAX, num=self.DIFFU_STEPS)
            for i in range(self.DIFFU_STEPS):
                self.alpha[i] = 1 - self.beta[i]
                self.alpha_bar[i] = np.prod(self.alpha[0:i+1])
        
        # Convert them into tensor format.
        self.beta = torch.tensor(self.beta, dtype=torch.float32)
        self.alpha = torch.tensor(self.alpha, dtype=torch.float32)
        self.alpha_bar = torch.tensor(self.alpha_bar, dtype=torch.float32)
        self.POS_EMB = torch.tensor(self.POS_EMB, dtype=torch.float32)

    # Diffusion process (forward process) method
    def forwardProcess(self, data, epsilon, step:int=None):
        alpha_bars = self.alpha_bar[step]
        x_t_p_1 = (torch.sqrt(alpha_bars) * data) + (torch.sqrt(1-alpha_bars) * epsilon)
        return x_t_p_1

    # Original Reverse Process Method
    def reverseProcess(self, inputs:list, size:int, predictor='main'):
        # Initialise noisy data (needed to be clipped).
        x_T = torch.normal(size=[size, self.ACTION_DIM], mean=self.DIFFU_MEAN, std=self.DIFFU_STD)

        # Start reverse processes
        for i in range(self.DEC_DIFFU_STEPS): 
            # Define reverse position indices (Have to be inverse since the diffusion schedule was created in forward process's order.)
            rev_pos = self.DEC_DIFFU_STEPS-i-1
            # rev_pos_emb = tf.reshape(tf.repeat(repeats=size, input=self.POS_EMB[rev_pos], axis=0), [-1, self.POS_DIM])
            rev_pos_emb = self.POS_EMB[rev_pos].repeat(size, 1)

            # Predict the noise for the reverse process (e_{theta})
            if predictor == 'main':
                epsilon_theta_t = self.epsilon(torch.concat((inputs, x_T, rev_pos_emb), dim=1))
            elif predictor == 'target':
                epsilon_theta_t = self.epsilon_tar(torch.concat((inputs, x_T, rev_pos_emb), dim=1))

            # Reverse process (Stochastic) (DDPM)
            x_t_m_1  = (x_T / torch.sqrt(self.alpha[rev_pos])) - \
                       (self.beta[rev_pos] * epsilon_theta_t / torch.sqrt(self.alpha[rev_pos]*(1-self.alpha_bar[rev_pos])))

            if i == (self.DEC_DIFFU_STEPS-1):
                x_t_m_1 += (torch.sqrt(self.beta[rev_pos]) * 0) 
            else:
                x_t_m_1 += (torch.sqrt(self.beta[rev_pos]) * torch.normal(size=[size, self.ACTION_DIM], mean=self.DIFFU_MEAN, std=self.DIFFU_STD))
            
            x_T = x_t_m_1  # Update the previous a (a_i) for the next iteration.

        return torch.clip(x_t_m_1, min=self.MIN_DIFFU_SPACE, max=self.MAX_DIFFU_SPACE)

    # Output Prediction Method (Main network)
    def predict(self, state, size:int):
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32)
            state = torch.reshape(state, [-1, self.STATE_DIM])
        return torch.squeeze(self.reverseProcess(state, size, predictor='main'))

    # Output Prediction Method (Target network)
    def predictTar(self, state, size:int):
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=torch.float32)
            state = torch.reshape(state, [-1, self.STATE_DIM])
        return torch.squeeze(self.reverseProcess(state, size, predictor='target'))
    
    # Combine memories
    def combineMemories(self, states, actions, rewards, next_states):
        if not torch.is_tensor(states):
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.float32)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.float32)
        # This is for the diffusion models only
        self.state_buffer = torch.cat((self.state_buffer, states), dim=0)
        self.action_buffer = torch.cat((self.action_buffer, actions), dim=0)
        self.reward_buffer = torch.cat((self.reward_buffer, rewards), dim=0)
        self.next_state_buffer = torch.cat((self.next_state_buffer, next_states), dim=0)
        print('Memories combined! ', ', memories_size: ', self.state_buffer.size())

    # Training method (for offliine training)
    def training(self, total_epoch:int, eval:bool=False, ft:bool=False):
        # Get Expected Bellman's Equation (local)
        def _getExpectedQValues(inputs):
            return batch_reward_tensor + GAMMA * torch.minimum(self.q_1_tar(inputs), self.q_2_tar(inputs))
        
        # Q network Training Method (local)
        def _trainQ1Network(inputs, y_true):
            self.q_1_optimizer.zero_grad()
            y_pred = self.q_1(inputs)
            _q_1_loss = torch.mean(torch.square(y_true - y_pred))

            return _q_1_loss

        # Q network Training Method (local)
        def _trainQ2Network(inputs, y_true):
            self.q_2_optimizer.zero_grad()
            y_pred = self.q_2(inputs)
            _q_2_loss = torch.mean(torch.square(y_true - y_pred))

            return _q_2_loss

        # Diffusion Models Training Method (local)
        def _trainDiffusionBeh(inputs, y_true):
            residual_noise = self.epsilon(inputs)
            # Compute for the MSE.
            _diffu_loss = torch.mean(torch.square(y_true - residual_noise), dim=1, keepdim=True)

            # Derived Q term
            _pred_a = self.predict(state=batch_state_tensor, size=_batch_size)
            if np.random.random() < 0.5:
                _q = self.q_1(torch.concat([batch_state_tensor, _pred_a], dim=1))
                _norm_q = torch.mean(torch.abs(self.q_1(torch.concat([batch_state_tensor, batch_action_tensor], dim=1))))
            else:
                _q = self.q_2(torch.concat([batch_state_tensor, _pred_a], dim=1))
                _norm_q = torch.mean(torch.abs(self.q_2(torch.concat([batch_state_tensor, batch_action_tensor], dim=1))))
            _q_term = torch.mean(_q) / _norm_q

            # Mean of Loss 
            _diffu_loss = torch.mean(_diffu_loss) - _q_term
            return _diffu_loss

        # Target Networks Updating Method
        def _updateTargetNetworks():
            # Update the target networks
            epsilon_tar_state_dict = self.epsilon_tar.state_dict()
            epsilon_state_dict = self.epsilon.state_dict()
            for key1 in epsilon_state_dict:
                epsilon_tar_state_dict[key1] = epsilon_state_dict[key1]*TAU + epsilon_tar_state_dict[key1]*(1-TAU)
            self.epsilon_tar.load_state_dict(epsilon_tar_state_dict)

            q_1_tar_state_dict = self.q_1_tar.state_dict()
            q_1_state_dict = self.q_1.state_dict()
            for key2 in q_1_state_dict:
                q_1_tar_state_dict[key2] = q_1_state_dict[key2]*TAU + q_1_tar_state_dict[key2]*(1-TAU)
            self.q_1_tar.load_state_dict(q_1_tar_state_dict)

            q_2_tar_state_dict = self.q_2_tar.state_dict()
            q_2_state_dict = self.q_2.state_dict()
            for key3 in q_2_state_dict:
                q_2_tar_state_dict[key3] = q_2_state_dict[key3]*TAU + q_2_tar_state_dict[key3]*(1-TAU)
            self.q_2_tar.load_state_dict(q_2_tar_state_dict)

        # ========================= Training Loop Start =========================
        # Extract dataset
        buffer_size = self.NUM_SAMPLE
        state_buffer = self.state_buffer
        action_buffer = self.action_buffer
        reward_buffer = self.reward_buffer
        next_state_buffer = self.next_state_buffer

        _batch_size = 256
        _num_gradient_step = buffer_size//_batch_size 
        _training_record = self.training_record if not ft else 0

        # Set models to training mode.
        self.epsilon.train()
        self.q_1.train()
        self.q_2.train()
        self.epsilon_tar.train()
        self.q_1_tar.train()
        self.q_2_tar.train()

        # Train main models
        while _training_record < total_epoch:
            start_time = time()
            diffu_loss_accum = 0
            q_1_loss_accum = 0
            q_2_loss_accum = 0

            # Get shuffle indices
            _sampling_indices = torch.randperm(buffer_size)

            # Start gradient steps loop
            for g in range(_num_gradient_step):
                # Get training batches
                batch_state_tensor = state_buffer[_sampling_indices[0+(g*_batch_size):_batch_size+(g*_batch_size)]]
                batch_action_tensor = action_buffer[_sampling_indices[0+(g*_batch_size):_batch_size+(g*_batch_size)]]
                batch_reward_tensor = reward_buffer[_sampling_indices[0+(g*_batch_size):_batch_size+(g*_batch_size)]]
                batch_next_state_tensor = next_state_buffer[_sampling_indices[0+(g*_batch_size):_batch_size+(g*_batch_size)]]

                # Prepare data for q learning
                _next_action = self.predictTar(state=batch_next_state_tensor, size=_batch_size)
                y_true = _getExpectedQValues(inputs=torch.concat([batch_state_tensor, _next_action], dim=1))
                q_1_loss = _trainQ1Network(inputs=torch.concat([batch_next_state_tensor, batch_action_tensor], dim=1), y_true=y_true) # State-Action network (Q)
                q_2_loss = _trainQ2Network(inputs=torch.concat([batch_next_state_tensor, batch_action_tensor], dim=1), y_true=y_true) # State-Action network (Q)
                q_1_loss_accum += q_1_loss.tolist()
                q_2_loss_accum += q_2_loss.tolist()

                self.q_1_optimizer.zero_grad()
                self.q_2_optimizer.zero_grad()
                q_1_loss.backward(retain_graph=True)
                q_2_loss.backward()
                self.q_1_optimizer.step()
                self.q_2_optimizer.step()

                # Prepare data for diffusion learning
                rand_t = torch.randint(low=0, high=self.DIFFU_STEPS, size=[_batch_size])
                encode_t_tensor = self.POS_EMB[rand_t]  # Retrieve
                epsilon_tensor = torch.normal(mean=self.DIFFU_MEAN, std=self.DIFFU_STD, size=[_batch_size, self.ACTION_DIM])
                forward_action_tensor = self.forwardProcess(data=batch_action_tensor, epsilon=epsilon_tensor, step=rand_t)
                self.epsilon_optimizer.zero_grad()
                diffu_loss = _trainDiffusionBeh(inputs=torch.concat([batch_state_tensor, forward_action_tensor, encode_t_tensor], dim=1), y_true=epsilon_tensor)
                diffu_loss.backward()
                self.epsilon_optimizer.step()
                diffu_loss_accum += diffu_loss.tolist()

                # Update target networks
                _updateTargetNetworks()

            # Append loss for recording.
            self.epsilon_loss_append.append(diffu_loss_accum/_num_gradient_step)
            self.q_1_loss_append.append(q_1_loss_accum/_num_gradient_step)
            self.q_2_loss_append.append(q_2_loss_accum/_num_gradient_step)

            # Increase training record after epoch finished.
            _training_record += 1
            self.training_record += 1

            # Print the status.
            print('Epoch: ', self.training_record,
                  ', Gradient_step: ', int(self.training_record*_num_gradient_step), 
                  ', Diffu_loss: ', round(diffu_loss_accum/_num_gradient_step, 4),
                  ', Q1_loss: ', round(q_1_loss_accum/_num_gradient_step, 4),
                  ', Q2_loss: ', round(q_2_loss_accum/_num_gradient_step, 4),
                  ', Time/Epoch: ', round(time()-start_time, 4))
               
            # Save the training_records
            self.recordSaving(path=self.training_record_path if not ft else self.ft_training_record_path)
            # Save the 2M training records (same gradient steps as baseline (SAC)).
            if self.training_record == 512 and not ft:
                self.recordSaving(path=self.training_record_2m_path)

            # Check for the breaking for evaluation.
            if eval and _training_record % 10 == 0:
                self.epsilon.eval()
                self.q_1.eval()
                self.q_2.eval()
                break
        # ========================== Training Loop End ==========================

    # Training Record Saving Method
    def recordSaving(self, path:str):
        torch.save({'training_record': self.training_record,
                    'norm_reward_training_append': self.norm_reward_training_append,
                    'epsilon': self.epsilon, 
                    'q_1': self.q_1,
                    'q_2': self.q_2, 
                    'epsilon_tar': self.epsilon_tar,
                    'q_1_tar': self.q_1_tar,
                    'q_2_tar': self.q_2_tar,
                    'epsilon_beh_loss_append': self.epsilon_loss_append, 
                    'q_1_loss_append': self.q_1_loss_append, 
                    'q_2_loss_append': self.q_2_loss_append}, path)