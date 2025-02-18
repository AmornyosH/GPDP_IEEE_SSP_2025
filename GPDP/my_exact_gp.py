import os
import torch
import numpy as np
from time import time

class myExactGP(torch.nn.Module):
    def __init__(self, params_dict:dict, dataset:dict, cuda:bool=False):
        super().__init__()
        self.alg = 'exact_gp'
        self.env = params_dict['environment']
        self.num_sample = 1000  # Fixed to 1000.
        self.x_dim = int(params_dict['state_dim'])
        self.y_dim = int(params_dict['action_dim'])
        # self.training_record_path = 'GPDP/training_records/GPDP_{}_{}_training_records'.format(self.env, self.alg)
        # self.training_checkpoint_path = 'GPDP/training_records/GPDP_{}_{}_checkpoint'.format(self.env, self.alg)
        self.training_record_path = 'GPDP/training_records/GPDP_2_{}_{}_training_records'.format(self.env, self.alg)
        self.training_checkpoint_path = 'GPDP/training_records/GPDP_2_{}_{}_checkpoint'.format(self.env, self.alg)
        self.cuda() if cuda else ...

        # Check for the training record file and the response from user...
        if os.path.isfile(self.training_record_path) is True:
            print('========== ({:s}) There exists a training record for this agent. Do you wish to load the exist one ?'.format(self.alg))
            _ans_1 = input('========== ({:s}) Press [y/n] and enter: ')
        else:
            _ans_1 = 'n'
        # Check for the answer...
        if _ans_1 == 'n' or _ans_1=='N' or _ans_1=='No' or _ans_1=='NO':
            print('========== Create new GP record and models!')
            self.mll_append = []  # Loss storage
            # Create hyperparameters
            self.sigma_n = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=True)
            self.ell = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32), requires_grad=True)
            _start = 0
            self.x_train = torch.tensor(dataset['arr_1'][0+_start:_start+self.num_sample], dtype=torch.float32)
            self.y_train = torch.tensor(dataset['arr_2'][0+_start:_start+self.num_sample], dtype=torch.float32)

        elif _ans_1 == 'y' or _ans_1=='Y' or _ans_1=='Yes' or _ans_1=='YES':
            print('========== Load training record.')
            # Load training records
            _training_records = torch.load(self.training_record_path, map_location=torch.device('cuda' if cuda else 'cpu'))
            self.mll_append = _training_records['loss_append']  # Loss append
            _state_dict = _training_records['state_dict']
            self.sigma_n = torch.nn.Parameter(_state_dict['sigma_n'], requires_grad=True)
            self.ell = torch.nn.Parameter(_state_dict['ell'], requires_grad=True)
            self.x_train = _training_records['x_train']
            self.y_train = _training_records['y_train']
            self.eval()
        else:
            print('========== Try another answer. ("y" for yes (create new) or "n" for no (load existing one)).')
            exit()

        # Initialised training kernels (K, K_inv).
        self.sigma_p = torch.tensor(2.0, dtype=torch.float32)  # Fixed signal variance to 2.00^2
        self.K = self.rbfKernel(X_1=self.x_train, X_2=self.x_train, noise=True)
        self.K_inv = torch.linalg.inv(self.K)

    def rbfKernel(self, X_1, X_2, noise=False):
        '''
        Isotropic squared exponential kernel.
        Args:
            X1: Array of m points (m x d).
            X2: Array of n points (n x d).
        Returns:
            (m x n) matrix.
        '''
        X_1 = torch.tensor(X_1, dtype=torch.float32) if not torch.is_tensor(X_1) else X_1
        X_2 = torch.tensor(X_2, dtype=torch.float32) if not torch.is_tensor(X_2) else X_2
        kernel = (self.sigma_p**2) * torch.exp(-(torch.cdist(X_1, X_2)**2)/(2*self.ell**2))
        if noise:
            # Noisy observation
            kernel += ((self.sigma_n**2) * torch.eye(len(X_1)))

        return kernel

    def predict(self, X_s):
        with torch.no_grad():
            x_test = X_s.view(-1, self.x_dim)
            _k_s = self.rbfKernel(X_1=self.x_train, X_2=x_test, noise=False)
            _k_ss = self.rbfKernel(X_1=x_test, X_2=x_test, noise=False)

            # We found that deriving mean and variance this way, provide better results.
            _mean = _k_s.T @ self.K_inv @ self.y_train
            _var = _k_ss - _k_s.T @ self.K_inv @ _k_s

            # # Cholesky decomposition
            # _L = torch.linalg.cholesky(self.K, upper=False)
            # _alpha = torch.linalg.solve_triangular(_L.T, torch.linalg.solve_triangular(_L, self.y_train, upper=False), upper=True)
            # _mean = _k_s.T @ _alpha
            # _v = torch.linalg.solve_triangular(_L, _k_s, upper=False)
            # _var = _k_ss - (_v.T @ _v)

        return _mean, _var

    def myTraining(self, total_epoch:int, ft:bool=False):
        _batch_size = self.num_sample if not ft else self.x_train.size()[0]  # minibatch size is the whole dataset.
        _gradient_step = 1
        _optimizer = torch.optim.Adam(self.parameters(), lr=1e-02)
        self.train()  # Set to training mode.
        _training_record = 0
        while _training_record < total_epoch:
            _start_time = time()
            for g in range(_gradient_step):
                _batch_x = self.x_train[g*_batch_size:_batch_size+(g*_batch_size)]
                _batch_y = self.y_train[g*_batch_size:_batch_size+(g*_batch_size)]

                _optimizer.zero_grad()
                _k = self.rbfKernel(X_1=_batch_x, X_2=_batch_x, noise=True)

                _L = torch.linalg.cholesky(_k, upper=False)
                _alpha = torch.linalg.solve_triangular(_L.T, torch.linalg.solve_triangular(_L, _batch_y, upper=False), upper=True)

                _mll = (-0.5 * _batch_y.T @ _alpha) - torch.sum(torch.diagonal(_L)) - \
                       (_batch_size*torch.log(torch.tensor(2*torch.pi))/2)
                _mll = -_mll.mean()
                _mll.backward(retain_graph=True)
                _optimizer.step()

                self.mll_append.append(_mll.tolist())

            _training_record += 1
        
        print('Gradient_Step: ', _training_record, 
                ', Loss: ', _mll.tolist(), 
                ', Time_usage: ', round(time()-_start_time, 4))

        self.recordSaving(path=self.training_record_path)
        
        self.K = self.rbfKernel(X_1=self.x_train, X_2=self.x_train, noise=True)
        self.K_inv = torch.linalg.inv(self.K)
        self.eval()  # Set to evaluation mode after the training is finished.

    def recordSaving(self, path:str):
        torch.save({'state_dict': self.state_dict(), 
                    'loss_append': self.mll_append,
                    'x_train': self.x_train, 
                    'y_train': self.y_train}, path)