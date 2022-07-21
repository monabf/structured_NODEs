import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn

from utils.config import Config
from utils.utils import reshape_pt1, reshape_dim1
from .learn_neural_ODE import Learn_NODE_difftraj

# Class to learn ODEs with NNs. Heritates from Learn_NODE_difftraj, but made
# for case where we have no ground truth to simulate our training and test
# trajectories. Rather, we have a certain number of training and test
# trajectories measured experimentally, and run Learn_NODE_difftraj with them.

LOCAL_PATH = os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
LOCAL_PATH_TO_SRC = LOCAL_PATH.split(os.sep + 'src', 1)[0]


class Learn_NODE_difftraj_exp_datasets(Learn_NODE_difftraj):
    """
    Learner that trains on a set of experimental training trajectories of given
    size, i.e. on several trajectories x0 -> xN of different length.

    Since these are experimental trajectories, if partial_obs is true,
    then we only have access to the observations and not to the full
    trajectories not only for training (as before) but also for testing: test
    rollouts only on observations. The dataset of trajectories is split into
    training, validation and testing trajectories (used in place of rollouts
    that are generated using the ground truth in the previous classes).

    All tensors concerning trajectories assumed of size nb_difftraj x N x n,
    i.e. all trajectories assumed of same length with same sampling times!
    Necessary for parallel simulations.
    """

    def __init__(self, X_train, U_train, submodel: nn.Module, config: Config,
                 X_test=None, U_test=None, sensitivity='autograd',
                 ground_truth_approx=True, validation=True,
                 dataset_on_GPU=False):
        self.X_test = X_test
        self.U_test = U_test
        super().__init__(X_train=X_train, U_train=U_train, submodel=submodel,
                         config=config, sensitivity=sensitivity,
                         ground_truth_approx=ground_truth_approx,
                         validation=validation, dataset_on_GPU=dataset_on_GPU)
        self.variables['X_test'] = self.X_test
        self.variables['U_test'] = self.U_test
        if X_test is None:
            self.variables['train_idx'] = self.train_idx
            self.variables['test_idx'] = self.test_idx

    def create_grid(self, constrain_u, grid_inf, grid_sup):
        # Create random grid for evaluation
        if self.difftraj:
            self.X_train, self.U_train = self.view_difftraj(
                self.X_train, self.U_train)
        nb_points = int(np.ceil(np.min([len(self.X_train), 1000])))
        self.grid_random_idx = torch.randint(0, len(self.X_train),
                                             size=(nb_points,))
        grid = reshape_pt1(self.X_train[self.grid_random_idx])
        grid_controls = reshape_pt1(self.U_train[self.grid_random_idx])
        if self.difftraj:
            self.X_train, self.U_train = self.unview_difftraj(
                self.X_train, self.U_train)
        return grid, grid_controls

    def create_true_predicted_grid(self, grid, grid_controls):
        if self.difftraj:
            self.X_train, self.U_train = self.view_difftraj(
                self.X_train, self.U_train)
        true_predicted_grid = reshape_pt1(self.X_train[self.grid_random_idx])
        if self.difftraj:
            self.X_train, self.U_train = self.unview_difftraj(
                self.X_train, self.U_train)
        return true_predicted_grid

    def create_rollout_list(self):
        # When no ground truth is available, rollouts are actually test data
        if self.X_test is None:
            # Create test rollouts from X_train
            if self.config.test_size is None:
                test_size = 0.38  # 0.3 of full dataset, so 0.38 after val split
            else:
                test_size = self.test_size
            self.train_test_split = train_test_split(
                np.arange(self.X_train.shape[0], dtype=int),
                test_size=test_size, shuffle=True)
            self.test_idx = self.train_idx.copy()[self.train_test_split[1]]
            self.train_idx = self.train_idx[self.train_test_split[0]]
            train_split = torch.as_tensor(self.train_test_split[0],
                                          device=self.X_train.device)
            test_split = torch.as_tensor(self.train_test_split[1],
                                         device=self.X_train.device)
            self.X_train, self.X_test = \
                torch.index_select(self.X_train, dim=0, index=train_split), \
                torch.index_select(self.X_train, dim=0, index=test_split)
            self.U_train, self.U_test = \
                torch.index_select(self.U_train, dim=0, index=train_split), \
                torch.index_select(self.U_train, dim=0, index=test_split)
            self.init_state_estim, self.init_state_estim_test = \
                torch.index_select(
                    self.init_state_estim, dim=0, index=train_split), \
                torch.index_select(
                    self.init_state_estim, dim=0, index=test_split)
            if self.init_state_model:
                self.init_state_obs, self.init_state_obs_test = \
                    torch.index_select(
                        self.init_state_obs, dim=0, index=train_split), \
                    torch.index_select(
                        self.init_state_obs, dim=0, index=test_split)
        if self.difftraj:
            # self.nb_difftraj = len(train_split)
            # self.nb_difftraj_test = len(test_split)
            # self.nb_rollouts = len(test_split)
            # self.specs['nb_difftraj_train'] = len(train_split)
            # self.specs['nb_difftraj_test'] = len(test_split)
            self.nb_difftraj = len(self.X_train)
            self.nb_difftraj_test = len(self.X_test)
            self.nb_rollouts = len(self.X_test)
            self.specs['nb_difftraj_train'] = len(self.X_train)
            self.specs['nb_difftraj_test'] = len(self.X_test)

        # TODO handling of rollouts is slow (for loop instead of parallel
        #  simulation), should run them all in parallel like regular NODE!
        rollout_list = []
        i = 0
        while i < self.nb_rollouts:
            if self.step > 0:
                # Only initialize rollout list at beginning of each fold
                return self.rollout_list
            if self.init_state_model:
                # For ground_truth_approx, init_state in rollout_list
                # contains the inputs to the recognition model for Xtest
                # since it is known anyway, so that it can be used in NODE
                # rollouts directly
                init_state = reshape_pt1(self.init_state_obs_test[i, 0])
            else:
                init_state = reshape_pt1(self.X_test[i, 0])
            control_traj = reshape_dim1(self.U_test[i])
            true_mean = reshape_dim1(self.X_test[i])
            rollout_list.append([init_state, control_traj, true_mean])
            i += 1
        return rollout_list
