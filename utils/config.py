import logging
import pickle

import seaborn as sb
import torch

sb.set_style('whitegrid')

# Class for efficiently handling configurations and parameters, enables to
# easily set them and remember them when one config is reused
# Read with config.key, set with config.update({'key': value}) or config[
# 'key'] = value

default_config = dict(true_meas_noise_var=0.,
                      process_noise_var=0.,
                      simu_solver='dopri5',
                      nb_rollouts=0,
                      nb_loops=1,
                      rollout_length=100,
                      sliding_window_size=None,
                      verbose=False,
                      monitor_experiment=True,
                      multioutput_GP=False,
                      sparse=None,
                      memory_saving=False,
                      restart_on_loop=False,
                      meas_noise_var=0.1,
                      batch_adaptive_gain=None,
                      nb_plotting_pts=500,
                      no_control=False,
                      full_rollouts=False,
                      max_rollout_value=500)


class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Check that necessary keys have been filled in
        mandatory_keys = ['system', 'nb_samples', 't0_span', 'tf_span', 't0',
                          'tf']
        for key in mandatory_keys:
            assert key in self, 'Mandatory key ' + key \
                                + ' was not given.'
        self['dt'] = (self.tf - self.t0) / (self.nb_samples - 1)
        self['t_eval'] = torch.linspace(self.t0, self.tf, self.nb_samples)
        if 'Continuous_model' in self['system']:
            self['continuous_model'] = True
        else:
            self['continuous_model'] = False
        if torch.cuda.is_available():
            self['cuda_device'] = 'cuda:' + str(
                torch.cuda.current_device())
        else:
            self['cuda_device'] = 'cpu'

        # Fill other keys with default values
        for key in default_config:
            if key not in self:
                self[key] = default_config[key]
        if 'rollout_controller' not in self:
            self['rollout_controller'] = \
                {'random': self['nb_rollouts']}

        # Warn / assert for specific points
        if self.t0 != 0:
            logging.warning(
                'Initial simulation time is not 0 for each scenario! This is '
                'incompatible with DynaROM.')
        assert not (self.batch_adaptive_gain and ('adaptive' in self.system)), \
            'Cannot adapt the gain both through a continuous dynamic and a ' \
            'batch adaptation law.'

        # Check same number of rollouts as indicated in rollout_controller
        nb_rollout_controllers = 0
        for key, val in self['rollout_controller'].items():
            nb_rollout_controllers += val
        assert nb_rollout_controllers == self['nb_rollouts'], \
            'The number of rollouts given by nb_rollouts and ' \
            'rollout_controller should match.'

        # Check if contains init_state_obs_T but not the equivalent for u,
        # in which case they are equal
        if self.init_state_obs_T and self.init_state_obs_Tu is None:
            self['init_state_obs_Tu'] = self.init_state_obs_T

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            if item.startswith('__') and item.endswith('__'):
                raise AttributeError(item)
            else:
                if not 'reg' in item:
                    logging.info(f'No item {item}')
                return None

    def __setattr__(self, item, value):
        try:
            self[item] = value
        except KeyError:
            raise AttributeError(item)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, d):
        self.__dict__.update(d)

    def save_to_file(self, filename):
        with open(filename, 'w') as f:
            for key, val in self.items():
                print(key, ': ', val, file=f)


class Test:

    def __init__(self, config: Config):
        self.a = 0
        self.config = config

    def __getattr__(self, item):
        return self.config.__getattr__(item)


if __name__ == '__main__':
    config = Config(system='Continuous/Louise_example/Basic_Louise_case',
                    nb_samples=int(1e4),
                    t0_span=0,
                    tf_span=int(1e2),
                    t0=0,
                    tf=int(1e2),
                    hyperparam_optim='fixed_hyperparameters')
    test = Test(config)
    print(test.config, test.config.t0, config.t0)
    print('Test keys:')
    for key in test.config:
        print(key, test.config[key])
