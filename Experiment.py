import random as rand
import numpy as np
import matplotlib.pyplot as plt
import torch
import gym
import os
from LODA import LODA  

class Experiment:

    """
        experiment_dict = {
            'name' : [FIXED]
            'input_dim' :  [FIXED]
            'intrinsic_dims' : [VARIABLE] {1, 4, 16, 64}
            'action_dim' : [FIXED]
            'policy_lr' : [FIXED] {critic_approximator_lr / e}
            'gradient_decays' : 0.9, [VARIABLE] {0.9, 0.5}
            'buffer_size' : 1024, [FIXED] 4096
            'intrinsic_weights' : 10, [VARIABLE] {0.1, 0.001, 10}
            'r_net_hidden_dim' : env.observation_space.shape[0] * 2, [FIXED]
            'ppo_epsilons' : 0.2, [VARIABLE] {None, 0.2}
            'reward_discount' : 0.95, [FIXED] 
            'seed' : seed [FIXED] 42
            'non_linearities' : [VARIABLE] {'relu', 'tanh'}
            'critic_approximator_lrs' : [VARIABLE] {3e-3, 3e-4}
            'critic_policy_hidden_dim' : [FIXED] env.observation_space.shape[0]
            'env_id' : [FIXED]
            'steps' : [FIXED]
            'eval_percentage' : [FIXED]
        }
    """

    def __init__(self, experiment_dict, root, runs=None):
        dummy_env = gym.make(experiment_dict['env_id'])
        self.experiment_dict = experiment_dict
        self.name = experiment_dict['name']
        self.input_dim = dummy_env.observation_space.shape[0]
        self.possible_intrinsic_dims = experiment_dict['intrinsic_dims']
        self.buffer_size = experiment_dict['buffer_size']
        self.action_dim = dummy_env.action_space.n
        self.possible_gradient_decays = experiment_dict['gradient_decays']
        self.possible_intrinsic_weights = experiment_dict['intrinsic_weights']
        self.possible_ppo_epsilons = experiment_dict['ppo_epsilons']
        self.reward_discount = experiment_dict['reward_discount']
        self.seeds = experiment_dict['seeds']
        self.possible_non_linearities = experiment_dict['non_linearities']
        self.possible_critic_approximator_lrs = experiment_dict['critic_approximator_lrs']
        self.env_id = experiment_dict['env_id']
        self.steps = experiment_dict['steps']
        self.eval_percentage = experiment_dict['eval_percentage']
        self.runs = []
        self.completed_runs = []
        self.run_count = 0
        self.path = root + f'{self.name}/' 
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        if runs is None:
            for intrinsic_dim in self.possible_intrinsic_dims:
                for gradient_decay in self.possible_gradient_decays:
                    for intrinsic_weight in self.possible_intrinsic_weights:
                        for ppo_epsilon in self.possible_ppo_epsilons:
                            for non_linearity in self.possible_non_linearities:
                                for c_a_lr in self.possible_critic_approximator_lrs:
                                    for seed in self.seeds:
                                        agent_dict = {
                                            'input_dim' : self.input_dim,
                                            'intrinsic_dim' : intrinsic_dim,
                                            'action_dim' : self.action_dim,
                                            'policy_lr' : c_a_lr / torch.e,
                                            'approximator_lr' : c_a_lr,
                                            'critic_lr' : c_a_lr,
                                            'gradient_decay' : gradient_decay,
                                            'buffer_size' : self.buffer_size,
                                            'intrinsic_weight' : intrinsic_weight,
                                            'policy_hidden_dim' : self.input_dim,
                                            'policy_non_linearity' : non_linearity,
                                            'r_net_hidden_dim' : self.input_dim * 2,
                                            'r_net_non_linearity' : non_linearity,
                                            'ppo_epsilon' : ppo_epsilon,
                                            'critic_hidden_dim' : self.input_dim,
                                            'critic_non_linearity' : non_linearity,
                                            'reward_discount' : 0.95,
                                            'seed' : seed
                                        }
                                        agent_dict['run_id'] = f'run_{self.run_count}'
                                        self.runs.append(Run(agent_dict, self.env_id, self.steps, self.path, self.eval_percentage))
                                        self.run_count += 1
        else:
            self.runs = runs
            self.run_count = len(self.runs)

    def run(self, verbose=False):
        assert len(self.runs) > 0
        total_runs = len(self.runs)
        while len(self.runs) > 0:
            current_run = self.runs.pop()
            current_run.run(verbose, info=f' | {total_runs - len(self.runs)}/{total_runs}')
            self.completed_runs.append(current_run)
        self.completed_runs.sort(key = lambda x:x.final_avg_extrinsic, reverse=True)
        self.save()

    def save(self):
        # Save expermient_dict
        with open(f'{self.path}{self.name}_experiment_dict.csv', 'w') as f:
            for key in self.experiment_dict:
                f.write("%s, %s\n" % (key, str(self.experiment_dict[key])))
        # Save rankings
        with open(f'{self.path}{self.name}_run_rankings.csv', 'w') as f:
            f.write("%s, %s\n" % ('Run_id', f'Final {self.eval_percentage * 100}% episodes extrinsic score'))
            for run in self.completed_runs:
                f.write("%s, %s\n" % (run.run_id, run.final_avg_extrinsic))

        

class Run:

    def __init__(self, agent_dict, env_id, steps, root, eval_percentage):
        self.agent_dict = agent_dict
        self.run_id = agent_dict['run_id']
        self.seed = agent_dict['seed']
        self.env_id = env_id
        self.steps = steps
        self._complete = False
        self._saved = False
        self.path = root + f'{self.run_id}/' 
        self.eval_percentage = eval_percentage

    def run(self, verbose=False, info=''):
        env = gym.make(self.env_id)
        self.set_seed(self.seed, env)
        agent = LODA(self.agent_dict)
        self.external_scores, self.intrinsic_scores = agent.run_steps(env, verbose, self.steps, info=info)
        self.external_errors = agent.external_errors
        self.intrinsic_errors = agent.intrinsic_errors
        self.external_advantages = agent.external_advantages
        self.intrinsic_advantages = agent.intrinsic_advantages
        self._complete = True
        self.save()

    def save(self):
        assert self._complete and not self._saved
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        
        # Save plots
        plt.plot(self.external_scores)
        plt.ylabel('Total Extrinsic Score')
        plt.xlabel('Episodes')
        plt.savefig(f'{self.path}{self.run_id}_extrinsic_scores.png')
        plt.close()
        plt.plot(self.intrinsic_scores)
        plt.ylabel('Total Intrinsic Score')
        plt.xlabel('Episodes')
        plt.savefig(f'{self.path}{self.run_id}_intrinsic_scores.png')
        plt.close()
        plt.plot(self.external_errors)
        plt.ylabel('Mean Extrinsic Errors')
        plt.xlabel('Optimization Steps')
        plt.savefig(f'{self.path}{self.run_id}_mean_extrinsic_errors.png')
        plt.close()
        plt.plot(self.intrinsic_errors)
        plt.ylabel('Mean Intrinsic Errors')
        plt.xlabel('Optimization Steps')
        plt.savefig(f'{self.path}{self.run_id}_mean_intrinsic_errors.png')
        plt.close()
        plt.plot(self.external_advantages)
        plt.ylabel('Mean Extrinsic Advantages')
        plt.xlabel('Optimization Steps')
        plt.savefig(f'{self.path}{self.run_id}_mean_extrinsic_advantages.png')
        plt.close()
        plt.plot(self.intrinsic_advantages)
        plt.ylabel('Mean Intrinsic Advantages')
        plt.xlabel('Optimization Steps')
        plt.savefig(f'{self.path}{self.run_id}_mean_intrinsic_advantages.png')
        plt.close()

        # Save agent_dict
        with open(f'{self.path}{self.run_id}_agent_dict.csv', 'w') as f:
            for key in self.agent_dict:
                f.write("%s, %s\n" % (key, str(self.agent_dict[key])))
            eval_list = self.external_scores[-int(len(self.external_scores) * self.eval_percentage):]
            self.final_avg_extrinsic = sum(eval_list) / len(eval_list)
            f.write("%s, %s\n" % ('Final_Avg_Extrinsic_Score', self.final_avg_extrinsic))

        self._saved = True

    def complete(self):
        return self._complete

    def set_seed(self, seed, env):
        rand.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        env.seed(seed)

    