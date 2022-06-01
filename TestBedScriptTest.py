from Experiment import Experiment

experiment_dict = {
            'name' : 'cartExperiment4Test',
            'intrinsic_dims' : [4, 16],
            'gradient_decays' : [0.9, 0.5],
            'buffer_size' : 256,
            'intrinsic_weights' : [0.001],
            'ppo_epsilons' : [None],
            'reward_discount' : 0.95,  
            'seeds' : [420,42],
            'non_linearities' : ['tanh'],
            'critic_approximator_lrs' : [3e-3, 1e-3],
            'env_id' : 'CartPole-v0',
            'steps' : 512,
            'eval_percentage' : 0.1
        }

experiment = Experiment(experiment_dict, '')

experiment.run(True)