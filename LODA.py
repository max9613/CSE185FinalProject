import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import nn
from torch.distributions import Categorical
"""
    Linear Operater Distillation Agent -- LODA
    - One random network taking an input of the current state, with a single hidden layer (although this could be any static computational graph) and output layer of size K
    - K Linear operators attempting to approximate the correpsonding dimension in the random networks output
    - K Single Hidden Layer policies, actor critic agents trained on external reward and corresponding intrinsic reward (ie squared error of the corresponding linear operator)
    - Behavioral policy logits are a weighted sum of the K parameterized policies, where the logit for the weight of the ith policy is -(squared error of the ith linear operator)
        - these logits are softmaxed to generate the actual weight

"""

def get_non_linearity(key):
    if type(key) is str:
        if key.capitalize() == 'RELU':
            return torch.nn.ReLU()
        else:
            return torch.nn.Tanh()
    else:
        return key

class LinearMultiHeadPolicy(nn.Module):

    def __init__(self, input_dim, action_dim, heads):
        super().__init__()
        self.action_dim = action_dim
        self.heads = heads
        self.matrix = nn.Linear(input_dim, action_dim * heads)

    def forward(self, x):
        x = self.matrix(x)
        return x.view(-1, self.heads, self.action_dim)

class NonLinearMultiHeadPolicy(nn.Module):

    def __init__(self, input_dim, action_dim, heads, hidden_dim, non_linearity):
        super().__init__()
        self.action_dim = action_dim
        self.heads = heads
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, action_dim * heads)
        self.non_linear = non_linearity

    def forward(self, x):
        x = self.non_linear(self.hidden(x))
        x = self.out(x)
        return x.view(-1, self.heads, self.action_dim)

class NonLinearCritic(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, non_linearity):
        super().__init__()
        self.hidden_0 = nn.Linear(input_dim, hidden_dim)
        self.hidden_1 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.non_linear = non_linearity

    def forward(self, x):
        x = self.non_linear(self.hidden_0(x))
        x = self.non_linear(self.hidden_1(x))
        return self.out(x)

class RandomNetwork(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, non_linearity):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim, bias=False)
        self.non_linear = non_linearity

    def forward(self, x):
        x = self.non_linear(self.hidden(x))
        return self.out(x).detach()

class LODA:

    def __init__(self, agent_dict):
        self.agent_dict = agent_dict
        self.input_dim = agent_dict['input_dim']
        self.intrinsic_dim = agent_dict['intrinsic_dim']
        self.action_dim = agent_dict['action_dim']
        self.policy_lr = agent_dict['policy_lr']
        self.gradient_decay = agent_dict['gradient_decay']
        self.buffer_size = agent_dict['buffer_size']
        self.intrinsic_weight = agent_dict['intrinsic_weight']
        self.reward_discount = agent_dict['reward_discount']
        self.filled = False
        self.position = 0
        self.intrinsic_errors = []
        self.external_errors = []
        self.intrinsic_advantages = []
        self.external_advantages = []
        self.optimization_dones = []
        # Buffers 
        self.state_buffer = torch.zeros(self.buffer_size, self.input_dim)
        self.action_buffer = torch.zeros(self.buffer_size)
        self.log_prob_buffer = torch.zeros(self.buffer_size)
        self.external_reward_buffer = torch.zeros(self.buffer_size)
        self.intrinsic_reward_buffer = torch.zeros(self.buffer_size, self.intrinsic_dim)
        self.external_value_buffer = torch.zeros(self.buffer_size)
        self.intrinsic_value_buffer = torch.zeros(self.buffer_size, self.intrinsic_dim)
        self.dones = torch.zeros(self.buffer_size)
        self.traces = torch.zeros(self.buffer_size)
        # Gradient accumulators
        self.hessian = torch.zeros(self.input_dim, self.input_dim)
        self.random_targets = torch.zeros(self.input_dim, self.intrinsic_dim)
        self.signal_targets = torch.zeros(self.input_dim, self.intrinsic_dim + 1)
        self.state_accum = torch.zeros(self.input_dim)
        self.decay_total = torch.zeros(1)
        self.approximators = torch.rand(self.input_dim, self.intrinsic_dim)
        if 'policy_hidden_dim' not in agent_dict or agent_dict['policy_hidden_dim'] is None:
            self.policies = LinearMultiHeadPolicy(self.input_dim, self.action_dim, self.intrinsic_dim)
        else:
            if 'policy_non_linearity' not in agent_dict:
                policy_non_linearity = nn.ReLU()
            else:
                policy_non_linearity = get_non_linearity(agent_dict['policy_non_linearity'])
            self.policies = NonLinearMultiHeadPolicy(self.input_dim, self.action_dim, self.intrinsic_dim, agent_dict['policy_hidden_dim'], policy_non_linearity)
        if 'random_network' not in agent_dict:
            self.random_network=RandomNetwork(self.input_dim, agent_dict['r_net_hidden_dim'], self.intrinsic_dim, get_non_linearity(agent_dict['r_net_non_linearity']))
        else:
            self.random_network = agent_dict['random_network']
        if 'ppo_epsilon' in agent_dict:
            self.ppo_epsilon = agent_dict['ppo_epsilon']
        else:
            self.ppo_epsilon = None
        if 'approximator_lr' in agent_dict and 'critic_lr' in agent_dict:
            self.approximator_lr = agent_dict['approximator_lr']
            self.critic_lr = agent_dict['critic_lr']
        else:
            self.approximator_lr = None
            self.critic_lr = None
        # First self.intrinsic_dim dimensions are intrinsic rewards critics, dim -1 is the external reward critic
        if 'critic_hidden_dim' not in agent_dict or agent_dict['critic_hidden_dim'] is None:
            self.critics = torch.rand(self.input_dim, self.intrinsic_dim + 1)
            self.non_linear_critic = False
        else:
            if 'critic_non_linearity' not in agent_dict:
                critic_non_linearity = nn.ReLU()
            else:
                critic_non_linearity = get_non_linearity(agent_dict['critic_non_linearity'])
            self.intrinsic_critics = NonLinearCritic(self.input_dim, agent_dict['critic_hidden_dim'], self.intrinsic_dim, critic_non_linearity)
            self.external_critic = NonLinearCritic(self.input_dim, agent_dict['critic_hidden_dim'], 1, critic_non_linearity)
            self.non_linear_critic = True
            self.critic_optimizer = optim.Adam(list(self.intrinsic_critics.parameters()) + list(self.external_critic.parameters()), lr=self.critic_lr)
        self.policy_optimizer = optim.Adam(self.policies.parameters(), lr=self.policy_lr, eps=1e-05)

    def approximate(self, x):
        return F.linear(x, self.approximators.transpose(0, 1), torch.zeros(1))

    def critique(self, x):
        if self.non_linear_critic:
            intrinsic = self.intrinsic_critics(x)
            external = self.external_critic(x)
            if len(x.shape) == 1:
                return torch.cat([intrinsic, external], dim=-1)
            else:
                return torch.cat([intrinsic, external.view(-1, 1)], dim=-1)
            
        return F.linear(x, self.critics.transpose(0, 1), torch.zeros(1))

    def get_policy(self, state, squared_errors=None):
        if squared_errors is None:
            truth = self.random_network(state)
            errors = (truth - self.approximate(state)).detach()
            squared_errors = (errors * errors)
        squared_errors = squared_errors.detach() * -1
        offset_errors = squared_errors - torch.max(squared_errors, dim=-1, keepdim=True)[0]
        errors_exp = torch.exp(offset_errors)
        """
        images = torch.sin(self.random_network(state))
        offset_images = images - torch.max(images, dim=-1, keepdim=True)[0]
        errors_exp = torch.exp(offset_images)
        """
        weights = errors_exp / torch.sum(errors_exp, dim=-1, keepdim=True)
        policies = self.policies(state)
        weighted_policy_out = policies * weights.view(-1, self.intrinsic_dim, 1)
        logits = torch.sum(weighted_policy_out, dim=-2)
        offset_logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]
        policy = Categorical(logits = offset_logits)
        return policy

    def get_sub_policies(self, state):
        logits = self.policies(state)
        offset_logits = logits - torch.max(logits, dim=-1, keepdim=True)[0]
        policy = Categorical(logits = offset_logits)
        return policy

    def select_action(self, state):
        policy = self.get_policy(state)
        return policy.sample().item()

    def start(self, state, reset=True):
        if reset:
            # Reset buffers 
            self.state_buffer = torch.zeros(self.buffer_size, self.input_dim)
            self.action_buffer = torch.zeros(self.buffer_size)
            self.log_prob_buffer = torch.zeros(self.buffer_size)
            self.external_reward_buffer = torch.zeros(self.buffer_size)
            self.intrinsic_reward_buffer = torch.zeros(self.buffer_size, self.intrinsic_dim)
            self.external_value_buffer = torch.zeros(self.buffer_size)
            self.intrinsic_value_buffer = torch.zeros(self.buffer_size, self.intrinsic_dim)
            self.dones = torch.zeros(self.buffer_size)
            self.traces = torch.zeros(self.buffer_size)
            # Reset accumulators
            self.hessian = torch.zeros(self.input_dim, self.input_dim)
            self.random_targets = torch.zeros(self.input_dim, self.intrinsic_dim)
            self.signal_targets = torch.zeros(self.input_dim, self.intrinsic_dim + 1)
            self.decay_total = torch.zeros(1)
            self.filled = False
            self.position = 0
            self.intrinsic_errors = []
            self.external_errors = []
            self.intrinsic_advantages = []
            self.external_advantages = []

        self.state_accum = torch.zeros(self.input_dim)
        state = torch.from_numpy(state)
        # Get policy
        policy = self.get_policy(state)
        # Add to buffers and select action
        self.state_buffer[self.position] = state
        self.action_buffer[self.position] = policy.sample().item()
        self.log_prob_buffer[self.position] = policy.log_prob(self.action_buffer[self.position]).detach()
        self.traces[self.position] = 1
        self.external_reward_buffer[self.position] *= 0
        self.intrinsic_reward_buffer[self.position] *= 0
        # Add to accumulators 
        self.hessian = state.outer(state) + (self.gradient_decay * self.hessian)
        self.state_accum = state + (self.gradient_decay * self.state_accum)
        self.decay_total = 1
        # Return action
        return self.action_buffer[self.position].int().item() 

    def step(self, state, reward, done):
        state = torch.from_numpy(state)
        prev_state = self.state_buffer[self.position]
        # Add info for last actions reward to buffers
        self.external_reward_buffer += self.traces * reward
        random_out = self.random_network(state)
        errors = ((random_out - self.approximate(state)) * (random_out - self.approximate(state))).detach()
        self.random_targets = (prev_state.view(-1, 1) * random_out) + (self.gradient_decay * self.random_targets)
        self.signal_targets = (self.state_accum.view(-1, 1) * torch.cat([errors, torch.ones(1) * reward], axis=-1)) + (self.gradient_decay * self.signal_targets)
        self.decay_total = 1 + (self.gradient_decay * self.decay_total)
        self.intrinsic_reward_buffer += self.traces.view(-1, 1) * errors
        self.traces *= self.reward_discount
        # Optimize Approximator:
        if self.approximator_lr is not None:
            approximator_gradient = (F.linear(self.hessian, self.approximators.transpose(0,1), torch.zeros(1)) - self.random_targets) / self.decay_total
            self.approximators -= self.approximator_lr * approximator_gradient
            if not self.non_linear_critic:
                critic_gradient = (F.linear(self.hessian, self.critics.transpose(0,1), torch.zeros(1)) - self.signal_targets) / self.decay_total
                self.critics -= self.critic_lr * critic_gradient
        prev_position = self.position
        self.position += 1
        if self.position >= self.buffer_size:
            self.position = 0
            self.filled = True
        if self.filled:
            self.update(state, done)
        
        self.external_reward_buffer[self.position] *= 0
        self.intrinsic_reward_buffer[self.position] *= 0
        # Add to accumulators 
        self.hessian = state.outer(state) + (self.gradient_decay * self.hessian)
        self.state_accum = state + (self.gradient_decay * self.state_accum)
        # Get policy
        policy = self.get_policy(state, self.intrinsic_reward_buffer[prev_position])
        # Add to buffers and select action
        self.state_buffer[self.position] = state
        self.action_buffer[self.position] = policy.sample().item()
        self.log_prob_buffer[self.position] = policy.log_prob(self.action_buffer[self.position]).detach()
        if done:
            self.dones[self.position] = 1
            self.traces = torch.zeros_like(self.traces)
        else:
            self.dones[self.position] = 0
        # Return action
        return self.action_buffer[self.position].int().item(), torch.sum(self.intrinsic_reward_buffer[prev_position], axis=-1)

    def update(self, current_state, done):
        assert self.filled
        if self.non_linear_critic:
            critic_vals = self.critique(self.state_buffer)
            intrinsic_vals = self.intrinsic_value_buffer + 0
            extrenal_vals = self.external_value_buffer + 0
            if not done:
                next_val = self.critique(current_state).detach()
                bootstrap_val = next_val * self.traces.view(-1, 1) * self.reward_discount
                intrinsic_vals += bootstrap_val.transpose(0,1)[:-1].transpose(0,1)
                extrenal_vals += bootstrap_val.transpose(0,1)[-1]
            intrinsic_advantages = intrinsic_vals - critic_vals.transpose(0,1)[:-1].transpose(0,1)
            external_advantages = extrenal_vals - critic_vals.transpose(0,1)[-1]
            total_advantages = intrinsic_advantages.sum(dim=-1) + external_advantages
            critic_loss = (total_advantages * total_advantages).mean() / 2
            critic_loss.backward()
            self.critic_optimizer.step()
        critic_vals = self.critique(self.state_buffer).detach()
        if not done:
            next_val = self.critique(current_state).detach()
            critic_vals += next_val * self.traces.view(-1, 1)
        intrinsic_advantages = self.intrinsic_value_buffer - critic_vals.transpose(0,1)[:-1].transpose(0,1)
        self.intrinsic_advantages.append((intrinsic_advantages).mean())
        self.intrinsic_errors.append((intrinsic_advantages * intrinsic_advantages).mean())
        external_advantages = self.external_value_buffer - critic_vals.transpose(0,1)[-1]
        self.external_advantages.append((external_advantages).mean())
        self.external_errors.append((external_advantages * external_advantages).mean())
        self.optimization_dones.append(1 if done else 0)
        #total_advantages = (intrinsic_advantages * self.intrinsic_weight) + external_advantages.view(-1, 1)
        # normalize advantages
        normalized_external = (external_advantages - external_advantages.mean()) / (external_advantages.std() + 1e-7)
        normalized_intrinsic = (intrinsic_advantages - intrinsic_advantages.mean()) / (intrinsic_advantages.std() + 1e-7)
        normalized_advantages = (normalized_intrinsic * self.intrinsic_weight) + normalized_external.view(-1, 1) #(total_advantages - total_advantages.mean()) / (total_advantages.std() + 1e-7)
        policies = self.get_sub_policies(self.state_buffer)
        log_probs = policies.log_prob(self.action_buffer.view(-1, 1))
        ratios = torch.exp(self.log_prob_buffer.view(-1, 1) - log_probs)
        if self.ppo_epsilon is None:
            loss = -(normalized_advantages * ratios).mean()
        else:
            # Surrogate Loss
            surr1 = ratios * normalized_advantages
            surr2 = torch.clamp(ratios, 1-self.ppo_epsilon, 1+self.ppo_epsilon) * normalized_advantages
            loss = -torch.min(surr1, surr2).mean()
        
        self.policy_optimizer.zero_grad()
        loss.backward()
        self.policy_optimizer.step()


    def run_steps(self, env, verbose, steps, info='', render=False):
        done = False
        step = 0
        external_score = 0
        intrinsic_score = 0
        external_history = []
        intrinsic_history = []
        action = self.start(env.reset())
        while step < steps:
            step += 1
            observation, reward, done, _ = env.step(action)
            if render:
                env.render()
            action, intrinsic_reward = self.step(observation, reward, done)
            external_score += reward
            intrinsic_score += intrinsic_reward
            if verbose:
                print('\rSTEP {}/{}{}\t\t'.format(step, steps, info), end = '')
            if done:
                external_history.append(external_score)
                intrinsic_history.append(intrinsic_score)
                external_score = 0
                intrinsic_score = 0
                action = self.start(env.reset(), reset=False)
        return external_history, intrinsic_history


    