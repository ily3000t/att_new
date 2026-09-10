import os
import torch
from amb.agents.base_agent import BaseAgent
from amb.models.actor.ppo_actor import PPOActor

class PPOAgent(BaseAgent):
    def __init__(self, args, obs_space, act_space, device=torch.device("cpu")):
        # save arguments
        self.args = args
        self.device = device

        self.obs_space = obs_space
        self.act_space = act_space

        self.actor = PPOActor(args, self.obs_space, self.act_space, self.device)

    def forward(self, obs, rnn_states, masks, available_actions=None):
        action_dist, rnn_states = self.actor(obs, rnn_states, masks, available_actions)
        
        return action_dist, rnn_states

    @torch.no_grad()
    def sample(self, obs, available_actions=None):
        action_dist = self.actor.sample(obs, available_actions)
        actions = action_dist.sample()

        return actions, action_dist

    @torch.no_grad()
    def perform(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        action_dist, rnn_states = self.actor(obs, rnn_states, masks, available_actions)
        actions = (action_dist.mode if deterministic else action_dist.sample())

        return actions, rnn_states
      
    @torch.no_grad()
    def worst_perform(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        action_dist, rnn_states = self.actor(obs, rnn_states, masks, available_actions)
        # print("here is ppo agent worst_perform")
        # 均为离散的
        if self.actor.action_type == "Discrete":
            action_probs = action_dist.probs
            actions = action_probs.argmin(dim=-1)
            actions = actions.unsqueeze(-1)
        elif self.actor.action_type == "Box":
            means = action_dist.mean
            stds = action_dist.stddev
            actions = means + 3 * stds * torch.sign(torch.randn_like(means))
            actions = torch.clamp(actions, self.actor.low, self.actor.high)

        return actions, rnn_states
    
    def mad_perform(self, obs, rnn_states, masks, available_actions=None, deterministic=False, iterations = 10, alpha=0.1, epsilon=1):
        '''Apply Maximal Action Difference (MAD) performance'''
        action_dist, rnn_states = self.actor(obs, rnn_states, masks, available_actions)
        obs_pert = obs.clone() + torch.randn(obs.shape) * 0.05  # 增加扰动，防止梯度消失
        for _ in range(iterations):
            obs_pert.requires_grad_(True)
            action_dist_pert, rnn_states_pert = self.actor(obs_pert, rnn_states, masks, available_actions)
            action_probs = action_dist.probs
            action_probs_pert = action_dist_pert.probs
            cost = torch.nn.functional.kl_div(action_probs, action_probs_pert, reduction="batchmean")
            cost.backward()
            noise = obs_pert.sgn() * alpha * self.obs_space
            obs_pert = obs + torch.clamp(obs_pert.detach()+ noise - obs, -epsilon * 1, epsilon * 1)

        action_dist, rnn_states = self.actor(obs_pert, rnn_states, masks, available_actions)
        if self.actor.action_type == "Discrete":
            actions = action_dist.mode if deterministic else action_dist.sample()
            actions = actions.argmax(dim=-1, keepdim=True)
        elif self.actor.action_type == "Box":
            actions = action_dist.mean
            
        return actions, rnn_states
    
    def get_action_distribution(self, obs, rnn_states=None, masks=None, available_actions=None):
        """Get Action Distribution"""
        if rnn_states is None:
            rnn_states = torch.zeros(
                (obs.shape[0], self.recurrent_n, self.rnn_hidden_size),
                device=obs.device
            )
        if masks is None:
            masks = torch.ones((obs.shape[0], 1), device=obs.device)

        action_dist, _ = self.actor(
            obs,
            rnn_states,
            masks,
            available_actions
        )
        return action_dist
    
    @torch.no_grad()
    def collect(self, obs, rnn_states, masks, available_actions=None, t=0):
        action_dist, rnn_states = self.actor(obs, rnn_states, masks, available_actions)
        actions = action_dist.sample()
        action_log_probs = action_dist.log_probs(actions)

        return actions, action_log_probs, rnn_states
    
    def restore(self, path):
        self.actor.load_state_dict(torch.load(os.path.join(path, "actor.pth")))

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.actor.state_dict(), os.path.join(path, "actor.pth"))

    def prep_training(self):
        self.actor.train()

    def prep_rollout(self):
        self.actor.eval()
