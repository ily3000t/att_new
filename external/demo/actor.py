import torch
import torch.nn as nn
from torch.distributions import Categorical, Uniform
from base.cnn import CNNLayer
from base.mlp import MLPBase
from base.act import ACTLayer
from utils.env_utils import check, get_shape_from_obs_space, get_onehot_shape_from_act_space


class RayActor(nn.Module):
    def __init__(self, obs_space, action_space, device=torch.device("cpu")):
        super(RayActor, self).__init__()
        args = None
        self.gain = 0.01
        self.hidden_sizes = [128, 128]
        self.initialization_method = "orthogonal_"
        self.activation_func = "relu"

        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        self.act_shape = get_onehot_shape_from_act_space(action_space)

        if len(obs_shape) == 3:
            self.cnn = CNNLayer(obs_shape, self.hidden_sizes, self.initialization_method, self.activation_func)
            input_dim = self.cnn.output_size
        else:
            self.cnn = nn.Identity()
            input_dim = obs_shape[0]

        self.base = MLPBase(args, input_dim)

        self.act = ACTLayer(
            action_space,
            self.hidden_sizes[-1],
            self.initialization_method,
            self.gain,
            args,
        )

        self.action_type = action_space.__class__.__name__
        if self.action_type == "Box":
            self.low = torch.tensor(action_space.low).to(**self.tpdv)
            self.high = torch.tensor(action_space.high).to(**self.tpdv)

        self.to(device)

    def sample(self, obs, available_actions=None):
        obs = check(obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if self.action_type == "Box":
            actor_out = torch.ones((obs.shape[0], self.act_shape)).to(**self.tpdv)
            action_dist = Uniform(actor_out * self.low, actor_out * self.high)
        elif self.action_type == "Discrete" and available_actions is not None:
            actor_out = torch.ones((obs.shape[0], self.act_shape)).to(**self.tpdv)
            actor_out[available_actions == 0] = -1e10
            action_dist = Categorical(logits=actor_out)

        return action_dist

    def forward(self, obs, rnn_states, masks, available_actions=None):
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv) if rnn_states is not None else None
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(self.cnn(obs))

        action_dist = self.act(actor_features, available_actions)

        return action_dist, rnn_states
