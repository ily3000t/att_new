import numpy as np
import torch


def test_bootstrap_reads_observation_and_hidden_state_after_last_transition(make_cacc_runner):
    runner = make_cacc_runner(episode_length=3, num_env_steps=3, use_popart=False)
    original_step = runner.envs.step
    transition = 0

    def step(actions):
        nonlocal transition
        transition += 1
        data = list(original_step(actions))
        data[1] = np.full_like(data[1], 100 + 10 * transition)
        return tuple(data)

    def critic(obs, states, masks):
        # A known numeric value exposes an off-by-one observation or RNN state.
        values = obs[:, :1] + states[:, 0, :1] + 1000 * masks
        return torch.as_tensor(values), torch.as_tensor(states + 1)

    values_seen = []

    def train(next_values):
        values_seen.append(next_values.copy())
        return [], {}

    runner.envs.step = step
    runner.critic = critic
    runner.train = train
    runner.run()
    # After 3 transitions: observation=130, input hidden state=3, mask=1.
    np.testing.assert_array_equal(values_seen[0], np.full((1, 8, 1), 1133))
