from agent import Agent


def get_agents(obs_space, act_space, n_agents):
    agent = Agent(obs_space, act_space)
    return [agent for _ in range(n_agents)]