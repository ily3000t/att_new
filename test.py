import numpy as np
from amb.envs.smac.StarCraft2Dual_Env import StarCraft2DualEnv

args = {
    "state_type": "FP",
    "map_name": "8m_dual",
}

if __name__ == "__main__":
    env = StarCraft2DualEnv(args, window_size_x=1280, window_size_y=720, seed=0)
    print(env.observation_space, env.share_observation_space, env.action_space, env.n_agents)
    env.reset()

    input()

    print(env.step(np.ones((2, 8, 1))))

    input()

    env.close()