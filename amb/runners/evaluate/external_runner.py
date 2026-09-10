import os
import sys
import torch
import numpy as np
import setproctitle
from amb.algorithms.igs import IGS
from amb.envs import LOGGER_REGISTRY
from amb.utils.trans_utils import _t2n
from amb.utils.env_utils import (
    make_eval_env,
    set_seed,
)
from amb.utils.model_utils import init_device
from amb.utils.config_utils import init_dir, save_config, get_task_name
from amb.utils.trans_utils import _t2n, gather, scatter
import importlib


class ExternalRunner:
    def __init__(self, args, algo_args, env_args):
        """Initialize the single/BaseRunner class.
        Args:
            args: command-line arguments parsed by argparse. Three keys: algo, env, exp_name.
            algo_args: arguments related to algo, loaded from config file and updated with unparsed command-line arguments.
            env_args: arguments related to env, loaded from config file and updated with unparsed command-line arguments.
        """
        self.args = args
        self.algo_args = algo_args
        self.env_args = env_args

        self.rnn_hidden_size = algo_args["train"]["hidden_sizes"][-1]
        self.recurrent_n = algo_args["train"]["recurrent_n"]

        self.victim_rnn_hidden_size = algo_args["victim"]["hidden_sizes"][-1]
        self.victim_recurrent_n = algo_args["victim"]["recurrent_n"]
        self.num_adv_agents = len(algo_args["train"]["adv_agent_ids"])

        self.n_eval_rollout_threads = algo_args["train"]["n_eval_rollout_threads"]
        self.share_param = algo_args["train"]["share_param"]

        set_seed(algo_args["train"])
        self.device = init_device(algo_args["train"])
        self.task_name = get_task_name(args["env"], env_args)
        self.run_dir, self.log_dir, self.save_dir, self.writter = init_dir(
            args["env"],
            env_args,
            args["algo"],
            args["exp_name"],
            args["run"],
            algo_args["train"]["seed"],
            logger_path=algo_args["train"]["log_dir"],
        )
        save_config(args, algo_args, env_args, self.run_dir)

        setproctitle.setproctitle(str(args["algo"]) + "-" + str(args["env"]) + "-" + str(args["exp_name"]))

        # set the config of env
        self.envs = make_eval_env(
            args["env"],
            algo_args["train"]["seed"],
            algo_args["train"]["n_eval_rollout_threads"],
            env_args,
        )
        self.num_agents = self.envs.n_agents
        self.action_type = self.envs.action_space[0].__class__.__name__

        self.logger = LOGGER_REGISTRY[args["env"]](
            args, algo_args, env_args, self.num_agents, self.writter, self.run_dir
        )

        # algorithm
        if self.share_param:
            for agent_id in range(1, self.num_agents):
                assert (
                    self.envs.observation_space[agent_id] == self.envs.observation_space[0]
                ), "Agents have heterogeneous observation spaces, parameter sharing is not valid."
                assert (
                    self.envs.action_space[agent_id] == self.envs.action_space[0]
                ), "Agents have heterogeneous action spaces, parameter sharing is not valid."

        # 将路径转为模块名
        module_path = os.path.abspath(args["load_external"])
        # 输出文件所在目录的绝对路径
        module_dir = os.path.dirname(module_path)
        sys.path.append(module_dir)
        # 输出文件的文件名
        module_name = os.path.basename(module_path).replace(".py", "")

        module = importlib.import_module(module_name)
        get_agents = getattr(module, "get_agents")
        self.agents = get_agents(self.envs.observation_space[0], self.envs.action_space[0], self.num_agents)

        # random noise attack
        self.attack = IGS(algo_args["train"], self.envs.action_space[0], self.device)
        self.perturb_timesteps = np.array(algo_args["train"]["perturb_timesteps"])

    def run(self):
        self.logger.init()
        self.logger.episode_init(0)
        self.eval()
        if self.args["attack"] == "random_noise":
            self.eval_random_noise()
        elif self.args["attack"] == "iterative_perturbation":
            self.eval_iterative_perturbation()
        elif self.args["attack"] == "adaptive_action":
            self.eval_adaptive_action()

    @torch.no_grad()
    def eval(self):
        print("===> Evaluate Agents Vanilla:")
        """Evaluate the model. All algorithms should fit this evaluation pipeline."""
        for agent in self.agents:
            agent.prep_rollout()

        # logger callback at the beginning of evaluation
        self.logger.eval_init(self.n_eval_rollout_threads)
        eval_episode = 0

        eval_obs, eval_share_obs, eval_available_actions = self.envs.reset()

        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, self.num_agents, self.recurrent_n, self.rnn_hidden_size),
            dtype=np.float32,
        )
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            eval_actions_collector = []
            for agent_id in range(self.num_agents):
                eval_actions, temp_rnn_state = self.agents[agent_id].perform(
                    eval_obs[:, agent_id],
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    eval_available_actions[:, agent_id] if eval_available_actions[0] is not None else None,
                    deterministic=True,
                )
                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)

            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.envs.step(eval_actions)

            eval_data = (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            )
            # logger callback at each step of evaluation
            self.logger.eval_per_step(eval_data)

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = 0

            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = 0

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    # logger callback when an episode is done
                    self.logger.eval_thread_done(eval_i)

            if eval_episode >= self.algo_args["train"]["eval_episodes"]:
                self.logger.eval_log(eval_episode)  # logger callback at the end of evaluation
                break

    def eval_perturbation(self):
        """Evaluate the model. All algorithms should fit this evaluation pipeline."""
        for agent in self.agents:
            agent.prep_rollout()

        self.logger.eval_init(self.n_eval_rollout_threads)  # logger callback at the beginning of evaluation
        eval_episode = 0

        eval_obs, eval_share_obs, eval_available_actions = self.envs.reset()

        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, self.num_agents, self.victim_recurrent_n, self.victim_rnn_hidden_size),
            dtype=np.float32,
        )
        eval_adv_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, self.num_adv_agents, self.recurrent_n, self.rnn_hidden_size), dtype=np.float32
        )
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        # shape: [n_threads, n_agents]
        adv_agent_ids = np.stack([self.get_certain_adv_ids() for _ in range(self.n_eval_rollout_threads)], axis=0)
        current_timesteps = np.zeros((self.n_eval_rollout_threads,), dtype=np.int32)

        while True:
            # 1. Get actions before attack (do not update rnn hidden states)
            eval_actions_collector = []
            for agent_id in range(self.num_agents):
                eval_actions, _ = self.agents[agent_id].perform(
                    eval_obs[:, agent_id],
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    eval_available_actions[:, agent_id] if eval_available_actions[0] is not None else None,
                    deterministic=True,
                )
                eval_actions_collector.append(_t2n(eval_actions))
            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)

            # 2. If targeted attack, generate adversarial actions
            eval_adv_actions = None
            if self.algo_args["train"]["targeted_attack"]:
                eval_adv_actions_collector = []
                for agent_id in range(self.num_adv_agents):
                    eval_adv_actions, temp_adv_rnn_state = self.agents[agent_id].perform(
                        gather(eval_obs, adv_agent_ids, axis=1)[:, agent_id],
                        eval_adv_rnn_states[:, agent_id],
                        gather(eval_masks, adv_agent_ids, axis=1)[:, agent_id],
                        gather(eval_available_actions, adv_agent_ids, axis=1)[:, agent_id]
                        if eval_available_actions[0] is not None
                        else None,
                        deterministic=True,
                    )
                    eval_adv_rnn_states[:, agent_id] = _t2n(temp_adv_rnn_state)
                    eval_adv_actions_collector.append(_t2n(eval_adv_actions))
                _eval_adv_actions = np.array(eval_adv_actions_collector).transpose(1, 0, 2)
                eval_adv_actions = eval_actions.copy()
                eval_adv_actions = scatter(eval_adv_actions, adv_agent_ids, _eval_adv_actions, axis=1)

            # 3. Perform attack to perturb the observations
            eval_obs_adv_collector = []
            for agent_id in range(self.num_agents):
                eval_obs_adv = self.attack.perturb(
                    self.agents[agent_id],
                    eval_obs[:, agent_id],
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    eval_available_actions[:, agent_id] if eval_available_actions[0] is not None else None,
                    eval_adv_actions[:, agent_id] if eval_adv_actions is not None else None,
                )
                eval_obs_adv_collector.append(_t2n(eval_obs_adv))
            _eval_obs_adv = np.stack(eval_obs_adv_collector, axis=1)

            perturb_mask = self.perturb_timesteps[current_timesteps]
            _eval_obs_adv = gather(_eval_obs_adv, adv_agent_ids, axis=1)
            eval_obs_adv = eval_obs.copy()
            eval_obs_adv[perturb_mask] = scatter(
                eval_obs_adv[perturb_mask], adv_agent_ids[perturb_mask], _eval_obs_adv[perturb_mask], axis=1
            )

            # 4. Get actions after attack (update rnn hidden states)
            eval_actions_collector = []
            for agent_id in range(self.num_agents):
                eval_actions, temp_rnn_state = self.agents[agent_id].perform(
                    eval_obs_adv[:, agent_id],
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    eval_available_actions[:, agent_id] if eval_available_actions[0] is not None else None,
                    deterministic=True,
                )
                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))
            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)

            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.envs.step(
                eval_actions
            )
            eval_data = (eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions)

            self.logger.eval_per_step(eval_data)  # logger callback at each step of evaluation

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = 0

            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = 0

            current_timesteps += 1
            current_timesteps[eval_dones_env == True] = 0

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    self.logger.eval_thread_done(eval_i)  # logger callback when an episode is done
                    adv_agent_ids[eval_i] = self.get_certain_adv_ids()

            if eval_episode >= self.algo_args["train"]["eval_episodes"]:
                self.logger.eval_log_adv(eval_episode)  # logger callback at the end of evaluation
                break


    def eval_random_noise(self):
        print("===> Evaluate Agents under Random Noise Attack:")
        self.eval_perturbation()

    def eval_iterative_perturbation(self):
        print("===> Evaluate Agents under Iterative Perturbation Attack:")
        self.eval_perturbation()

    def eval_adaptive_action(self):
        print("===> Evaluate Agents under Adaptive Action Attack:")
        self.eval_perturbation()


    def close(self):
        """Close environment, writter, and log file."""
        self.envs.close()
        if self.algo_args["train"]["use_eval"] and self.envs is not self.envs:
            self.envs.close()
        self.writter.close()
        self.logger.close()


    def get_certain_adv_ids(self):
        adv_agent_ids = self.algo_args["train"]["adv_agent_ids"]
        n_random = len([t for t in adv_agent_ids if t < 0])
        random_ids = [i for i in range(self.num_agents) if i not in adv_agent_ids]
        confirm_ids = [t for t in adv_agent_ids if t >= 0]
        # the order of adv ids is important, and thus cannot be converted to masks!
        adv_ids = np.array(confirm_ids + np.random.choice(random_ids, n_random, replace=False).tolist(), dtype=np.int32)

        return adv_ids