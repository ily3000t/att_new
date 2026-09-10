import os
import random
import time
import torch
import numpy as np
import setproctitle
from amb.algorithms import ALGO_REGISTRY
from amb.envs import LOGGER_REGISTRY
from amb.utils.trans_utils import _t2n, gather, scatter
from amb.utils.env_utils import (
    make_eval_env,
    make_train_env,
    make_render_env,
    set_seed,
)
from amb.utils.model_utils import init_device
from amb.utils.config_utils import init_dir, save_config, get_task_name


class BaseRunner:
    def __init__(self, args, algo_args, env_args):
        """Initialize the traitor/BaseRunner class.
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
        
        self.episode_length = algo_args["train"]["episode_length"]
        self.n_rollout_threads = algo_args["train"]["n_rollout_threads"]
        self.n_eval_rollout_threads = algo_args['train']['n_eval_rollout_threads']
        self.perturb_timesteps = np.array(algo_args["train"]["perturb_timesteps"])

        self.share_param = algo_args["train"]['share_param']
        self.victim_share_param = algo_args["victim"]['share_param']

        set_seed(algo_args["train"])
        self.device = init_device(algo_args["train"])
        self.task_name = get_task_name(args["env"], env_args)
        if not self.algo_args['train']['use_render']:
            self.run_dir, self.log_dir, self.save_dir, self.writter = init_dir(
                args["env"],
                env_args,
                args["algo"] + "-" + args["victim"],
                args["exp_name"],
                args["run"],
                algo_args["train"]["seed"],
                logger_path=algo_args["train"]["log_dir"],
            )
            save_config(args, algo_args, env_args, self.run_dir)

        setproctitle.setproctitle(
            str(args["algo"]) + "-" + str(args["env"]) + "-" + str(args["exp_name"])
        )

        # set the config of env
        if self.algo_args['train']['use_render']:  # make envs for rendering
            (
                self.envs,
                self.manual_render,
                self.manual_delay,
                self.env_num,
            ) = make_render_env(args["env"], algo_args["train"]["seed"], env_args)
        else:  # make envs for training and evaluation
            self.eval_envs = (
                make_eval_env(
                    args["env"],
                    algo_args["train"]["seed"],
                    algo_args["train"]["n_eval_rollout_threads"],
                    env_args,
                )
                if algo_args["train"]["use_eval"]
                else None
            )
            # 使用num_env_steps判断是否需要训练，如果不需要训练，则不创建训练环境，避免类似dexhands、network等环境无法创建多个环境实例的问题
            if args["env"] == "dexhands" and self.algo_args['train']["use_eval"] is True:
                # TODO 这种方法只能解决一部分问题，类似adaptive action这种需要训练的攻击方法还是无法使用
                self.envs = self.eval_envs
            else:
                self.envs = make_train_env(
                    args["env"],
                    algo_args["train"]["seed"],
                    algo_args["train"]["n_rollout_threads"],
                    env_args,
                )
        self.num_agents = self.envs.n_agents
        self.num_adv_agents = self.num_agents if algo_args["train"]["adv_all"] \
                                  else len(algo_args["train"]["adv_agent_ids"])
        self.action_type = self.envs.action_space[0].__class__.__name__

        print("share_observation_space: ", self.envs.share_observation_space)
        print("observation_space: ", self.envs.observation_space)
        print("action_space: ", self.envs.action_space, self.action_type)

        # algorithm
        if self.share_param:
            for agent_id in range(1, self.num_agents):
                assert (
                    self.envs.observation_space[agent_id] == self.envs.observation_space[0]
                ), "Agents have heterogeneous observation spaces, parameter sharing is not valid."
                assert (
                    self.envs.action_space[agent_id] == self.envs.action_space[0]
                ), "Agents have heterogeneous action spaces, parameter sharing is not valid."

        self.victims = []
        if self.victim_share_param:
            agent = ALGO_REGISTRY[args["victim"]].create_agent(
                algo_args["victim"],
                self.envs.observation_space[0],
                self.envs.action_space[0],
                device=self.device,
            )
            agent.prep_rollout()
            for agent_id in range(self.num_agents):
                self.victims.append(agent)
        else:
            for agent_id in range(self.num_agents):
                agent = ALGO_REGISTRY[args["victim"]].create_agent(
                    algo_args["victim"],
                    self.envs.observation_space[agent_id],
                    self.envs.action_space[agent_id],
                    device=self.device,
                )
                agent.prep_rollout()
                self.victims.append(agent)

        # cannot support heterogeneous spaces in random-id adv policy
        self.algo = ALGO_REGISTRY[args["algo"]](
            algo_args["train"],
            self.num_adv_agents,
            self.envs.observation_space[:self.num_adv_agents],
            self.envs.share_observation_space[0],
            self.envs.action_space[:self.num_adv_agents],
            device=self.device,
        )

        self.agents = self.algo.agents
        self.critic = self.algo.critic

        if self.algo_args['train']['use_render'] is False:
            self.logger = LOGGER_REGISTRY[args["env"]](
                args, algo_args, env_args, self.num_adv_agents, self.writter, self.run_dir
            )

    def get_certain_adv_ids(self):
        if self.algo_args["train"]["adv_all"]:
            adv_agent_ids = [t for t in range(self.num_agents)]
        else:
            adv_agent_ids = self.algo_args["train"]["adv_agent_ids"]
        n_random = len([t for t in adv_agent_ids if t < 0])
        random_ids = [i for i in range(self.num_agents) if i not in adv_agent_ids]
        confirm_ids = [t for t in adv_agent_ids if t >= 0]
        # the order of adv ids is important, and thus cannot be converted to masks!
        adv_ids = np.array(confirm_ids + np.random.choice(random_ids, n_random, replace=False).tolist(), dtype=np.int32)

        return adv_ids
    
    def get_adv_rewards(self, data):
        if self.args["env"] == "mamujoco":
            if "reward_run" not in data["infos"][0][0]:
                rewards = np.array([data["infos"][i][0]["reward_forward"] for i in range(len(data["infos"]))])
                rewards = rewards.reshape(data["rewards"].shape)
            else:
                rewards = np.array([data["infos"][i][0]["reward_run"] for i in range(len(data["infos"]))])
                rewards = rewards.reshape(data["rewards"].shape)
        else:
            rewards = data["rewards"]
        return -rewards

    def run(self):
        raise NotImplementedError

    # 不攻击的话就调成0
    def whether_traitor_eps(self):
        eps = self.algo_args["train"]["traitor_eps"]
        random_number = random.random()
        return random_number < eps
    
    @torch.no_grad()
    def eval(self):
        """Evaluate the model. All algorithms should fit this evaluation pipeline."""
        print(50 * "-")
        print("n_eval_rollout_threads: ", self.n_eval_rollout_threads)
        print("eval_episodes: ", self.algo_args["train"]["eval_episodes"])
        print("eval_recover_mode: ", self.algo_args["extra"]["eval_recover_mode"] if "eval_recover_mode" in self.algo_args["extra"] else False)
        print("recover_episode: ", self.algo_args["extra"]["recover_episode"] if "recover_episode" in self.algo_args["extra"] else int(self.algo_args["train"]["eval_episodes"] / 2))
        print(50 * "-")

        self.algo.prep_rollout()

        self.logger.eval_init(self.n_eval_rollout_threads)  # logger callback at the beginning of evaluation
        eval_episode = 0

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.victim_recurrent_n, self.victim_rnn_hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            eval_actions_collector = []
            for agent_id in range(self.num_agents):
                eps_perform = self.victims[agent_id].perform
                # if self.whether_traitor_eps():
                #     eps_perform = self.victims[agent_id].worst_perform
                eval_actions, temp_rnn_state = eps_perform(
                    eval_obs[:, agent_id],
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    eval_available_actions[:, agent_id]
                    if eval_available_actions[0] is not None else None,
                    deterministic=True,
                )
                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)

            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)

            eval_data = (eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions)

            self.logger.eval_per_step(eval_data)  # logger callback at each step of evaluation

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = 0

            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = 0

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    self.logger.eval_thread_done(eval_i)  # logger callback when an episode is done

            if eval_episode >= self.algo_args["train"]["eval_episodes"]:
                self.logger.eval_log(eval_episode)  # logger callback at the end of evaluation
                break

    @torch.no_grad()
    def eval_adv(self):
        """Evaluate the model. All algorithms should fit this evaluation pipeline."""
        self.algo.prep_rollout()

        self.logger.eval_init(self.n_eval_rollout_threads)  # logger callback at the beginning of evaluation
        eval_episode = 0

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.victim_recurrent_n, self.victim_rnn_hidden_size), dtype=np.float32)
        eval_adv_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_adv_agents, self.recurrent_n, self.rnn_hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
        
        # shape: [n_threads, n_agents]
        adv_agent_ids = np.stack([self.get_certain_adv_ids() for _ in range(self.n_eval_rollout_threads)], axis=0)
        current_timesteps = np.zeros((self.n_eval_rollout_threads,), dtype=np.int32)

        while True:
            eval_actions_collector = []
            for agent_id in range(self.num_agents):
                eps_perform = self.victims[agent_id].perform
                if self.whether_traitor_eps():
                    eps_perform = self.victims[agent_id].worst_perform
                    # print(f"use worst perform {agent_id}")
                else:
                    # print("not use worst")
                    pass
                eval_actions, temp_rnn_state = eps_perform(
                    eval_obs[:, agent_id],
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    eval_available_actions[:, agent_id]
                    if eval_available_actions[0] is not None else None,
                    deterministic=True,
                )
                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))
            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)

            # if execute_recover, then use the victim policy to inference without adversarial policy
            execute_recover = False
            if "eval_recover_mode" in self.algo_args["extra"] and self.algo_args["extra"]["eval_recover_mode"]:
                recover_episode = self.algo_args["extra"]["recover_episode"] if "recover_episode" in self.algo_args["extra"] else int(self.algo_args["train"]["eval_episodes"] / 2)
                execute_recover = eval_episode >= recover_episode
                if execute_recover:
                    self.logger.eval_log_recovery_timestep()

            if not execute_recover:
                eval_adv_actions_collector = []
                for agent_id in range(self.num_adv_agents):
                    eval_adv_actions, temp_adv_rnn_state = self.agents[agent_id].perform(
                        gather(eval_obs, adv_agent_ids, axis=1)[:, agent_id],
                        eval_adv_rnn_states[:, agent_id],
                        gather(eval_masks, adv_agent_ids, axis=1)[:, agent_id],
                        gather(eval_available_actions, adv_agent_ids, axis=1)[:, agent_id]
                        if eval_available_actions[0] is not None else None,
                        deterministic=True,
                    )
                    eval_adv_rnn_states[:, agent_id] = _t2n(temp_adv_rnn_state)
                    eval_adv_actions_collector.append(_t2n(eval_adv_actions))
                eval_adv_actions = np.array(eval_adv_actions_collector).transpose(1, 0, 2)

                perturb_mask = self.perturb_timesteps[current_timesteps]
                eval_actions[perturb_mask] = scatter(eval_actions[perturb_mask], adv_agent_ids[perturb_mask], eval_adv_actions[perturb_mask], axis=1)

                R_pi_values = [0.0 for _ in range(self.num_agents)]  # 保存每个 agent 的 R_π 值
                adv_agents = [adv_agent_ids[0, t] for t in range(self.num_adv_agents)]
                eps = 1e-8
                adv_agent_count = eval_adv_actions.shape[1]
                for agent_id in range(self.num_agents):
                    if agent_id < adv_agent_count:  # 仅对扰动过的 agent 计算 KL 散度
                        # 取出原始动作分布
                        original_dist = eval_actions[:, agent_id, :]  
                        adv_dist = eval_adv_actions[:, agent_id, :]  

                        # 避免概率分布中有 0，影响 log 计算
                        original_dist = torch.clamp(torch.from_numpy(original_dist), min=eps)
                        adv_dist = torch.clamp(torch.from_numpy(adv_dist), min=eps)

                        # 计算 KL 散度
                        kl_div_value = torch.nn.functional.kl_div(torch.log(adv_dist), original_dist, reduction='batchmean')/(eval_actions.shape[0])
                        R_pi_values[agent_id] = kl_div_value.item()

                R_pi_dict = {f"agent_{agent_id}": R_pi_values[agent_id] for agent_id in range(self.num_agents)}
                self.logger.eval_log_Rpi(R_pi_dict)

            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
            eval_data = (eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions)
            
            self.logger.eval_per_step(eval_data, "eval_adv_per_step_rewards")  # logger callback at each step of evaluation

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

    def eval_slice(self):
        # 判断os.path.dirname(self.algo_args['victim']['model_dir'])下是否有slice文件夹，如果没有则return
        slice_path = os.path.join(os.path.dirname(self.algo_args['victim']['model_dir']), 'slice')
        if not os.path.exists(slice_path) or len(os.listdir(slice_path)) == 0:
            return
        pth_dir_list = os.listdir(slice_path)
        # 将pth_dir_list中的文件夹按照数字大小排序
        pth_dir_list.sort(key=lambda x: int(x))
        # 遍历self.algo_args['victim']['model_dir']目录下的所有文件夹
        for folder_name in pth_dir_list:
            victim_model_dir = os.path.join(os.path.dirname(self.algo_args['victim']['model_dir']), 'slice', folder_name)
            # restore slice victim
            if os.path.isdir(victim_model_dir):
                print("Restore victim from", victim_model_dir)
                if self.victim_share_param:
                    self.victims[0].restore(victim_model_dir)
                else:
                    for agent_id in range(self.num_agents):
                        self.victims[agent_id].restore(os.path.join(victim_model_dir, str(agent_id)))
                
                self.eval()
                self.logger.eval_log_slice("vanilla", folder_name)
                self.algo_args["extra"]["eval_recover_mode"] = False
                self.eval_adv()
                self.logger.eval_log_slice("adv", folder_name)
                self.algo_args["extra"]["eval_recover_mode"] = True
                self.eval_adv()
                self.logger.eval_log_slice("resilience", folder_name)

    @torch.no_grad()
    def render(self):
        """Render the model"""
        print("start rendering")
        self.algo.prep_rollout()

        for _ in range(self.algo_args['train']['render_episodes']):
            eval_rnn_states = np.zeros((self.env_num, self.num_agents, self.victim_recurrent_n, self.victim_rnn_hidden_size), dtype=np.float32)
            eval_adv_rnn_states = np.zeros((self.env_num, self.num_adv_agents, self.recurrent_n, self.rnn_hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.env_num, self.num_agents, 1), dtype=np.float32)

            adv_agent_ids = np.stack([self.get_certain_adv_ids() for _ in range(self.env_num)], axis=0)
            current_timesteps = np.zeros((self.env_num,), dtype=np.int32)

            eval_obs, _, eval_available_actions = self.envs.reset()
            rewards = 0
            while True:
                eval_obs = np.expand_dims(np.array(eval_obs), axis=0)
                if eval_available_actions is not None:
                    eval_available_actions = np.expand_dims(np.array(eval_available_actions), axis=0)
                    
                eval_actions_collector = []
                for agent_id in range(self.num_agents):
                    eval_actions, temp_rnn_state = self.victims[agent_id].perform(
                        eval_obs[:, agent_id],
                        eval_rnn_states[:, agent_id],
                        eval_masks[:, agent_id],
                        eval_available_actions[:, agent_id]
                        if eval_available_actions is not None else None,
                        deterministic=True,
                    )
                    eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                    eval_actions_collector.append(_t2n(eval_actions))
                eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)

                eval_adv_actions_collector = []
                for agent_id in range(self.num_adv_agents):
                    eval_adv_actions, temp_adv_rnn_state = self.agents[agent_id].perform(
                        gather(eval_obs, adv_agent_ids, axis=1)[:, agent_id],
                        eval_adv_rnn_states[:, agent_id],
                        gather(eval_masks, adv_agent_ids, axis=1)[:, agent_id],
                        gather(eval_available_actions, adv_agent_ids, axis=1)[:, agent_id]
                        if eval_available_actions is not None else None,
                        deterministic=True,
                    )
                    eval_adv_rnn_states[:, agent_id] = _t2n(temp_adv_rnn_state)
                    eval_adv_actions_collector.append(_t2n(eval_adv_actions))
                eval_adv_actions = np.array(eval_adv_actions_collector).transpose(1, 0, 2)

                perturb_mask = self.perturb_timesteps[current_timesteps]
                eval_actions[perturb_mask] = scatter(eval_actions[perturb_mask], adv_agent_ids[perturb_mask], eval_adv_actions[perturb_mask], axis=1)

                eval_obs, _, eval_rewards, eval_dones, _, eval_available_actions = self.envs.step(eval_actions[0])
                rewards += eval_rewards[0][0]
                if self.manual_render:
                    self.envs.render()
                if self.manual_delay:
                    time.sleep(0.1)
                eval_dones_env = np.all(eval_dones)
                current_timesteps += 1
                if eval_dones_env:
                    print(f'total reward of this episode: {rewards}')
                    break
                
        if "smac" in self.args["env"]:  # replay for smac, no rendering
            if "v2" in self.args["env"]:
                self.envs.env.save_replay()
            else:
                self.envs.save_replay()

    def restore(self):
        """Restore the model"""
        if self.algo_args['victim']['model_dir'] is not None:  # restore victim
            print("Restore victim from", self.algo_args['victim']['model_dir'])
            if self.victim_share_param:
                self.victims[0].restore(self.algo_args['victim']['model_dir'])
            else:
                for agent_id in range(self.num_agents):
                    self.victims[agent_id].restore(os.path.join(self.algo_args['victim']['model_dir'], str(agent_id)))

        if self.algo_args['train']['model_dir'] is not None:  # restore model
            print("Restore model from", self.algo_args['train']['model_dir'])
            self.algo.restore(str(self.algo_args['train']['model_dir']))

    def save(self):
        """Save the model"""
        self.algo.save(str(self.save_dir))

    def close(self):
        """Close environment, writter, and log file."""
        if self.algo_args['train']['use_render']:
            self.envs.close()
        else:
            self.envs.close()
            if self.algo_args["train"]["use_eval"] and self.eval_envs is not self.envs:
                self.eval_envs.close()
            self.writter.close()
            self.logger.close()