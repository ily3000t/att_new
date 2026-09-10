import os
import torch
from torch import nn
import numpy as np
from amb.agents.ppo_agent import PPOAgent
from amb.models.critic.v_critic import VCritic
from amb.utils.model_utils import update_linear_schedule, get_grad_norm, huber_loss, mse_loss
from amb.utils.env_utils import check


class MAPPO:
    def __init__(self, args, num_agents, obs_spaces, share_obs_space, act_spaces, device=torch.device("cpu")):
        # save arguments
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.num_agents = num_agents
        self.share_param = args['share_param']

        self.data_chunk_length = args["data_chunk_length"]
        self.use_recurrent_policy = args["use_recurrent_policy"]
        self.use_policy_active_masks = args["use_policy_active_masks"]
        self.action_aggregation = args["action_aggregation"]

        self.ppo_epoch = args["ppo_epoch"]
        self.actor_num_mini_batch = args["actor_num_mini_batch"]
        self.critic_epoch = args["critic_epoch"]
        self.critic_num_mini_batch = args["critic_num_mini_batch"]

        self.entropy_coef = args["entropy_coef"]
        self.clip_param = args["clip_param"]
        self.use_max_grad_norm = args["use_max_grad_norm"]
        self.max_grad_norm = args["max_grad_norm"]

        self.use_clipped_value_loss = args["use_clipped_value_loss"]
        self.value_loss_coef = args["value_loss_coef"]
        self.use_huber_loss = args["use_huber_loss"]
        self.huber_delta = args["huber_delta"]

        self.lr = args["lr"]
        self.critic_lr = args["critic_lr"]
        self.opti_eps = args["opti_eps"]
        self.weight_decay = args["weight_decay"]

        self.obs_spaces = obs_spaces
        self.share_obs_space = share_obs_space
        self.act_spaces = act_spaces
        self.action_type = self.act_spaces[0].__class__.__name__
        
        self.agents = []
        self.actors = []
        self.actor_optimizers = []

        if self.share_param:
            agent = PPOAgent(args, obs_spaces[0], act_spaces[0], device)
            optimizer = torch.optim.Adam(
                agent.actor.parameters(),
                lr=self.lr,
                eps=self.opti_eps,
                weight_decay=self.weight_decay,
            )
            for agent_id in range(self.num_agents):
                self.agents.append(agent)
                self.actors.append(agent.actor)
                self.actor_optimizers.append(optimizer)
        else:
            for agent_id in range(self.num_agents):
                agent = PPOAgent(args, obs_spaces[agent_id], act_spaces[agent_id], device)
                optimizer = torch.optim.Adam(
                    agent.actor.parameters(),
                    lr=self.lr,
                    eps=self.opti_eps,
                    weight_decay=self.weight_decay,
                )
                self.agents.append(agent)
                self.actors.append(agent.actor)
                self.actor_optimizers.append(optimizer)

        self.critic = VCritic(args, self.share_obs_space, self.device)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
            eps=self.opti_eps,
            weight_decay=self.weight_decay,
        )

    @staticmethod
    def create_agent(args, obs_space, act_space, device=torch.device("cpu")):
        return PPOAgent(args, obs_space, act_space, device)

    def lr_decay(self, episode, episodes):
        if self.share_param:
            update_linear_schedule(self.actor_optimizers[0], episode, episodes, self.lr)
        else:
            for agent_id in range(self.num_agents):
                update_linear_schedule(self.actor_optimizers[agent_id], episode, episodes, self.lr)

    def evaluate_actions(self, agent_id, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        action = check(action).to(**self.tpdv)
        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        action_dist, _ = self.actors[agent_id](obs, rnn_states, masks, available_actions)
        action_log_probs = action_dist.log_probs(action)
        if active_masks is not None:
            if self.action_type == "Discrete":
                dist_entropy = (
                    action_dist.entropy() * active_masks.squeeze(-1)
                ).sum() / active_masks.sum()
            else:
                dist_entropy = (
                    action_dist.entropy() * active_masks.squeeze(-1)
                ).sum() / active_masks.sum()
        else:
            dist_entropy = action_dist.entropy().mean()

        return action_log_probs, dist_entropy

    def update_actor(self, sample, agent_id, ernie_regularizer=None):
        obs = sample["obs"]
        rnn_states_actor = sample["rnn_states_actor"]
        actions = sample["actions"]
        masks = sample["masks"]
        active_masks = sample["active_masks"]
        old_action_log_probs = sample["action_log_probs"]
        target_advantage = sample["advantages"]
        available_actions = None
        if "available_actions" in sample:
            available_actions = sample["available_actions"]

        old_action_log_probs = check(old_action_log_probs).to(**self.tpdv)
        target_advantage = check(target_advantage).to(**self.tpdv)
        active_masks = check(active_masks).to(**self.tpdv)

        # reshape to do in a single forward pass for all steps
        action_log_probs, dist_entropy = self.evaluate_actions(
            agent_id, obs, rnn_states_actor, actions, masks, available_actions, active_masks)
        # update actor
        imp_weights = getattr(torch, self.action_aggregation)(
            torch.exp(action_log_probs - old_action_log_probs), dim=-1, keepdim=True)

        surr1 = imp_weights * target_advantage
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * target_advantage

        if self.use_policy_active_masks:
            policy_action_loss = (
                -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True) * active_masks
            ).sum() / active_masks.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        # 加入 ERNIE 正则项
        if self.args.get("use_regulation") is not None:
            ernie_lambda = self.args.get("ernie_lambda", 0.2)
            policy_loss = policy_loss + ernie_lambda * ernie_regularizer

        self.actor_optimizers[agent_id].zero_grad()

        (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self.use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.actors[agent_id].parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_grad_norm(self.actors[agent_id].parameters())

        self.actor_optimizers[agent_id].step()

        return policy_loss, dist_entropy, actor_grad_norm, imp_weights
    
    def cal_value_loss(self, values, value_preds_batch, return_batch, value_normalizer=None):
        """Calculate value function loss.
        Args:
            values: (torch.Tensor) value function predictions.
            value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
            return_batch: (torch.Tensor) reward to go returns.
            value_normalizer: (PopArt) normalize the rewards, denormalize critic outputs.
        Returns:
            value_loss: (torch.Tensor) value function loss.
        """
        if value_normalizer is not None:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
                -self.clip_param, self.clip_param
            )
            error_clipped = value_normalizer(return_batch) - value_pred_clipped
            error_original = value_normalizer(return_batch) - values
        else:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(
                -self.clip_param, self.clip_param
            )
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self.use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self.use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        value_loss = value_loss.mean()

        return value_loss
    
    def compute_ernie_regularizer(self, obs, rnn_states, masks, available_actions, agent_id, epsilon):
        """计算 ERNIE 对抗正则项
        R_π(ok;θk) = max |δ|≤ϵ D(πθk(ok+ δ),πθk(ok))
        
        Args:
            obs: 观察值
            rnn_states: RNN隐藏状态
            masks: 掩码
            available_actions: 可用动作（如果有）
            agent_id: 智能体ID
            epsilon: 扰动大小上限
            
        Returns:
            ernie_reg: 计算出的正则项值
        """
        # 将numpy数组转换为tensor
        obs_tensor = check(obs).to(**self.tpdv)
        rnn_states_tensor = check(rnn_states).to(**self.tpdv)
        masks_tensor = check(masks).to(**self.tpdv)
        
        if available_actions is not None:
            available_actions_tensor = check(available_actions).to(**self.tpdv)
        else:
            available_actions_tensor = None
        
        # 获取原始策略输出
        with torch.no_grad():
            action_dist_original, _ = self.actors[agent_id](obs_tensor, rnn_states_tensor, masks_tensor, available_actions_tensor)
        
        # 创建可扰动的观察值副本
        obs_perturbed = obs_tensor.clone().detach().requires_grad_(True)
        
        # 获取扰动观察值下的策略输出
        action_dist_perturbed, _ = self.actors[agent_id](obs_perturbed, rnn_states_tensor, masks_tensor, available_actions_tensor)
        
        # 根据动作空间类型计算策略分布之间的距离
        if self.action_type == "Discrete":
            # 对于离散动作，使用KL散度
            original_probs = action_dist_original.probs
            perturbed_probs = action_dist_perturbed.probs
            # 计算KL散度: KL(p||q) = sum(p * log(p/q))
            distance = torch.sum(original_probs * torch.log(original_probs / (perturbed_probs + 1e-10) + 1e-10), dim=-1).mean()
        else:
            # 对于连续动作，计算均值和标准差之间的差异
            # 假设是高斯分布，使用均值的MSE和标准差的相对变化
            distance = torch.nn.functional.mse_loss(
                action_dist_perturbed.mean, action_dist_original.mean)
            
            if hasattr(action_dist_perturbed, 'stddev') and hasattr(action_dist_original, 'stddev'):
                std_diff = torch.nn.functional.mse_loss(
                    action_dist_perturbed.stddev, action_dist_original.stddev)
                distance += std_diff
        
        # 计算梯度
        distance.backward(retain_graph=True)
        
        # 获取梯度
        gradient = obs_perturbed.grad.data
        
        # 归一化梯度
        grad_norm = torch.norm(gradient, p=2, dim=-1, keepdim=True)
        grad_norm = torch.clamp(grad_norm, min=1e-8)  # 防止除零
        normalized_gradient = gradient / grad_norm
        
        # 生成最大化距离的对抗扰动
        delta = epsilon * normalized_gradient
        
        # 应用扰动到观察值
        with torch.no_grad():
            obs_adversarial = obs_tensor + delta
        
        # 计算对抗样本下的策略输出
        action_dist_adversarial, _ = self.actors[agent_id](obs_adversarial, rnn_states_tensor, masks_tensor, available_actions_tensor)
        
        # 计算最终正则项值
        with torch.no_grad():
            if self.action_type == "Discrete":
                original_probs = action_dist_original.probs
                adversarial_probs = action_dist_adversarial.probs
                ernie_reg = torch.sum(original_probs * torch.log(original_probs / (adversarial_probs + 1e-10) + 1e-10), dim=-1).mean()
            else:
                ernie_reg = torch.nn.functional.mse_loss(
                    action_dist_adversarial.mean, action_dist_original.mean)
                
                if hasattr(action_dist_adversarial, 'stddev') and hasattr(action_dist_original, 'stddev'):
                    std_diff = torch.nn.functional.mse_loss(
                        action_dist_adversarial.stddev, action_dist_original.stddev)
                    ernie_reg += std_diff
        
        return ernie_reg

    def update_critic(self, sample, value_normalizer=None):
        share_obs = sample["share_obs"]
        rnn_states_critic = sample["rnn_states_critic"]
        value_preds = sample["value_preds"]
        returns = sample["returns"]
        masks = sample["masks"]

        value_preds = check(value_preds).to(**self.tpdv)
        returns = check(returns).to(**self.tpdv)

        values, _ = self.critic(share_obs, rnn_states_critic, masks)

        value_loss = self.cal_value_loss(
            values, value_preds, returns, value_normalizer=value_normalizer)

        self.critic_optimizer.zero_grad()

        (value_loss * self.value_loss_coef).backward()

        if self.use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(
                self.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_grad_norm(self.critic.parameters())

        self.critic_optimizer.step()

        return value_loss, critic_grad_norm

    def train_critic(self, buffers, value_normalizer=None):
        train_info = {}

        train_info["value_loss"] = 0
        train_info["critic_grad_norm"] = 0

        for _ in range(self.critic_epoch):
            data_generators = []
            for agent_id in range(self.num_agents):
                if self.use_recurrent_policy:
                    data_generator = buffers[agent_id].chunk_generator(self.actor_num_mini_batch, self.data_chunk_length)
                else:
                    data_generator = buffers[agent_id].step_generator(self.actor_num_mini_batch)
                data_generators.append(data_generator)

            for batches in self.share_generator(data_generators):
                value_loss, critic_grad_norm = self.update_critic(batches, value_normalizer=value_normalizer)

                train_info["value_loss"] += value_loss.item()
                train_info["critic_grad_norm"] += critic_grad_norm

        num_updates = self.critic_epoch * self.critic_num_mini_batch

        for k, _ in train_info.items():
            train_info[k] /= num_updates

        return train_info

    def train_actor(self, buffer, agent_id, ernie_regularizer=None):
        train_info = {}
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["ratio"] = 0
        if self.args.get("use_regulation"):
            train_info["ernie_reg"] = 0 

        for _ in range(self.ppo_epoch):
            if self.use_recurrent_policy:
                data_generator = buffer.chunk_generator(self.actor_num_mini_batch, self.data_chunk_length)
            else:
                data_generator = buffer.step_generator(self.actor_num_mini_batch)

            for sample in data_generator:
                policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update_actor(sample, agent_id, ernie_regularizer)

                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()
                if ernie_regularizer is not None:
                    train_info["ernie_reg"] += ernie_regularizer.item()

        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def share_param_train_actor(self, buffers, ernie_regularizers=None):
        train_info = {}
        train_info["policy_loss"] = 0
        train_info["dist_entropy"] = 0
        train_info["actor_grad_norm"] = 0
        train_info["ratio"] = 0
        train_info["ernie_reg"] = 0 if self.args.get("use_regulation") else None

        for _ in range(self.ppo_epoch):
            data_generators = []
            for agent_id in range(self.num_agents):
                if self.use_recurrent_policy:
                    data_generator = buffers[agent_id].chunk_generator(self.actor_num_mini_batch, self.data_chunk_length)
                else:
                    data_generator = buffers[agent_id].step_generator(self.actor_num_mini_batch)
                data_generators.append(data_generator)

            for batches in self.share_generator(data_generators):
                ernie_reg = None
                if self.args.get("use_regulation", False):
                    ernie_reg = sum(ernie_regularizers) / len(ernie_regularizers)
                policy_loss, dist_entropy, actor_grad_norm, imp_weights = self.update_actor(batches, 0, ernie_regularizers)

                train_info["policy_loss"] += policy_loss.item()
                train_info["dist_entropy"] += dist_entropy.item()
                train_info["actor_grad_norm"] += actor_grad_norm
                train_info["ratio"] += imp_weights.mean()
                if ernie_reg is not None:
                    train_info["ernie_reg"] += ernie_reg.item()

        num_updates = self.ppo_epoch * self.actor_num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info
    
    def share_generator(self, data_generators):
        """if actor and critic use the same buffer, when actors have heterogeneous input, there will be exceptions in train_critic()."""
        for _ in range(self.actor_num_mini_batch):
            batches = {}
            for generator in data_generators:
                sample = next(generator)
                for key in sample:
                    if key not in batches:
                        batches[key] = []
                    batches[key].append(sample[key])
            for key in batches:
                if batches[key][0] is None:
                    batches[key] = None
                else:
                    # 在train critic时，actor和critic使用同一个buffer，actor的输入是不同的，会导致train_critic()出现异常
                    # 因此需要先判断是否所有的sample的shape都一样
                    # 如果一致则拼接，否则返回None
                    # 这样，当异构agent时，依旧可以使用同构的share_obs来训练critic
                    if all(sample.shape[1:] == batches[key][0].shape[1:] for sample in batches[key]):
                        batches[key] = np.concatenate(batches[key], axis=0)
                    else:
                        batches[key] = None
            yield batches

    def prep_training(self):
        for actor in self.actors:
            actor.train()
        self.critic.train()

    def prep_rollout(self):
        for actor in self.actors:
            actor.eval()
        self.critic.eval()

    def save(self, path):
        if self.share_param:
            self.agents[0].save(path)
        else:
            for agent_id in range(self.num_agents):
                self.agents[agent_id].save(os.path.join(path, str(agent_id)))
        torch.save(self.critic.state_dict(), os.path.join(path, "critic.pth"))

    def restore(self, path):
        if self.share_param:
            self.agents[0].restore(path)
        else:
            for agent_id in range(self.num_agents):
                self.agents[agent_id].restore(os.path.join(path, str(agent_id)))
        self.critic.load_state_dict(torch.load(os.path.join(path, "critic.pth")))