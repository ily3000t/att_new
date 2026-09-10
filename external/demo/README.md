## 说明文档

### 自定义智能体的用法

#### 如何实现一个新的自定义智能体

在 `/amb/models` 中有已经定义好的基础网络结构，可以在后续智能体构建过程中调用或者自定义新的网络

在 `/amb/agents` 中已经实现了 `coma_agent` , `ddpg_agent` ， `ppo_agnet` ， `q_agent` 等智能体，可以以此为基础修改智能体或者自定义新智能体

修改 `/amb/algorithms/{算法名}.py` ，选择自己使用的智能体

#### 自定义智能体接口介绍

智能体的配置参数全部写在 `amb/configs/algo_cfgs/{算法名}.yaml` 中，这些参数会被以字典形式读入

你的环境类需要满足 `amb/agents/base_agent.py` 中的所有接口，包括输入输出的类型限制和要求


### 针对base_agent中的各个方法的理解

- `collect(obs, rnn_states, masks, available_actions=None, t=0) → action_dist` 注意collect中有个参数t，是指在训练过程中当前的时间步，故collect是用在训练过程中，代表的是训练阶段收集数据的表现。
- `perform(obs, rnn_states, masks, available_actions=None, deterministic=False) → action_dist` perform是在测试阶段收集数据用的，deterministic参数决定测试时是否要用随机策略。
- `sample(obs, available_actions=None) → action_dist` 是用在warmup阶段，执行的是随机采样，不走任何的策略，纯随机，目的是准备数据用的。
- `forward(obs, rnn_states, masks, available_actions=None)` 用在了`igs`攻击的时候，针对forward出的结果进行攻击。
- 只有off-policy中有warmup_steps，on-policy中的数据直接用来训练。在warmup_steps里的用sample()，在正式训练的steps里用collect()。