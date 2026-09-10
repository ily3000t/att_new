import json
import os


i18n = {
    "zh": {
        "exp_name": "实现细节",
        "vanilla_reward": "正常训练",
        "adv_reward": "对抗攻击",
        "vanilla_win_rate": "攻击前胜率",
        "adv_win_rate": "攻击后胜率",
        "srr": "自身鲁棒性",
        "tpr": "Trick性能",
        "trr": "Trick鲁棒性",
        "w-srr": "自身鲁棒性",
        "w-tpr": "Trick性能",
        "w-trr": "Trick鲁棒性",
        "Reward change rate": "回报变化率",
        "adaptive_action": "自适应动作扰动",
        "iterative_perturbation": "最优动作抑制扰动",
        "random_noise": "随机噪声",
        "random_policy": "随机策略",
        "traitor": "零和博弈",
        "reward": "平均回报",
        "trick": "实现细节",
        "A": "探索与利用",
        "B": "网络架构",
        "C": "优化器",
        "D": "优势估计",
        "E": "多智能体特性",
        "Timestep": "时间步",
        "Environments": "环境",
        "Reward": "平均全局奖励",
        "Episode Reward": "平均全局奖励",
        "Change Rate": "平均全局奖励变化率",
        "Adversarial Methods": "对抗攻击算法",
        "Tricks": "技巧",
        "Robustness Change Rate": "鲁棒性变化率",
    },
    "en": {
        "exp_name": "Trick",
        "vanilla_reward": "Vanilla",
        "adv_reward": "Adversarial",
        "vanilla_win_rate": "Vanilla Win Rate",
        "adv_win_rate": "Adversarial Win Rate",
        "srr": "Self Robustness Rate",
        "tpr": "Trick Performance Rate",
        "trr": "Trick Robustness Rate",
        "w-srr": "Self Robustness Rate (Win Rate)",
        "w-tpr": "Trick Performance Rate (Win Rate)",
        "w-trr": "Trick Robustness Rate (Win Rate)",
        "Reward change rate": "Reward change rate",
        "adaptive_action": "Adaptive Action",
        "iterative_perturbation": "Iterative Perturbation",
        "random_noise": "Random Noise",
        "random_policy": "Random Policy",
        "traitor": "Traitor",
        "reward": "Episode Reward",
        "trick": "Trick",
        "A": "Exploration and Exploitation",
        "B": "Network Architecture",
        "C": "Optimizer",
        "D": "Advantage Estimation",
        "E": "Multi-Agent Feature",
    },
}

def get_display(flag, str):
    return i18n[flag][str] if str in i18n[flag] else str

def switch_display(flag, str1, str2):
    return str1 if i18n[flag] == "en" else str2


YLIM = {
    "pettingzoo_mpe_simple_speaker_listener_v4-continuous_maddpg": [-110, -10],
    "pettingzoo_mpe_simple_spread_v3-continuous_maddpg": [-120, -20],
}

BOXPLOT_YLIM = {
    "mappo": [-1, 1],
    "maddpg": [-5, 5],
    "qmix": [-1, 1],
    "all": [-2, 2],
    "mamujoco": [-1, 1],
    "pettingzoo_mpe": [-1, 1],
    "smac": [-1, 1],
    "random_noise": [-1, 1],
    "random_policy": [-1, 1],
    "traitor": [-1, 1],
    "adaptive_action": [-1, 1],
    "iterative_perturbation": [-1, 1],
}

# 读取scheme.json
SCHEME_CFG = json.load(open(os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "settings/scheme.json")), "r"))
