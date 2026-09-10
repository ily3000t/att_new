import os
from scipy.stats import kendalltau
import pandas as pd
import numpy as np


def cal_kendalltau_correlation(excel_path, args, mean_attack=False):
    xlsx = pd.ExcelFile(excel_path)

    exp_name, vanilla_reward = [], []
    adv_reward = {name: [] for name in xlsx.sheet_names}
    for i, sheet_name in enumerate(xlsx.sheet_names):
        # 读取每个工作表中的特定列数据
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=0)
        if i == 0:
            exp_name = df["exp_name"].tolist()
            exp_name.append(exp_name[0])
            exp_name = exp_name[1:]
            # 对于exp_name的每个元素，去掉timestep_前缀，并将default改为final
            exp_name = [name.split("_")[-1] for name in exp_name]
            exp_name = ["final" if name == "default" else name for name in exp_name]

            vanilla_reward = df["vanilla_reward"].tolist()
            vanilla_reward.append(vanilla_reward[0])
            vanilla_reward = vanilla_reward[1:]

        ar = df["adv_reward"].tolist()
        ar.append(ar[0])
        ar = ar[1:]
        adv_reward[sheet_name] = ar
    
    # 是否对5种攻击的reward求均值
    if mean_attack:
        # 对于adv_reward的数据，遍历每个item，将对应下标的数据求均值和标准差
        attack_ts = [[] for _ in range(len(adv_reward["random_noise"]))]
        for i in range(len(attack_ts)):
            for name, value in adv_reward.items():
                attack_ts[i].append(value[i])

        # 对attack_ts的每个item求均值和标准差
        attack_mean = [np.mean(item) for item in attack_ts]
        # attack_std = [np.std(item) for item in attack_ts]
        adv_reward = {"mean of five attacks": attack_mean}
    
    # 对于exp_name[:-1]的数据，替换为从10开始的等差数列，差值为10
    exp_name[:-1] = np.arange(10, (len(exp_name) - 1) * 10 + 10, 10)

    # 如果vanilla_reward中的元素都相等，则将第一个元素减去1e-6
    if len(set(vanilla_reward)) == 1:
        vanilla_reward[0] = vanilla_reward[0] - 1e-6

    # 计算Kendall Tau相关系数
    kendall_results = {"env": [], "scenario": [], "algo": [], "attack": [], "tau": [], "p": [], "direction": [], "level": [], "significance": []}
    for name, values in adv_reward.items():
        coef, p = kendalltau(vanilla_reward, values)
        kendall_results["env"].append(args.env)
        kendall_results["scenario"].append(args.scenario)
        kendall_results["algo"].append(args.algo)
        kendall_results["attack"].append(name)
        kendall_results["tau"].append(coef)
        kendall_results["p"].append(p)
        kendall_results["direction"].append("正相关性" if coef > 0 else "负相关性" if coef < 0 else "无相关性")
        kendall_results["level"].append("较强" if abs(coef) > 0.7 else "中等" if abs(coef) > 0.3 else "较弱" if abs(coef) > 0 else "无")
        kendall_results["significance"].append("统计显著" if p < 0.05 else "显著性不强" if p < 0.5 else "显著性较低" if p < 1 else "无")

    # 将结果写入到excel文件中
    df = pd.DataFrame(kendall_results)
    
    # 往指定的excel文件中追加数据
    output = os.path.join(args.out, 'data', "es_kendalltau_correlation.xlsx")
    mode = "w" if not os.path.exists(output) else "a"
    with pd.ExcelWriter(output, mode=mode, if_sheet_exists="overlay" if mode =="a" else None, engine="openpyxl") as writer:
        # 读取已有的数据
        if mode == "a":
            df_old = pd.read_excel(output)
            # 合并数据，如果有重复的数据则去重
            df = pd.concat([df_old, df], ignore_index=True).drop_duplicates(subset=["env", "scenario", "algo", "attack"], keep="last")
        
        # 按attack列排序
        df = df.sort_values(by="attack")
        # 追加数据，不附盖
        df.to_excel(writer, index=False)
        