import os
from matplotlib import pyplot as plt
import numpy as np

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from utils.plot.colors import line_colors
from utils.plot.config import get_display, switch_display

def plot_train_reward(argv, SCHEME_CFG):
    # 根据env, scenario, algo,找到对应的文件夹
    trail_dir = os.path.join(argv["result_path"], argv["env"], argv["scenario"], "single", argv["algo"])
    print(trail_dir)

    _plot_tb_data(argv["groupby"], trail_dir, "env/train_episode_rewards", "Episode Reward", 0.9, argv, SCHEME_CFG)
    _plot_tb_data(argv["groupby"], trail_dir, "env/incre_win_rate", "Win Rate", 0.9, argv, SCHEME_CFG)
    _plot_tb_data(argv["groupby"], trail_dir, "env/eval_return_mean", "Episode Reward", 0.6, argv, SCHEME_CFG)
    _plot_tb_data(argv["groupby"], trail_dir, "env/eval_win_rate", "Win Rate", 0.6, argv, SCHEME_CFG)


def _plot_tb_data(groupby, trail_dir, tag_name, ylabel, weight, argv, SCHEME_CFG):
    dfs = {}
    is_plot = False

    listdir = os.listdir(trail_dir)
    # 先遍历SCHEME_CFG["params"]，再去check目录更好，因为可以保证顺序
    for trick_name, trick_tag in SCHEME_CFG["params"].items():
        # 如果trick_name不在listdir中，跳过
        # trick_name可能是listdir中的元素的一部分，因此需要判断是否包含
        if not any([trick_name in item for item in listdir]):
            continue

        # 找到与trick_name所有相关的文件夹
        trick_dirs = [item for item in listdir if trick_name in item]
        for trick_value_name in trick_dirs:

            # 同时支持scheme和trick两种分组方式，因此需要替换trick_tag
            trick_tag = trick_tag if groupby == "trick" else trick_tag[0]

            if trick_tag not in dfs:
                dfs[trick_tag] = pd.DataFrame(columns=["step", trick_value_name])

            # 读取TensorBoard日志
            log_dir = os.path.join(trail_dir, trick_value_name)
            # 取出log_dir下最新的子文件夹
            log_dir = sorted(os.listdir(log_dir))[-1]
            log_dir = os.path.join(trail_dir, trick_value_name, log_dir, "logs")

            # 判断log_dir是否存在，如果不存在，打印后跳过
            if not os.path.exists(log_dir):
                print(f"{log_dir} not exists")
                continue

            # 创建EventAccumulator对象以读取TensorBoard日志
            event_acc = EventAccumulator(log_dir)
            event_acc.Reload()
            # 判断tag_name是否在event_acc中，如果不在，打印后跳过
            if tag_name in event_acc.Tags()["scalars"]:
                is_plot = True

            # 如果event_acc.Tags()["scalars"]为空，跳过
            if not event_acc.Tags()["scalars"]:
                continue
            elif tag_name not in event_acc.Tags()["scalars"]:
                continue
                
            tag_values = event_acc.Scalars(tag_name)

            # 每个trick_name对应一列
            dfs[trick_tag][trick_value_name] = np.nan
            for value in tag_values:
                # 判断df中是否有step=value.step的行，如果没有，新增一行
                if not dfs[trick_tag][dfs[trick_tag]["step"] == value.step].empty:
                    dfs[trick_tag].loc[dfs[trick_tag]["step"] == value.step, trick_value_name] = value.value

                else:
                    # 新增一行全NaN的数据
                    dfs[trick_tag].loc[len(dfs[trick_tag])] = [np.nan for _ in range(len(dfs[trick_tag].columns))]
                    # 填充列
                    dfs[trick_tag].loc[len(dfs[trick_tag]) - 1, "step"] = value.step
                    dfs[trick_tag].loc[len(dfs[trick_tag]) - 1, trick_value_name] = value.value

    # 画图
    if is_plot:
        _plot_train_line(dfs, tag_name.split("/")[1], ylabel, weight, argv, SCHEME_CFG)


def _plot_train_line(dfs, tag_name, ylabel, weight, argv, SCHEME_CFG):
    # 将tag_data转为df
    for tag, data in dfs.items():

        # 如果data为空，跳过
        if data.empty:
            continue

        # Create the plot with shaded area in a similar color but more transparent
        plt.figure(figsize=(9, 6))

        # 按step升序排序
        data = data.sort_values(by="step", ascending=True)
        # 画图
        # 获取data的列名
        column_names = data.columns.values.tolist()
        # 去除step列
        column_names.remove("step")

        # 画默认曲线
        if tag != SCHEME_CFG["params"]["default"]:
            default_df = dfs[SCHEME_CFG["params"]["default"]]
            smoothed_values = tensorboard_smoothing(default_df["default"], weight=weight)
            plt.plot(default_df["step"], smoothed_values, color=line_colors[0], label="default")
            plt.fill_between(default_df["step"], smoothed_values, default_df["default"], color=line_colors[0], alpha=0.2)

        # 画trick曲线
        for i, column_name in enumerate(column_names):
            # 从data中挑出step和column_name两列
            clean_data = data[["step", column_name]]
            # 去除data中column_name列为NaN的行
            clean_data = clean_data.dropna(subset=[column_name])
            clean_data = clean_data.reset_index(drop=True)

            # Apply TensorBoard-style smoothing
            smoothed_values = tensorboard_smoothing(clean_data[column_name], weight=weight)
            plt.plot(clean_data["step"], smoothed_values, color=line_colors[i + 1], label=column_name)
            plt.fill_between(
                clean_data["step"], smoothed_values, clean_data[column_name], color=line_colors[i + 1], alpha=0.2
            )

        plt.title(f"{argv['env']}_{argv['scenario']}_{argv['algo']}", fontproperties='Times New Roman')
        plt.xlabel(get_display(argv["i18n"], "Timestep"))
        plt.ylabel(get_display(argv["i18n"], ylabel))
        plt.yticks(fontproperties='Times New Roman')
        plt.xticks(fontproperties='Times New Roman')
        plt.legend(prop={'family': 'Times New Roman'})
        plt.grid(True, linestyle="--")
        plt.tight_layout()

        save_dir = os.path.join(
            argv["out"], "figures", argv["i18n"], argv["type"], argv["env"], argv["scenario"], argv["algo"], tag_name
        )
        os.makedirs(save_dir, exist_ok=True)
        figure_name = os.path.join(save_dir, f'{tag}.{argv["type"]}')
        plt.savefig(figure_name, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved to {figure_name}")


def tensorboard_smoothing(values, weight):
    if len(values) == 0:
        return []
    last = values[0]
    smoothed = []
    for point in values:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed
