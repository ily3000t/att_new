import argparse
import json
from matplotlib.ticker import PercentFormatter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import os
from utils.path import get_env_scenario_algos
from utils.plot.boxplot import boxplot_cr
from utils.plot.errorbar import errorbar_metrics
from utils.plot.bar import bar_metrics, barstd_metrics
from utils.plot.tensorboard import plot_train_reward
from utils.plot.necessity import plot_necessity
from utils.plot.line import line_tricks, line_earlystopping
from utils.plot.colors import line_markers, macaron_colors_1

config = {
    "font.family":'serif',
    "font.size": 15,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
    "axes.unicode_minus": False, # 解决负号无法显示的问题
}
matplotlib.rcParams.update(config)
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # macos font
# plt.rcParams["font.family"] = ["WenQuanYi Micro Hei"]  # linux


# 读取scheme.json
SCHEME_CFG = json.load(open(os.path.abspath(os.path.join(os.path.dirname(__file__), "settings/scheme.json")), "r"))


def _read_one_excel_cr(path):
    # 读取Excel文件
    xlsx = pd.ExcelFile(path)
    sheet_names = xlsx.sheet_names

    # Loading data from all sheets
    sheets_data = {sheet: pd.read_excel(path, sheet_name=sheet) for sheet in sheet_names}

    # 取出第一个sheet的exp_name列
    exp_names = sheets_data[sheet_names[0]]["exp_name"]
    # Metrics to calculate mean and std
    origin_metrics = ["TPR", "TRR"]

    # Calculating mean and std for each metric, for each row across all sheets
    row_wise_results = {}
    for i, exp_name in enumerate(exp_names):
        row_metrics = {metric: [] for metric in origin_metrics}

        for sheet in sheet_names:
            for metric in origin_metrics:
                row_metrics[metric].append(sheets_data[sheet].iloc[i][metric])

        # 再把一行的rSRR、TPR、TRR三个指标合并成一个list
        row_metrics["CR"] = [row_metrics[metric] for metric in origin_metrics]
        row_metrics["CR"] = np.array(row_metrics["CR"]).flatten().tolist()

        row_wise_results[exp_name] = row_metrics

    return row_wise_results


def _read_one_sheet_cr(path, sheet_name):
    # Loading data from all sheets
    sheet_data = pd.read_excel(path, sheet_name=sheet_name)

    # 取出第一个sheet的exp_name列
    exp_names = sheet_data["exp_name"]
    # Metrics to calculate mean and std
    origin_metrics = ["TPR", "TRR"]

    # Calculating mean and std for each metric, for each row across all sheets
    row_wise_results = {}
    for i, exp_name in enumerate(exp_names):
        row_metrics = {metric: [] for metric in origin_metrics}

        for metric in origin_metrics:
            row_metrics[metric].append(sheet_data.iloc[i][metric])

        # 再把一行的rSRR、TPR、TRR三个指标合并成一个list
        row_metrics["CR"] = [row_metrics[metric] for metric in origin_metrics]
        row_metrics["CR"] = np.array(row_metrics["CR"]).flatten().tolist()

        row_wise_results[exp_name] = {"CR": row_metrics["CR"]}

    return row_wise_results


def _read_one_excel_metrics(path):
    # 读取Excel文件
    xlsx = pd.ExcelFile(path)
    sheet_names = xlsx.sheet_names

    # Loading data from all sheets
    sheets_data = {sheet: pd.read_excel(path, sheet_name=sheet) for sheet in sheet_names}

    # 取出第一个sheet的exp_name列
    exp_names = sheets_data[sheet_names[0]]["exp_name"]
    # Metrics to calculate mean and std
    origin_metrics = ["TPR", "TRR", "rSRR"]

    # Calculating mean and std for each metric, for each row across all sheets
    row_wise_results = {}
    for i, exp_name in enumerate(exp_names):
        row_metrics = {metric: [] for metric in origin_metrics}

        for sheet in sheet_names:
            for metric in origin_metrics:
                row_metrics[metric].append(sheets_data[sheet].iloc[i][metric])

        row_wise_results[exp_name] = {
            metric: {"mean": np.array(row_metrics[metric]).mean(), "std": np.array(row_metrics[metric]).std()}
            for metric in origin_metrics
        }

    return row_wise_results


def plot_tricks(excel_path, argv, ylabel="Episode Reward"):
    # 读取Excel文件
    xlsx = pd.ExcelFile(excel_path)

    # 遍历所有工作表，合并df
    combined_df = pd.DataFrame()
    for i, sheet in enumerate(xlsx.sheet_names):
        # 读取每个工作表中的所有数据
        df = pd.read_excel(excel_path, sheet_name=sheet, header=0, index_col=0)
        # 给df增加一列scheme，用于分组
        if argv["groupby"] == "scheme":
            df["scheme"] = df["exp_name"].apply(lambda x: next((value for key, value in SCHEME_CFG["params"].items() if key in x), "Unknown")[0])
        elif argv["groupby"] == "trick":
            df["scheme"] = df["exp_name"].apply(lambda x: next((value for key, value in SCHEME_CFG["params"].items() if key in x), "Unknown"))
        else:
            raise ValueError(f"Unknown groupby: {argv['groupby']}")

        if i == 0:
            # 填充默认数据列
            combined_df["scheme"] = df["scheme"]
            combined_df["exp_name"] = df["exp_name"]
            if ylabel == "Win Rate":
                combined_df["vanilla_win_rate"] = df["vanilla_win_rate"]
            else:
                combined_df["vanilla_reward"] = df["vanilla_reward"]

        # 对combined_df增加一列，以sheet作为列名，值为df中的adv_reward
        if ylabel == "Win Rate":
            combined_df[sheet] = df["adv_win_rate"]
        else:
            combined_df[sheet] = df["adv_reward"]

    # 把scheme相同的数据分组
    grouped = combined_df.groupby("scheme")
    row_default = grouped.get_group("+").copy()
    for name, group in grouped:
        if name == SCHEME_CFG["params"]["default"]:
            continue
        # 合并两个df
        plot_data = pd.concat([row_default, group], ignore_index=True)
        # plot_data去除scheme列
        plot_data = plot_data.drop(columns=["scheme"])
        # 画图
        figurename = os.path.join(
            argv["out"],
            "figures",
            argv["i18n"],
            argv["type"],
            argv["env"],
            argv["scenario"],
            argv["algo"],
            "tricks",
            f'{"winrate" if ylabel == "Win Rate" else "reward"}_{name}.{argv["type"]}',
        )
        line_tricks(plot_data, name, ylabel, figurename, argv)


def plot_metrics(excel_path, argv):
    # 读取Excel文件
    xlsx = pd.ExcelFile(excel_path)

    # 遍历所有工作表，依次画图
    for sheet_name in xlsx.sheet_names:
        # 读取每个工作表中的特定列数据
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=0)
        # 画图
        title = f"{argv['env']}_{argv['scenario']}_{argv['algo']}_{sheet_name}"
        figurename = os.path.join(
            argv["out"],
            "figures",
            argv["i18n"],
            argv["type"],
            argv["env"],
            argv["scenario"],
            argv["algo"],
            "metrics",
            f'{title}.{argv["type"]}',
        )
        bar_metrics(df, title, figurename, argv)


def boxplot_envalgo(excel_path, argv):
    row_wise_results = _read_one_excel_cr(excel_path)

    filename = os.path.basename(excel_path).split(".")[0]
    figurename = os.path.join(
        argv["out"], "figures", argv["i18n"], argv["type"], "boxplot", f'{filename}.{argv["type"]}'
    )
    boxplot_cr(row_wise_results, filename, figurename, argv)


def boxplot_algo(argv, metrics=["CR"]):
    # 遍历argv["out"]/data下的所有excel，按最后的算法分组
    excel_paths = {"mappo": [], "maddpg": [], "qmix": []}
    for root, _, files in os.walk(os.path.join(os.path.dirname(__file__), argv["out"], "data")):
        for file in files:
            if file.endswith("_tricks.xlsx"):
                # 获取file的文件名
                algo_name = file.split("_")[-2]
                excel_paths[algo_name].append(os.path.join(root, file))

    # 依次遍历每个算法下的所有excel，合并列名相同的数据
    for algo_name, paths in excel_paths.items():
        row_wise_results = {}
        # 遍历这个算法下所有的excel，数据合并到row_wise_results
        for i, path in enumerate(paths):
            one_results = _read_one_excel_cr(path)

            if i == 0:
                row_wise_results = one_results
            else:
                for exp_name, mts in one_results.items():
                    for metric, values in mts.items():
                        if exp_name not in row_wise_results:
                            row_wise_results[exp_name] = one_results[exp_name]
                        else:
                            row_wise_results[exp_name][metric].extend(values)

        figurename = os.path.join(
            argv["out"], "figures", argv["i18n"], argv["type"], "boxplot", f'algo_{algo_name}.{argv["type"]}'
        )
        boxplot_cr(row_wise_results, algo_name, figurename, argv, metrics)


def boxplot_env(argv, metrics=["CR"]):
    # 遍历argv["out"]/data下的所有excel，按最后的算法分组
    excel_paths = {"mamujoco": [], "pettingzoo_mpe": [], "smac": []}
    for root, _, files in os.walk(os.path.join(os.path.dirname(__file__), argv["out"], "data")):
        for file in files:
            if file.endswith("_tricks.xlsx"):
                # 获取file的文件名
                if "mamujoco" in file:
                    env_name = "mamujoco"
                elif "pettingzoo_mpe" in file:
                    env_name = "pettingzoo_mpe"
                elif "smac" in file:
                    env_name = "smac"
                excel_paths[env_name].append(os.path.join(root, file))

    # 依次遍历每个算法下的所有excel，合并列名相同的数据
    for env_name, paths in excel_paths.items():
        row_wise_results = {}
        # 遍历这个算法下所有的excel，数据合并到row_wise_results
        for i, path in enumerate(paths):
            one_results = _read_one_excel_cr(path)

            if i == 0:
                row_wise_results = one_results
            else:
                for exp_name, mts in one_results.items():
                    for metric, values in mts.items():
                        if exp_name not in row_wise_results:
                            row_wise_results[exp_name] = one_results[exp_name]
                        else:
                            row_wise_results[exp_name][metric].extend(values)

        figurename = os.path.join(
            argv["out"], "figures", argv["i18n"], argv["type"], "boxplot", f'env_{env_name}.{argv["type"]}'
        )
        boxplot_cr(row_wise_results, env_name, figurename, argv, metrics)


def boxplot_attack(argv):
    # 遍历argv["out"]/data下的所有excel，按最后的算法分组
    excel_paths = []
    for root, _, files in os.walk(os.path.join(os.path.dirname(__file__), argv["out"], "data")):
        for file in files:
            if file.endswith("_tricks.xlsx"):
                # 获取file的文件名
                excel_paths.append(os.path.join(root, file))

    attack_methods = ["random_noise", "iterative_perturbation", "adaptive_action","learn_obs_all", "learn_obs_sin", "pert_act", "pert_obs_all", "pert_obs_sin", "learn_act", "random_policy", "traitor"]

    for attack_method in attack_methods:

        row_wise_results = {}
        for i, path in enumerate(excel_paths):
            # 遍历这个算法下所有的excel，数据合并到row_wise_results
            one_results = _read_one_sheet_cr(path, attack_method)

            if i == 0:
                row_wise_results = one_results
            else:
                for exp_name, metrics in one_results.items():
                    for metric, values in metrics.items():
                        if exp_name not in row_wise_results:
                            row_wise_results[exp_name] = {metric: values}
                        else:
                            row_wise_results[exp_name][metric].extend(values)

        figurename = os.path.join(
            argv["out"], "figures", argv["i18n"], argv["type"], "boxplot", f'attack_{attack_method}.{argv["type"]}'
        )
        boxplot_cr(row_wise_results, attack_method, figurename, argv)


def boxplot_all(argv):
    # 遍历argv["out"]/data下的所有excel，按最后的算法分组
    excel_paths = []
    for root, _, files in os.walk(os.path.join(os.path.dirname(__file__), argv["out"], "data")):
        for file in files:
            if file.endswith("_tricks.xlsx"):
                # 获取file的文件名
                excel_paths.append(os.path.join(root, file))

    # 依次遍历每个算法下的所有excel，合并列名相同的数据
    row_wise_results = {}
    for i, path in enumerate(excel_paths):
        # 遍历这个算法下所有的excel，数据合并到row_wise_results
        one_results = _read_one_excel_cr(path)

        if i == 0:
            row_wise_results = one_results
        else:
            for exp_name, metrics in one_results.items():
                for metric, values in metrics.items():
                    if exp_name not in row_wise_results:
                        row_wise_results[exp_name] = {metric: values}
                    else:
                        row_wise_results[exp_name][metric].extend(values)

    figurename = os.path.join(argv["out"], "figures", argv["i18n"], argv["type"], "boxplot", f'all.{argv["type"]}')
    boxplot_cr(row_wise_results, "all", figurename, argv)


def errorbar_mean_attack_metric(excel_path, argv):
    row_wise_results = _read_one_excel_metrics(excel_path)

    title = f"{argv['env']}_{argv['scenario']}_{argv['algo']}"
    figurename = os.path.join(
        argv["out"],
        "figures",
        argv["i18n"],
        argv["type"],
        "errorbar",
        f'{argv["env"]}_{argv["scenario"]}_{argv["algo"]}.{argv["type"]}',
    )
    errorbar_metrics(row_wise_results, title, figurename, argv)


def barstd_metric(excel_path, argv):
    row_wise_results = _read_one_excel_metrics(excel_path)
    title = f"{argv['env']}_{argv['scenario']}_{argv['algo']}"
    figurename = os.path.join(
        argv["out"],
        "figures",
        argv["i18n"],
        argv["type"],
        "barstd",
        f'{title}.{argv["type"]}',
    )
    barstd_metrics(row_wise_results, title, figurename, argv)


def plot_early_stopping(env_name, scenario_name, algo_name, argv):
    excel_path = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        argv["out"],
        "data",
        env_name,
        scenario_name,
        algo_name,
        f"{env_name}_{scenario_name}_{algo_name}_earlystopping.xlsx",
    )

    # 读取Excel文件，遍历每个工作表，依次画图
    xlsx = pd.ExcelFile(excel_path)
    exp_name, vanilla_reward, vanilla_winrate = [], [], []
    adv_reward = {name: [] for name in xlsx.sheet_names}
    adv_winrate = {name: [] for name in xlsx.sheet_names}
    for i, sheet_name in enumerate(xlsx.sheet_names):
        # 读取每个工作表中的特定列数据
        df = pd.read_excel(excel_path, sheet_name=sheet_name, header=0)
        # df去掉exp_name=default的行
        df = df[df["exp_name"] != "default"]
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

            vanilla_winrate = df["vanilla_win_rate"].tolist()
            vanilla_winrate.append(vanilla_winrate[0])
            vanilla_winrate = vanilla_winrate[1:]

        ar = df["adv_reward"].tolist()
        ar.append(ar[0])
        ar = ar[1:]
        adv_reward[sheet_name] = ar

        aw = df["adv_win_rate"].tolist()
        aw.append(aw[0])
        aw = aw[1:]
        adv_winrate[sheet_name] = aw

    # 对于exp_name的数据，替换为从1e6开始的等差数列，差值为1e6
    exp_name = np.arange(1e6, len(exp_name) * 1e6 + 1e6, 1e6)

    title = f"{env_name}_{scenario_name}_{algo_name}"
    dirname = os.path.join(
        argv["out"],
        "figures",
        argv["i18n"],
        argv["type"],
        argv["env"],
        argv["scenario"],
        argv["algo"],
        "earlystopping",
    )
    line_earlystopping(
        exp_name, vanilla_reward, adv_reward, title, "Reward", os.path.join(dirname, f"es_reward.{argv['type']}")
    )
    if env_name == "smac":
        line_earlystopping(
            exp_name,
            vanilla_winrate,
            adv_winrate,
            title,
            "Win Rate",
            os.path.join(dirname, f"es_winrate.{argv['type']}"),
        )


def plot_early_stopping_total(argv, mean_attack=False):
    # 遍历argv["out"]/data下的所有excel，按最后的算法分组
    excel_paths = {"mamujoco": [], "pettingzoo_mpe": [], "smac": []}
    for root, _, files in os.walk(os.path.join(os.path.dirname(__file__), argv["out"], "data")):
        for file in files:
            if file.endswith("_earlystopping.xlsx"):
                env_name = root.split("/")[-3]
                # 获取file的文件名
                excel_paths[env_name].append(os.path.join(root, file))

    golden_ratio = 2.0
    width = 20  # 假设宽度为10单位
    height = width / golden_ratio  # 根据黄金比例计算高度
    # 创建2个子图，排列在3行4列
    fig, axs = plt.subplots(3, 4, figsize=(width, height))

    # 依次遍历每个算法下的所有excel，合并列名相同的数据
    for idx, (env_name, paths) in enumerate(excel_paths.items()):

        order = ["mappo", "maddpg", "qmix"]
        # 对paths重新按照algo符合order顺序排序
        paths = sorted(paths, key=lambda x: order.index(x.split("/")[-2]))

        for jdx, path in enumerate(paths):

            xlsx = pd.ExcelFile(path)
            env_name, scenario_name, algo_name = path.split("/")[-4:-1]
            title = f"{scenario_name.replace('simple_', '').replace('-continuous', '').replace('_v4', '').replace('_v3', '')} {algo_name.upper()}"

            exp_name, vanilla_reward = [], []
            adv_reward = {name: [] for name in xlsx.sheet_names}

            for i, sheet_name in enumerate(xlsx.sheet_names):
                # 读取每个工作表中的特定列数据
                df = pd.read_excel(path, sheet_name=sheet_name, header=0)
                # df去掉exp_name=default的行
                df = df[df["exp_name"] != "default"]
                if i == 0:
                    vanilla_reward = df["vanilla_reward"].tolist()

                adv_reward[sheet_name] = df["adv_reward"].tolist()

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
                adv_reward = {"五种对抗攻击全局奖励均值": attack_mean}

            exp_name = np.arange(1e6, len(vanilla_reward) * 1e6 + 1e6, 1e6)

            # 绘制以exp_name为横轴，vanilla_reward为纵轴的折线图
            axs[idx][jdx].plot(
                exp_name,
                vanilla_reward,
                color="grey",
                label="原始全局奖励",
                marker=line_markers[0],
                linestyle="-",
                linewidth=1.5,
            )
            for i, (name, value) in enumerate(adv_reward.items()):
                axs[idx][jdx].plot(
                    exp_name, value, label=name, marker=line_markers[i + 1], linestyle="-", linewidth=1.5
                )

            axs[idx][jdx].set_title(title, fontproperties="Times New Roman")
            axs[idx][jdx].set_xlabel("时间步")
            axs[idx][jdx].set_ylabel("平均全局奖励")
            for tick in axs[idx][jdx].get_yticklabels():
                tick.set_fontname("Times New Roman")
            for tick in axs[idx][jdx].get_xticklabels():
                tick.set_fontname("Times New Roman")
            # 设置x轴范围
            # axs[idx][jdx].set_xlim(0, 1.1e7)

    # 在整个图形上方创建统一的图例
    lines_labels = [axs[0][0].get_legend_handles_labels()]
    lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    fig.legend(lines, labels, loc="upper center", ncol=6, bbox_to_anchor=(0.5, 1.05))
    plt.tight_layout()

    # 保存图表到文件
    figurename = os.path.join(
        argv["out"], "figures", argv["i18n"], argv["type"], "earlystopping", f"earlystopping_total.{argv['type']}"
    )
    save_dir = os.path.dirname(figurename)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(figurename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved to {figurename}")


def plot_share_param(data_path, argv):
    df = pd.read_csv(data_path, header=0)
    # 保留trick=share_param=False的数据
    df = df[df["trick"] == "share\_param\_False"]
    print(df)
    # 按照scenario和algo分组，计算每个scenarios下的TPR、TRR、rSRR的均值和标准差
    df = df.groupby(["scenario", "algo"]).agg(
        {
            "TPR": ["mean", "std"],
            "TRR": ["mean", "std"],
            "rSRR": ["mean", "std"],
        }
    )
    df.columns = ["_".join(col).strip() for col in df.columns.values]
    # 以algo重新排序
    df = df.reset_index().sort_values(by=["algo", "scenario"], ascending=[False, False])
    # 将scenario列中的-continuous、_v4、_v3替换为空
    df["scenario"] = df["scenario"].str.replace("-continuous", "").str.replace("_v4", "").str.replace("_v3", "")
    
    print(df)

    # 画图
    fig, ax = plt.subplots(figsize=(9, 6))
    df.plot(
        kind="bar",
        x="scenario",
        y=["TPR_mean", "TRR_mean", "rSRR_mean"],
        ax=ax,
        rot=0,
        color=macaron_colors_1[:3],
        edgecolor="grey",
    )
    # 在y=0处添加一条水平线
    ax.set_ylabel("metrics")
    ax.set_title("Metrics of share_param=False")
    ax.legend(["TPR", "TRR", "rSRR"])
    ax.set_ylim(-1, 1)
    # 将x轴的标签旋转45度，显示为scenario_algo
    ax.set_xticklabels(df["algo"] + "_" + df["scenario"], rotation=25, ha="right")
    plt.axhline(y=0, color="grey", linestyle="-", linewidth=0.5)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.tight_layout()

    # 保存图表到文件
    figurename = os.path.join('out', 'figures', argv["i18n"], argv["type"], "extend", f"share_param.{argv['type']}")
    save_dir = os.path.dirname(figurename)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(figurename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved to {figurename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", type=str, default=None, help="env name")
    parser.add_argument("-s", "--scenario", type=str, default=None, help="scenario or map name")
    parser.add_argument("-a", "--algo", type=str, default=None, help="algo name")
    parser.add_argument("-o", "--out", type=str, default="out", help="output dir")
    parser.add_argument("-f", "--file", type=str, default=None, help="Excel file path")
    parser.add_argument("-i", "--i18n", type=str, default="en", choices=["en", "zh"], help="Choose the language")
    parser.add_argument("-t", "--type", type=str, default="png", choices=["png", "pdf"], help="save figure type")
    parser.add_argument("--result_path", type=str, default="results", help="result path")
    parser.add_argument(
        "-g", "--groupby", type=str, default="trick", choices=["trick", "scheme"], help="Group by trick or scheme"
    )
    parser.add_argument("--show", action="store_true", help="Whether to show the plot")
    args, _ = parser.parse_known_args()
    argv = vars(args)

    for env_name, scenario_name, algo_name in get_env_scenario_algos(args):
        argv["env"] = env_name
        argv["scenario"] = scenario_name
        argv["algo"] = algo_name

        if argv["file"] is not None:
            excel_path = argv["file"]
        else:
            excel_path = os.path.join(
                argv["out"],
                "data",
                argv["env"],
                argv["scenario"],
                argv["algo"],
                f"{argv['env']}_{argv['scenario']}_{argv['algo']}_tricks.xlsx",
            )

        # 画训练对比曲线图
        plot_train_reward(argv, SCHEME_CFG)

        # 评一个trick方案下所有攻击的reward
        # x轴为trick，y轴为reward，每个trick方案一张图
        plot_tricks(excel_path, argv)
        if argv["env"] == "smac":
            plot_tricks(excel_path, argv, ylabel="Win Rate")

        # 画metrics
        plot_metrics(excel_path, argv)

        # # 合并attack、metrics的箱线图，每个（环境+算法）一张图，共12张，看的是算法在特定环境上的不同trick的表现
        boxplot_envalgo(excel_path, argv)

        # # 合并attack，画metrics的errorbar图
        # errorbar_mean_attack_metric(excel_path, argv)
        # # 合并attack，画metrics的bar图，带着std
        # barstd_metric(excel_path, argv)

        # 画early stopping图
        # plot_early_stopping(env_name, scenario_name, algo_name, argv)

    # # 合并env、attack、metrics的箱线图，每个算法一张图，共3张，看的是算法在所有环境上的不同trick的表现
    # boxplot_algo(argv, metrics=["TPR", "TRR"])
    # # 合并algo、attack、metrics的箱线图，每个环境一张图，共3张，看的是环境在所有算法上的不同trick的表现
    # boxplot_env(argv, metrics=["TPR", "TRR"])
    # 合并algo、env、metrics的箱线图，每个环境一张图，共3张，看的是环境在所有算法上的不同trick的表现
    # boxplot_attack(argv)

    # 画metric必要性分析图
    # plot_necessity(argv)

    # 画early stopping总图，3行4列
    # plot_early_stopping_total(argv)
    # plot_early_stopping_total(argv, mean_attack=True)

    # x轴为share_param_False下的所有环境，y轴为metrics，共一个图
    # plot_share_param("./out/data/all.csv", argv)
