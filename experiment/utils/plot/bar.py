import os

from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np

from utils.plot.config import YLIM, get_display, i18n
from utils.plot.colors import ray_colors,  macaron_colors_1


def bar(df, excel_path, category, name, argv):
    filename = os.path.basename(excel_path).split(".")[0]
    display = i18n[argv["i18n"]]

    # 新画布
    golden_ratio = 1.618
    width = 15  # 假设宽度为10单位
    height = width / golden_ratio  # 根据黄金比例计算高度
    plt.figure(figsize=(width, height))

    # 计算列数和条形的宽度
    num_columns = len(df.columns) - 1  # 减去exp_name列
    bar_width = 0.12 if len(df) < 10 else 0.25
    space_between_bars = 0.02 if len(df) < 10 else 0
    total_bar_space = num_columns * bar_width + (num_columns - 1) * space_between_bars

    # 生成每个柱状图的位置
    positions = np.arange(len(df))
    bar_positions = [positions + (bar_width + space_between_bars) * i for i in range(num_columns)]

    # 确保组间有足够的间隔
    for idx, pos in enumerate(bar_positions):
        bar_positions[idx] = pos + (idx // num_columns)

    # 创建柱状图
    for i, column in enumerate(df.columns[1:]):
        plt.bar(
            bar_positions[i], df[column], color=ray_colors[i], width=bar_width, edgecolor="grey", label=display[column]
        )

    # 添加图表细节
    # 判断YLIM是否有该filename的key，如果有，则设置Y轴范围
    if filename in YLIM:
        plt.ylim(YLIM[filename])
    # plt.xlabel(display[scheme[0]])
    plt.xticks(positions + total_bar_space / 2, df["exp_name"], rotation=0 if len(df) < 10 else 45, ha="right")
    plt.ylabel(display["reward"])
    # 在y=0处添加一条水平线
    plt.axhline(y=0, color="grey", linestyle="--", linewidth=1)
    # plt.title(scheme_name)
    plt.legend(ncol=10, frameon=True, loc="upper center", bbox_to_anchor=(0.5, 1))
    plt.tight_layout()

    # 保存图表到文件
    # ./out/figures/en/png/smac/3m/mappo/smac_3m_mappo_A1.png
    save_dir = os.path.join(
        argv["out"], "figures", argv["i18n"], argv["type"], argv["env"], argv["scenario"], argv["algo"], category
    )
    os.makedirs(save_dir, exist_ok=True)
    figure_name = os.path.join(
        save_dir, f'{argv["env"]}_{argv["scenario"]}_{argv["algo"]}_{category}_{name}.{argv["type"]}'
    )
    plt.savefig(figure_name, dpi=300, bbox_inches="tight")
    print(f"Saved to {figure_name}")

    # 展示图表
    if argv["show"]:
        plt.show()


def bar_metrics(df, title, figurename, argv):
    # 新画布
    golden_ratio = 1.618
    width = 15  # 假设宽度为10单位
    height = width / golden_ratio  # 根据黄金比例计算高度
    if (argv["i18n"] == "en"):
        plt.rcParams["font.serif"] = ["Times New Roman"]  # linux
    fig, ax = plt.subplots(figsize=(width, height))

    n_categories = len(df)
    bar_width = 0.25
    r1 = np.arange(n_categories)
    r2 = [x + bar_width for x in r1]
    r3 = [x + 2 * bar_width for x in r1]

    # 绘制 SRR 的折线图
    ax.plot(r3, df["SRR"], color="grey", label="SRR", marker="o", linestyle="--", linewidth=1.5)

    # 绘制其他柱状图
    ax.bar(r1, df["TPR"], color=macaron_colors_1[0], width=bar_width, edgecolor="grey", label="TPR")
    ax.bar(r2, df["TRR"], color=macaron_colors_1[1], width=bar_width, edgecolor="grey", label="TRR")
    ax.bar(r3, df["rSRR"], color=macaron_colors_1[2], width=bar_width, edgecolor="grey", label="rSRR")

    # 在y=0处添加一条水平线
    ax.axhline(y=0, color="black", linewidth=0.5)

    # 设置Y轴为百分比格式
    ax.yaxis.set_major_formatter(PercentFormatter(1))

    ax.set_xlabel(get_display(argv["i18n"], "Tricks"))
    ax.set_ylabel(get_display(argv["i18n"], "Change Rate"))
    ax.set_title(title, fontproperties='Times New Roman')
    ax.set_xticks([r + bar_width for r in range(n_categories)])
    ax.set_xticklabels(df["exp_name"], rotation=45, ha="right")
    for tick in ax.get_yticklabels():
        tick.set_fontname("Times New Roman")
    for tick in ax.get_xticklabels():
        tick.set_fontname("Times New Roman")
    ax.legend(prop={'family': 'Times New Roman'})
    plt.tight_layout()

    # 保存图表到文件
    save_dir = os.path.dirname(figurename)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(figurename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved to {figurename}")


def barstd_metrics(row_wise_results, title, figurename, argv):
    # 获取row_wise_results中的所有keys
    exp_names = list(row_wise_results.keys())
    # 位置
    positions = range(len(exp_names))

    # Plotting the line chart with std as the shaded area
    golden_ratio = 1.618
    width = 18  # 假设宽度为10单位
    height = width / golden_ratio  # 根据黄金比例计算高度
    plt.figure(figsize=(width, height))

    # 对每个x坐标的点进行轻微的水平偏移
    offset = 0.2  # 偏移量
    num_points_per_x = len(row_wise_results[exp_names[0]].keys())
    for idx, cat in enumerate(positions):
        for point, metric in enumerate(row_wise_results[exp_names[idx]].keys()):
            x_val = cat + (point - num_points_per_x / 2) * offset  # 计算偏移后的x值
            if idx == 0:
                plt.bar(
                    x_val,
                    row_wise_results[exp_names[idx]][metric]["mean"],
                    yerr=row_wise_results[exp_names[idx]][metric]["std"],
                    edgecolor="grey",
                    width=0.2,
                    capsize=3,
                    ecolor="#aaa",
                    color=macaron_colors_1[point],
                    label=metric,
                )
            else:
                plt.bar(
                    x_val,
                    row_wise_results[exp_names[idx]][metric]["mean"],
                    yerr=row_wise_results[exp_names[idx]][metric]["std"],
                    edgecolor="grey",
                    width=0.2,
                    capsize=3,
                    ecolor="#aaa",
                    color=macaron_colors_1[point],
                )

    # 在y=0处添加一条水平线
    plt.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    # 设置Y轴为百分比格式
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    # plt.grid(True, axis='both', linestyle='--')
    plt.xticks(positions, exp_names, rotation=35, ha="right", fontproperties='Times New Roman')
    plt.xlabel(get_display(argv["i18n"], "Tricks"))
    plt.ylabel(get_display(argv["i18n"], "Robustness Change Rate"))
    plt.title(title, fontproperties='Times New Roman')
    plt.yticks(fontproperties='Times New Roman')
    plt.legend(prop={'family': 'Times New Roman'})
    plt.tight_layout()

    # 保存图表到文件
    save_dir = os.path.dirname(figurename)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(figurename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved to {figurename}")