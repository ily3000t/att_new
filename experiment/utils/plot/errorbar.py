import os
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from utils.plot.config import BOXPLOT_YLIM, get_display
from utils.plot.colors import sci_colors, line_markers


def errorbar_metrics(row_wise_results, title, figurename, argv):
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
    offset = 0.1  # 偏移量
    num_points_per_x = len(row_wise_results[exp_names[0]].keys())
    for idx, cat in enumerate(positions):
        for point, metric in enumerate(row_wise_results[exp_names[idx]].keys()):
            x_val = cat + (point - num_points_per_x / 2) * offset  # 计算偏移后的x值
            plt.errorbar(
                x_val,
                row_wise_results[exp_names[idx]][metric]["mean"],
                yerr=row_wise_results[exp_names[idx]][metric]["std"],
                fmt=line_markers[point],
                color=sci_colors[point],
                capsize=0,
                ecolor=sci_colors[point],
                alpha=0.4,
            )

    # 将errorbar的点按照metric连接起来，形成折线图
    for point, metric in enumerate(row_wise_results[exp_names[0]].keys()):
        x_vals = [cat + (point - num_points_per_x / 2) * offset for cat in positions]
        y_vals = [row_wise_results[exp_name][metric]["mean"] for exp_name in exp_names]
        plt.plot(x_vals, y_vals, linestyle="--", marker=line_markers[point], color=sci_colors[point], label=metric)

    # 在y=0处添加一条水平线
    plt.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    # 设置Y轴为百分比格式
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    # plt.grid(True, axis='both', linestyle='--')
    plt.xticks(positions, exp_names, rotation=35, ha="right")
    plt.xlabel(get_display(argv["i18n"], "Tricks"))
    plt.ylabel(get_display(argv["i18n"], "Robustness Change Rate"))
    plt.title(title, fontproperties='Times New Roman')
    plt.yticks(fontproperties='Times New Roman')
    plt.xticks(fontproperties='Times New Roman')
    plt.legend(prop={'family': 'Times New Roman'})
    plt.tight_layout()

    # 保存图表到文件
    save_dir = os.path.dirname(figurename)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(figurename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved to {figurename}")