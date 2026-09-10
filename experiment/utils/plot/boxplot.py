import json
import os
from matplotlib import patches, pyplot as plt
from matplotlib.ticker import PercentFormatter

from utils.plot.config import BOXPLOT_YLIM, get_display
from utils.plot.colors import macaron_colors_1

SCHEME_CFG = json.load(open(os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "settings/scheme.json")), "r"))

def boxplot_cr(row_wise_results, title, figurename, argv, metrics=["CR"]):
    # 获取row_wise_results中的所有keys
    exp_names = list(row_wise_results.keys())
    # 排序，按照SCHEME_CFG["params"]的中的keys顺序
    exp_names.sort(key=lambda x: next((value for key, value in SCHEME_CFG["params"].items() if key in x), "Unknown"))
    # 设置每个箱线图的位置
    positions = range(1, len(exp_names) + 1)

    # Plotting the line chart with std as the shaded area
    golden_ratio = 1.618
    width = 15  # 假设宽度为10单位
    height = width / golden_ratio  # 根据黄金比例计算高度
    plt.figure(figsize=(width, height))

    for exp_name, pos in zip(exp_names, positions):
        for idx, metric in enumerate(metrics):
            plt.boxplot(
                row_wise_results[exp_name][metric],
                positions=[pos + idx * 0.3],
                widths=0.3,
                capwidths=0.2,
                patch_artist=True,
                showmeans=True,
                showfliers=True,
                boxprops=dict(facecolor=macaron_colors_1[idx]),
                meanprops=dict(marker="o", markerfacecolor="black", markeredgecolor="black", markersize=4),
                medianprops=dict(marker=None, color="black", linewidth=1.5),
                flierprops=dict(marker="o", markerfacecolor=macaron_colors_1[idx], markeredgecolor=macaron_colors_1[idx]),
            )

    # algo_name = title.split("_")[-1]
    if title in BOXPLOT_YLIM:
        plt.ylim(BOXPLOT_YLIM[title])
    # 在y=0处添加一条水平线
    plt.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    # 设置Y轴为百分比格式
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    # plt.grid(True, axis='both', linestyle='--')
    plt.xticks(positions, exp_names, rotation=45, ha="right", fontproperties='Times New Roman')
    plt.yticks(fontproperties='Times New Roman')
    plt.xlabel(get_display(argv["i18n"], "Tricks"))
    plt.ylabel(get_display(argv["i18n"], "Robustness Change Rate"))
    plt.title(title, fontproperties='Times New Roman')

    # 创建图例
    legend_elements = [patches.Patch(facecolor=macaron_colors_1[idx], label=metric) for idx, metric in enumerate(metrics)]
    plt.legend(handles=legend_elements, loc='upper right', prop={'family': 'Times New Roman'})

    plt.tight_layout()

    # 保存图表到文件
    save_dir = os.path.dirname(figurename)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(figurename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved to {figurename}")