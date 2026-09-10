# 此方法的功能是从实验日志目录中把实验结果导出到csv文件中
# 用法：python export.py <实验日志目录> <导出csv文件名>
# 例如：python export.py ./results/smac/2s3z ./logs/smac_2s3z_results.csv
# 导出的csv文件中包含的列有：实验时间、实验名称、实验参数、实验结果

import argparse
import json
import os
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from utils.path import get_esa_via_results
from utils.analysis.es_correlation import cal_kendalltau_correlation
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ATTACKS = [
    # --perturbation--
    "random_noise",
    "random_noise_all",
    "iterative_perturbation",
    "adaptive_action",
    "pert_obs_all",
    "pert_obs_sin",
    "pert_act",
    "learn_obs_all",
    "learn_obs_sin",
    "env_random",
    # --traitor-- 
    "pert_act_all",
    "learn_act",
    "learn_act_all",
    "random_policy",
    "random_policy_all",
    "traitor",
]

# 读取scheme.json
SCHEME_CFG = json.load(open(os.path.abspath(os.path.join(os.path.dirname(__file__), "settings/scheme.json")), "r"))

def export_results(env, scenario, algo, attack, data_dir, trick_cfg, result_path, trick):
    # 构建数据输出目录，如果没有则创建
    csv_file = os.path.join(data_dir, env, scenario, algo, f"{attack}.csv")
    if not os.path.exists(os.path.dirname(csv_file)):
        os.makedirs(os.path.dirname(csv_file))

    # 使用dataframe来存储数据
    df = pd.DataFrame(columns=["env", "scenario", "algo", "attack", "scheme", "tag", "exp_name", "before_reward", "vanilla_reward", "adv_reward", "resilience_reward", "vanilla_win_rate", "adv_win_rate"])
    # 先导出默认配置的实验结果
    _record_row(df, env, scenario, algo, attack, "default", result_path)
    if trick==None: # 正常导出
        # tricks配置文件
        tricks_path = trick_cfg
        # 读取tricks.json
        with open(tricks_path, "r") as f:
            TRICKS = json.load(f)
            # 按照ATTACKS和TRICKS的顺序，组织csv文件的列名
            for key, value in TRICKS[algo].items():
                exp_name = ""
                if isinstance(value, dict):  # 如果是dict，则特殊处理，组合命令
                    exp_name = f"{key}"
                    _record_row(df, env, scenario, algo, attack, exp_name, result_path)
                elif isinstance(value, list):  # 如果是list，则遍历list生成命令
                    for v in value:
                        if isinstance(v, list):
                            # 原样转成字符串
                            v = '"' + str(v) + '"'
                        v = (
                            str(v)
                            .replace('"', "")
                            .replace("[", "")
                            .replace("]", "")
                            .replace(",", "_")
                            .replace(" ", "")
                        )
                        exp_name = f"{key}_{v}"
                        _record_row(df, env, scenario, algo, attack, exp_name, result_path)
    else:   # 导出top3_trick生成的trick
        exp_name = trick
        _record_row(df, env, scenario, algo, attack, exp_name, result_path)
    # 计算指标
    print(df)
    #dfn = _calculate_metrics(df)
    dfn = df
    # 保存到csv文件的sheet2中
    dfn.to_csv(csv_file, index=False)

    # 再保存一份用于latex中显示的excel文件
    display_path = os.path.join(data_dir, "latex", "metric", f"{env}_{scenario}_{algo}_{attack}.csv")
    if not os.path.exists(os.path.dirname(display_path)):
        os.makedirs(os.path.dirname(display_path))
    # 只保留dfn的exp_name, vanilla_reward, adv_reward, SRR, rSRR, TPR, TRR列，并且针对SRR, rSRR, TPR, TRR列的数据乘100后保留两位小数
    dfn_display = dfn[["exp_name", "vanilla_reward", "adv_reward"]]#, "SRR", "rSRR", "TPR", "TRR"]]
    dfn_display["vanilla_reward"] = dfn_display["vanilla_reward"].apply(lambda x: round(x, 2) if x is not None else None)
    dfn_display["adv_reward"] = dfn_display["adv_reward"].apply(lambda x: round(x, 2) if x is not None else None)
    # dfn_display["SRR"] = dfn_display["SRR"].apply(lambda x: round(x * 100, 2) if x is not None else None)
    # dfn_display["rSRR"] = dfn_display["rSRR"].apply(lambda x: round(x * 100, 2) if x is not None else None)
    # dfn_display["TPR"] = dfn_display["TPR"].apply(lambda x: round(x * 100, 2) if x is not None else None)
    # dfn_display["TRR"] = dfn_display["TRR"].apply(lambda x: round(x * 100, 2) if x is not None else None)
    # 修改表头exp_name为实现细节，vanilla_reward为\(r\)，adv_reward为\(r^A\)，SRR为SRR(\%)，rSRR为rSRR(\%)，TPR为TPR(\%)，TRR为TRR(\%)
    dfn_display.columns = ["实现细节", r"\(r\)", r"\(r^A\)"]#, "SRR(\%)", "rSRR(\%)", "TPR(\%)", "TRR(\%)"]

    dfn_display.to_csv(display_path, index=False)

def export_early_stopping_results(env, scenario, algo, attack, data_dir, result_path):
    # 构建数据输出目录，如果没有则创建
    csv_file = os.path.join(data_dir, env, scenario, algo, "earlystopping", f"{attack}.csv")
    if not os.path.exists(os.path.dirname(csv_file)):
        os.makedirs(os.path.dirname(csv_file))
    # 使用dataframe来存储数据
    df = pd.DataFrame(columns=["env", "scenario", "algo", "attack", "scheme", "tag", "exp_name", "before_reward", "vanilla_reward", "adv_reward", "resilience_reward", "vanilla_win_rate", "adv_win_rate"])
    # 导出early stopping结果
    _record_row(df, env, scenario, algo, attack, "default", result_path, type="earlystopping")
    # 计算指标
    # dfn = _calculate_metrics(df)
    dfn = df
    # 保存到csv文件的sheet2中
    print(dfn)
    dfn.to_csv(csv_file, index=False)

def _record_row(df, env, scenario, algo, attack, exp_name, result_path, type="tricks"):
    #  如果trail_name中含有["random_noise", "iterative_perturbation", "adaptive_action"]
    #  则从perturbation目录中导出
    #  否则从traitor目录中导出
    if attack in ["random_noise", "random_noise_all",  "iterative_perturbation", "adaptive_action", "pert_obs_sin", "pert_obs_all", "learn_obs_all", "learn_obs_sin", "env_random"]:
        log_dir = os.path.join(
            result_path,
            env,
            scenario,
            "perturbation",
            "mappo-" + algo,
            f"{attack}_{exp_name}_res",
        )
    elif attack in ["random_policy", "random_policy_all", "traitor", "learn_act", "learn_act_all", "pert_act", "pert_act_all"]:
        log_dir = os.path.join(
            result_path, env, scenario, "traitor", "mappo-" + algo, f"{attack}_{exp_name}_res"
        )
    else:
        return

    # 从SCHEME_CFG的tricks得到exp_name对应的tag
    tag = next((value for key, value in SCHEME_CFG["params"].items() if key in exp_name), "Unknown")
    scheme = tag[0]

    # 判断log_dir是否存在，如果不存在，则插入一条空数据
    if not os.path.exists(log_dir) or len(os.listdir(log_dir)) == 0:
        # df增加一行
        df.loc[len(df)] = [env, scenario, algo, attack, scheme, tag, exp_name, None, None, None, None, None, None]
        return
    
    # 默认导出log_dir下最新的实验结果
    date_dir = sorted(os.listdir(log_dir))[-1]
    # 如果不存在result.txt，则插入一条空数据
    if "result.txt" not in os.listdir(os.path.join(log_dir, date_dir)):
        # df增加一行
        df.loc[len(df)] = [env, scenario, algo, attack, scheme, tag, exp_name, None, None, None, None, None, None]
        return
    # 读取date_dir下的result.txt
    with open(os.path.join(log_dir, date_dir, "result.txt"), "r") as f:
        if type == "tricks":
            # df增加一行
            df.loc[len(df)] = [env, scenario, algo, attack, scheme, tag, exp_name, None, None, None, None, None, None]
            # before_reward的值去查询训练日志，取timestep=0的reward，即训练前的reward
            train_log = os.path.join(result_path, env, scenario, "single", algo, f"{exp_name}")
            latest_log = sorted(os.listdir(train_log))[-1]
            with open(os.path.join(train_log, latest_log, "progress.txt"), "r") as f2:
                # 判断如果f2为空，则使用tensorboard数据
                content = f2.readlines()
                if len(content) == 0:
                    # 找到tensorboard数据
                    tb_dir = os.path.join(train_log, latest_log, "logs")
                    print(tb_dir)
                    # 创建EventAccumulator对象以读取TensorBoard日志
                    event_acc = EventAccumulator(tb_dir)
                    event_acc.Reload()
                    tag_values = event_acc.Scalars("env/train_episode_rewards")
                    df.loc[len(df) -1, "before_reward"] = float(tag_values[0].value)

                else:
                    for line in content:
                        arr = line.replace("\n", "").split(",")
                        if arr[0] == "0":
                            df.loc[len(df) -1, "before_reward"] = float(arr[1])
                            break

            for line in f.readlines():
                # 如果为空行，则跳过
                if line == "\n":
                    continue
                arr = line.replace("\n", "").split(",")
                if arr[2] == "final":
                    df.loc[len(df) -1, arr[1] + "_reward"] = float(arr[3])
                    if len(arr) == 5:
                        df.loc[len(df) -1, arr[1] + "_win_rate"] = float(arr[4])

        elif type == "earlystopping":
            for line in f.readlines():
                # 如果为空行，则跳过
                if line == "\n":
                    continue
                arr = line.replace("\n", "").split(",")
                if arr[1] == "vanilla":
                    # df增加一行
                    if arr[2] == "final":
                        df.loc[len(df)] = [env, scenario, algo, attack, scheme, tag, "default", None, None, None, None, None, None]
                    else:
                        df.loc[len(df)] = [env, scenario, algo, attack, scheme, tag, arr[2], None, None, None, None, None, None]
                    # before_reward的值去查询训练日志，取timestep=0的reward，即训练前的reward
                    train_log = os.path.join(result_path, env, scenario, "single", algo, f"{exp_name}")
                    latest_log = sorted(os.listdir(train_log))[-1]
                    with open(os.path.join(train_log, latest_log, "progress.txt"), "r") as f2:
                        # 判断如果f2为空，则使用tensorboard数据
                        content = f2.readlines()
                        if len(content) == 0:
                            # 找到tensorboard数据
                            tb_dir = os.path.join(train_log, latest_log, "logs")
                            # 创建EventAccumulator对象以读取TensorBoard日志
                            event_acc = EventAccumulator(tb_dir)
                            event_acc.Reload()
                            tag_values = event_acc.Scalars("env/train_episode_rewards")
                            df.loc[len(df) -1, "before_reward"] = float(tag_values[0].value)

                        else:
                            for line in content:
                                ar = line.replace("\n", "").split(",")
                                if ar[0] == "0":
                                    df.loc[len(df) -1, "before_reward"] = float(ar[1])
                                    break

                df.loc[len(df) -1, arr[1] + "_reward"] = float(arr[3])
                if len(arr) == 5:
                    df.loc[len(df) -1, arr[1] + "_win_rate"] = float(arr[4])

def _calculate_metrics(df):
    # 先抓出default行的数据
    df_default = df[df["exp_name"] == "default"]
    baseline_r = df_default["vanilla_reward"].values[0] 
    baseline_ra = df_default["adv_reward"].values[0]
    # print(df_default["resilience_reward"])
    # print(df_default["resilience_reward"].values)
    print(f"检查 df['exp_name'] 的所有唯一值{df['exp_name'].unique()}")
    print(df_default)
    baseline_rsa = df_default["resilience_reward"].values[0]
    baseline_w = df_default["vanilla_win_rate"].values[0]
    baseline_range = df_default["vanilla_reward"].values[0] - df_default["before_reward"].values[0]

    # 模型自身reward变化率，符号代表方向，数值代表变化率
    df["SRR"] = (df["adv_reward"] - df["vanilla_reward"]) / (df["vanilla_reward"] - df["before_reward"])
    df["rSRR"] = df["SRR"] - df[df["exp_name"] == "default"]["SRR"].values[0]
    df["TPR"] = (df["vanilla_reward"] - baseline_r) / baseline_range
    df["TRR"] = (df["adv_reward"] - baseline_ra) / baseline_range
    df["CR"] = 0.5 * (df["TPR"] + df["TRR"])

    df["wr-SRR"] = df["adv_win_rate"] - df["vanilla_win_rate"]
    df["wr-TPR"] = df["vanilla_win_rate"] - baseline_w
    df["wr-TRR"] = df["adv_win_rate"] - baseline_w

    # 处理异常数据，例如rnn完全没有训好的数据，应该置空
    # 异常数据1. vanilla_reward < before_reward
    df.loc[df["vanilla_reward"] < df["before_reward"], "SRR"] = 0
    df.loc[df["vanilla_reward"] < df["before_reward"], "rSRR"] = 0
    df.loc[df["vanilla_reward"] < df["before_reward"], "TPR"] = 0
    df.loc[df["vanilla_reward"] < df["before_reward"], "TRR"] = 0
    df.loc[df["vanilla_reward"] < df["before_reward"], "CR"] = 0
    df.loc[df["vanilla_reward"] < df["before_reward"], "wr-SRR"] = 0
    df.loc[df["vanilla_reward"] < df["before_reward"], "wr-TPR"] = 0
    df.loc[df["vanilla_reward"] < df["before_reward"], "wr-TRR"] = 0

    # 调整列顺序，将vanilla_win_rate和adv_win_rate列移动到TRR后面
    columns = list(df.columns)
    columns.remove("vanilla_win_rate")
    columns.remove("adv_win_rate")
    columns = columns[:-3] + ["vanilla_win_rate", "adv_win_rate"] + columns[-3:]
    return df[columns]


def combine_exported_csv(env, scenario, algo, data_dir, trick):
    dir_path = os.path.join(data_dir, env, scenario, algo)
    # 先将多个csv文件合并成一个，每个csv文件对应一个sheet
    excel_path = os.path.join(dir_path, f"{env}_{scenario}_{algo}_tricks_res.xlsx") if trick==None else os.path.join(dir_path, f"{env}_{scenario}_{algo}_{trick}.xlsx")
    
    # 合并tricks
    _combine_csvs(dir_path, excel_path, data_dir)
    
    # 遍历dir_path下的所有文件夹，将文件加内的csv合并成一个excel
    for f in os.listdir(dir_path):
        if os.path.isdir(os.path.join(dir_path, f)):
            _combine_csvs(os.path.join(dir_path, f), os.path.join(dir_path, f"{env}_{scenario}_{algo}_{f}_res.xlsx"))


def merge_all_data(data_dir):
    # 遍历argv["out"]/data下的所有excel，按最后的算法分组
    excel_paths = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith("_tricks_res.xlsx"):
                # 获取file的文件名
                excel_paths.append(os.path.join(root, file))

    # 输出文件
    out_file = os.path.join(data_dir, "all.csv")
    with open(out_file, "w") as out_f:
        out_f.write("env,scenario,algo,attack,scheme,tag,trick,r0,r,ra,rsa,SRR,rSRR,TPR,TRR,CR,w,wa,wSRR,wTPR,wTRR\n")
        for excel_path in excel_paths:
            # 读取excel文件
            df = pd.read_excel(excel_path, sheet_name=None)
            for _, sheet in df.items():
                # 将df["attack"]列的数据中的_替换为空格
                sheet["attack"].replace("_", " ", regex=True, inplace=True)
                # 将df["trick"]列的数据中的_替换为\_
                sheet["exp_name"].replace("_", "\_", regex=True, inplace=True)
                # 将sheet的数据追加写入out_file
                sheet.to_csv(out_f, header=False, index=False)


def _combine_csvs(dir_path, excel_path, data_dir=None):
    file_names = [file for file in os.listdir(dir_path) if file.endswith(".csv")]
    file_names.remove('resilience_finetune.csv') if 'resilience_finetune.csv' in file_names else file_names
    if len(file_names) == 0:
        return
    # file_names按照 random_noise， iterative_perturbation， adaptive_action， random_policy， traitor 排序
    file_names.sort(key=lambda x: ATTACKS.index(x.split(".")[0]))
    # 使用xlsxwriter引擎，可以写入多个sheet
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # 初始化一个空的df
        df_display = pd.DataFrame()
        # 遍历file_names
        for idx, file_name in enumerate(file_names):
            attack = file_name.split(".")[0]
            # 读取CSV文件
            df = pd.read_csv(os.path.join(dir_path, file_name), header=0, index_col=0)
            
            if idx == 0:
                # df的exp_name, vanilla_reward列赋值给df_display的exp_name, vanilla_reward列
                df_display["exp_name"] = df["exp_name"]
                df_display["vanilla_reward"] = df["vanilla_reward"].apply(lambda x: round(x, 2))
                df_display["resilience_reward"] = df['resilience_reward'].apply(lambda x: round(x, 2))
            
            # df的adv_reward列赋值给df_display的adv_reward列
            df_display[attack] = df["adv_reward"].apply(lambda x: round(x, 2))
            
            # 将DataFrame写入不同的sheet
            df.to_excel(writer, sheet_name=f'{attack}', index=True)
            print("Experiments results exported to", excel_path)
            # 删除原来的csv文件
            os.remove(os.path.join(dir_path, file_name))
            # 判断原来的csv文件所在的目录是否为空，如果为空，则删除该目录
            if len(os.listdir(dir_path)) == 0:
                os.rmdir(dir_path)

        # 将df_display写入一个新的csv文件
        # if data_dir is not None:
            # df_display.columns = ["实现细节", "原始奖励", "随机噪声", "最优动作抑制", "自适应动作", "随机策略", "内鬼"]
            # display_path = os.path.join(data_dir, "latex", "algo", f"{env}_{scenario}_{algo}.csv")
            # if not os.path.exists(os.path.dirname(display_path)):
            #     os.makedirs(os.path.dirname(display_path))
            # df_display.to_csv(display_path, index=False)
            # print("Experiments results exported to", display_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", type=str, default=None, help="env name")
    parser.add_argument("-s", "--scenario", type=str, default=None, help="scenario or map name")
    parser.add_argument("-a", "--algo", type=str, default=None, help="algo name")
    parser.add_argument("-o", "--out", type=str, default="out", help="out dir")
    parser.add_argument("-t", "--trick", type=str, default=None)
    parser.add_argument("--result_path", type=str, default="results", help="result path")
    parser.add_argument("--trick_cfg", type=str, default="settings/tricks.json", help="trick config path")
    parser.add_argument("--attack", type=str, default=None, help="attack method list")
    args, _ = parser.parse_known_args()

    data_dir = os.path.join(args.out, "data")

    for env, scenario, algo in get_esa_via_results(args, args.result_path):
        args.env = env
        args.scenario = scenario
        args.algo = algo

        if args.attack is not None:
            ATTACKS = [args.attack]
        for method in ATTACKS:
            export_results(args.env, args.scenario, args.algo, method, data_dir, args.trick_cfg, args.result_path, args.trick)
            # early stopping的实验结果单独导出
            if args.trick==None:
                export_early_stopping_results(args.env, args.scenario, args.algo, method, data_dir, args.result_path)
            else:
                continue

        # combine all csv files
        excel_path = combine_exported_csv(args.env, args.scenario, args.algo, data_dir, args.trick)

        # 计算early stopping的Kendall Tau相关系数
        # excel_path = os.path.join(
        #     data_dir,
        #     args.env,
        #     args.scenario,
        #     args.algo,
        #     f"{args.env}_{args.scenario}_{args.algo}_earlystopping_res.xlsx",
        # )
        
        # 导出early stopping的Kendall Tau相关系数
        # cal_kendalltau_correlation(excel_path, args, True)
    
    # 合并所有数据到一个csv中
    merge_all_data(data_dir)
