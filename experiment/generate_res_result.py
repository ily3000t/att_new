import argparse
import json
import os

ATTACK_CONF = {
    "random_noise": "--run perturbation --algo.num_env_steps 0 --algo.perturb_iters 0 --algo.adaptive_alpha False --algo.targeted_attack False",
    #"iterative_perturbation": "--run perturbation --algo.num_env_steps 0  --algo.perturb_iters 10 --algo.adaptive_alpha True --algo.targeted_attack False",
    "random_policy": "--run traitor --algo.num_env_steps 0",
    
    "pert_act": "--algo.use_eval True --run traitor --algo.num_env_steps 0  --algo.perturb_iters 0 --algo.adaptive_alpha False --algo.targeted_attack False --perturbation_eps 0.2",
    "pert_act_all": "--algo.use_eval True --run traitor --algo.num_env_steps 0  --algo.perturb_iters 0 --algo.adaptive_alpha False --algo.targeted_attack False --perturbation_eps 0.1 --adv_all True",
    "random_noise_all": "--run perturbation --algo.num_env_steps 0 --algo.perturb_iters 0 --algo.adaptive_alpha False --algo.targeted_attack False, --adv_all True",
    "pert_obs_all": "--algo.use_eval True --run perturbation --algo.num_env_steps 0  --algo.perturb_iters 10 --algo.adaptive_alpha True --algo.targeted_attack False --adv_all True",
    "pert_obs_sin": "--algo.use_eval True --run perturbation --algo.num_env_steps 0  --algo.perturb_iters 10 --algo.adaptive_alpha True --algo.targeted_attack False --adv_all False --adv_eps 0.1",
    "random_policy_all": "--run traitor --algo.num_env_steps 0 --adv_all True", 
    "env_random": "--run perturbation --algo.num_env_steps 0 --algo.perturb_iters 0 --algo.adaptive_alpha False --algo.targeted_attack False --env_randomization True",
    "learn_act_all": "--run traitor --algo.num_env_steps 0 --traitor_eps 0.1 --adv_all True ",
    
    "env_random": "--run perturbation --algo.num_env_steps 0 --algo.perturb_iters 0 --algo.adaptive_alpha False --algo.targeted_attack False --env_randomization True",
    "learn_act": "--algo.use_eval True --run traitor --algo.num_env_steps 0 --traitor_eps 0.2",
    "learn_obs_all": "--algo.use_eval True --run perturbation --algo.num_env_steps 0 --algo.perturb_iters 10 --algo.adaptive_alpha True --algo.targeted_attack True --adv_all True",
    "learn_obs_sin": "--algo.use_eval True --run perturbation --algo.num_env_steps 0 --algo.perturb_iters 10 --algo.adaptive_alpha True --algo.targeted_attack True --adv_all False --adv_eps 0.1",
    #"traitor": "--run traitor --algo.num_env_steps 0",
    #"adaptive_action": "--run perturbation --algo.num_env_steps 0 --algo.perturb_iters 10 --algo.adaptive_alpha True --algo.targeted_attack True",
    "learn_act_all": "--run traitor --algo.num_env_steps 0 --traitor_eps 0.1 --adv_all True ",
    
}

ATTACK_CONF_STAGE_2 = {
    "learn_act": "--algo.use_eval True --run traitor --algo.num_env_steps 0 --traitor_eps 0.2",
    "learn_obs_all": "--algo.use_eval True --run perturbation --algo.num_env_steps 0 --algo.perturb_iters 10 --algo.adaptive_alpha True --algo.targeted_attack True --adv_all True",
    "learn_obs_sin": "--algo.use_eval True --run perturbation --algo.num_env_steps 0 --algo.perturb_iters 10 --algo.adaptive_alpha True --algo.targeted_attack True --adv_all False --adv_eps 0.1",
    "traitor": "--run traitor --algo.num_env_steps 0",
    "adaptive_action": "--run perturbation --algo.num_env_steps 0 --algo.perturb_iters 10 --algo.adaptive_alpha True --algo.targeted_attack True",
}

TRIAN_ENTRY_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "single_train.py")


def pre_process_args(args):
    # 如果config_path存在，则需覆盖env, scenario, algo
    if hasattr(args, "config_path") and args.config_path is not None:
        with open(args.config_path, "r") as f:
            config = json.load(f)
        args.env = config["main_args"]["env"]
        args.algo = config["main_args"]["algo"]
        # 针对scenario需要特殊判断，因为不是所有的env都有scenario
        if "scenario" in config["env_args"]:
            args.scenario = config["env_args"]["scenario"]
            if args.env == "mamujoco":
                args.scenario += "-" + config["env_args"]["agent_conf"]
            elif args.env == "pettingzoo_mpe":
                args.scenario += "-" + ("continuous" if config["env_args"]["continuous_actions"] else "discrete")
        elif "map_name" in config["env_args"]: # smac
            args.scenario = config["env_args"]["map_name"]
        elif "task" in config["env_args"]: # dexhands
            args.scenario = config["env_args"]["task"]
        elif "env_name" in config["env_args"]: # football
            args.scenario = config["env_args"]["env_name"]
        elif "env" in config["env_args"]:
            args.scenario = config["env_args"]["env"]
    
    # 判断args中是否存在gpu, trick, extra等字段
    args.gpu = 0 if not hasattr(args, "gpu") else args.gpu
    args.trick = None if not hasattr(args, "trick") else args.trick
    args.extra = "" if not hasattr(args, "extra") else args.extra
    args.num_workers = 2 if not hasattr(args, "num_workers") else args.num_workers
    args.out = "./out" if not hasattr(args, "out") else args.out
    args.stage = 0 if not hasattr(args, "stage") else args.stage
    args.slice = False if not hasattr(args, "slice") else args.slice



def generate_train_scripts(args):
    # 预处理args
    pre_process_args(args)

    out_dir = args.out
    env = args.env
    scenario = args.scenario
    algo = args.algo
    config_path = args.config_path
    trick_config_path = args.trick_config_path
    trick = args.trick
    num_workers = args.num_workers
    gpu = args.gpu
    extra = args.extra

    # 构建数据输出目录，如果没有则创建
    logs_dir = os.path.join(out_dir, "logs", env, scenario, algo)
    # settings目录
    settings_dir = os.path.join("settings", env, scenario)
    # scripts目录
    scripts_dir = os.path.join(out_dir, "scripts")

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(settings_dir, exist_ok=True)
    os.makedirs(scripts_dir, exist_ok=True)

    # baseline配置文件
    config_path = os.path.join(settings_dir, algo + ".json") if config_path is None else config_path
    config_path = os.path.abspath(config_path)
    # tricks配置文件
    tricks_path = os.path.join("settings", "tricks.json") if trick_config_path is None else trick_config_path
    tricks_path = os.path.abspath(tricks_path)
    # 读取tricks.json
    with open(tricks_path, "r") as f:
        TRICKS = json.load(f)
    # 生成脚本文件
    file_name = os.path.join(scripts_dir, f"train_res_{env}_{scenario}_{algo}.sh")
    with open(file_name, "w") as f:
        # 生成默认命令
        _write_train_command(f, config_path, "default", logs_dir, trick=trick, gpu=gpu, extra=extra)
        # 遍历tricks
        # os.path.join(settings_dir, "tricks.json")
        for key, value in TRICKS[algo].items():
            trick_str = ""
            exp_name = ""
            if isinstance(value, dict):  # 如果是dict，则特殊处理，组合命令
                exp_name = f"{key}"
                # 遍历字典，生成命令
                for k, v in value.items():
                    trick_str += f" --algo.{k} {v}"
                # 生成命令
                _write_train_command(f, config_path, exp_name, logs_dir, trick_str=trick_str, trick=trick, gpu=gpu, extra=extra)
            elif isinstance(value, list):  # 如果是list，则遍历list生成命令
                for v in value:
                    if isinstance(v, list):
                        # 原样转成字符串
                        v = '"' + str(v) + '"' if key != "hidden_sizes" else '"' + str(v).replace(" ", "") + '"'
                    trick_str = f" --algo.{key} {v}"
                    v = (
                        str(v)
                        .replace('"', "")
                        .replace("[", "")
                        .replace("]", "")
                        .replace(",", "_")
                        .replace(" ", "")
                    )
                    exp_name = f"{key}_{v}"
                    # 生成命令
                    _write_train_command(f, config_path, exp_name, logs_dir, trick_str=trick_str, trick=trick, gpu=gpu, extra=extra)

    print(f"python parallel.py -s {file_name} -o {out_dir} -n {num_workers}")

    return file_name


def _write_train_command(file, config_path, exp_name, logs_dir, trick_str="", trick=None, gpu=0, extra=None):
    if trick is not None and trick != exp_name:
        return
    ld = os.path.join(logs_dir, exp_name)
    os.makedirs(ld, exist_ok=True)
    command = f"CUDA_VISIBLE_DEVICES={gpu} python -u {TRIAN_ENTRY_PATH} --load_config {config_path} --exp_name {exp_name} {trick_str} {'--headless' if extra=='headless' else extra if extra else ''} > {ld}/train_res.log 2>&1"
    file.write(command + "\n")

def generate_eval_scripts(args):
     # 预处理args
    pre_process_args(args)
    
    out_dir = args.out
    env = args.env
    scenario = args.scenario
    algo = args.algo
    stage = args.stage
    slice = args.slice
    method = args.method
    trick = args.trick
    num_workers = args.num_workers
    gpu = args.gpu
    extra = args.extra
    attack_config = args.attack_config if hasattr(args, "attack_config") else None
    log_dir = args.log_dir if hasattr(args, "log_dir") else None
    
    recover_resilience_finetune = args.recover_resilience_finetune
    resilience_finetune_dir = os.path.join(out_dir, "data", env, scenario, algo)
    os.makedirs(resilience_finetune_dir, exist_ok=True)
    resilience_finetune_path = os.path.abspath(os.path.join(resilience_finetune_dir, "resilience_finetune.csv"))
    if recover_resilience_finetune:
        write_header = True
        import csv
        if os.path.exists(resilience_finetune_path):
            with open(resilience_finetune_path, 'r', encoding='utf-8') as file:
                reader = csv.reader(file)
                rows = list(reader)
                if len(rows) > 3:
                    write_header = False
        if write_header:        
            with open(resilience_finetune_path, 'a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['trick', 'attack_method', 'cooperate_reward', 'adv_reward', 'resilience_reward'])
        extra += f' --eval_recover_mode True --eval_recover_log_path {resilience_finetune_path} --algo.use_eval True --algo.num_env_steps 0'
            
    # scripts目录
    scripts_dir = os.path.join(out_dir, "scripts")
    os.makedirs(scripts_dir, exist_ok=True)
    models_dir = os.path.join(out_dir, "results", env, scenario, "single", algo)
    # 根据环境不同，生成各自的命令
    # 注意此处algo mappo是用作攻击的，此处设置为固定使用mappo作为攻击RL算法
    base_cfg = f"--algo mappo --env {env}"
    if env == "smac":
        base_cfg += f" --env.map_name {scenario}"
    elif env == "mamujoco":
        ss = scenario.split("-")
        base_cfg += f" --env.scenario {ss[0]} --env.agent_conf {ss[1]}"
    elif env == "pettingzoo_mpe":
        ss = scenario.split("-")
        base_cfg += f" --env.scenario {ss[0]} --env.continuous_actions {True if ss[1] == 'continuous' else False}"
    elif env == "dexhands":
        base_cfg += f" --env.task {scenario} --algo.use_eval True"
    elif env == "network":
        base_cfg += f" --env.scenario {scenario} --algo.use_eval True --algo.episode_length 1510"
    elif env == "quads":
        base_cfg += f" --env.scenario {scenario} --env.conf.quads_mode {scenario} --algo.episode_length 1510"
    elif env == "voltage":
        base_cfg += f" --env.scenario {scenario}"
    else:
        raise NotImplementedError
    if attack_config is not None:
        base_cfg += f" --attack_config {attack_config}"
    if log_dir is not None:
        base_cfg += f" --algo.log_dir {log_dir}"
        models_dir = os.path.join(log_dir, env, scenario, "single", algo)

    # 生成脚本文件
    file_name = os.path.join(scripts_dir, f"eval_res_{env}_{scenario}_{algo}.sh" if stage == 0 else f"eval_res_{env}_{scenario}_{algo}_stage_{stage}.sh")
    with open(file_name, "w") as f:
        # 读取victims_dir目录列表
        victims_dirs = os.listdir(models_dir)
        # 排序,但是default要放在最前
        victims_dirs.remove("default")
        victims_dirs.sort()
        victims_dirs.insert(0, "default")

        victim_tuples = []
        # 遍历victims_dirs
        for victim_dir in victims_dirs:
            # 查看victim_dir是否是目录，如果是，看子目录有几个
            if os.path.isdir(os.path.join(models_dir, victim_dir)):
                # 获取最新的目录
                latest_victim_dir = sorted(os.listdir(os.path.join(models_dir, victim_dir)))[-1]
                victim_tuples.append((victim_dir, latest_victim_dir))
                # # 读取子目录列表
                # sub_victims_dirs = os.listdir(os.path.join(models_dir, victim_dir))
                # # 遍历子目录列表
                # for sub_victim_dir in sub_victims_dirs:
                #     victim_tuples.append((victim_dir, sub_victim_dir))

        for attack_method, attack_cfg in ATTACK_CONF.items() if stage != 2 else ATTACK_CONF_STAGE_2.items():
            if method is not None and method != attack_method:
                continue
            for victim_dir, sub_victim_dir in victim_tuples:
                # 构建数据输出目录，如果没有则创建
                logs_dir = os.path.join(
                    out_dir,
                    "logs",
                    env,
                    scenario,
                    algo,
                    victim_dir,
                )
                if recover_resilience_finetune:
                    extra_this = extra + f" --resilience_log_trick {victim_dir} --resilience_log_attack_method {attack_method}"
                else:
                    extra_this = extra
                os.makedirs(logs_dir, exist_ok=True)

                if stage == 1 and attack_method in ["adaptive_action", "traitor"]:
                    if victim_dir != "default":
                        continue
                    elif env == "dexhands":
                        # 将base_cfg中的--algo.use_eval True替换为--algo.use_eval False
                        base_cfg = base_cfg.replace("--algo.use_eval True", "--algo.use_eval False")

                if attack_method in ["adaptive_action", "traitor", "learn_act", "learn_obs_all", "learn_obs_sin"]:
                    type = "traitor" if attack_method in ["traitor", "learn_act"] else "perturbation"
                    adv_model_dir = os.path.join("results", env, scenario, type, "mappo-" + algo, f"{attack_method}_default")
                    if not os.path.exists(adv_model_dir):
                        continue
                    dir_list = sorted(os.listdir(adv_model_dir), reverse=True)
                    # 从后往前遍历dir_list
                    for i in range(len(dir_list)):
                        latest_models_dir = os.path.join(adv_model_dir, sorted(os.listdir(adv_model_dir))[i], "models")
                        # 判断latest_models_dir下是否存在文件，如果为空，则继续往前找到最后一个非空的文件夹
                        if os.listdir(latest_models_dir):
                            break
                    # 生成命令
                    if trick is not None and trick != victim_dir:
                        continue
                    command = f"CUDA_VISIBLE_DEVICES={gpu} python -u {TRIAN_ENTRY_PATH} {base_cfg} --algo.slice {slice if victim_dir != 'default' else True} --algo.model_dir {latest_models_dir} --load_victim {os.path.join(models_dir, victim_dir, sub_victim_dir)} --exp_name {attack_method}_{victim_dir}_res {attack_cfg} {extra_this if extra_this else ''}"
                    output =  f" > {logs_dir}/{attack_method}_res.log 2>&1"
                    if env == 'dexhands':
                        if '--headless' not in command:
                            command += ' --headless'
                        else:
                            command = ' '.join(command.split('--headless')) + '--headless'
                    command = command + output
                    if victim_dir == "default" and env != "dexhands":
                        # command = "# " + command
                        pass
                    f.write(command + "\n")
                    continue
                # 生成命令
                if trick is not None and trick != victim_dir:
                    continue
                command = f"CUDA_VISIBLE_DEVICES={gpu} python -u {TRIAN_ENTRY_PATH} {base_cfg} --algo.slice {slice if victim_dir != 'default' else True} --load_victim {os.path.join(models_dir, victim_dir, sub_victim_dir)} --exp_name {attack_method}_{victim_dir}_res {attack_cfg} {extra_this if extra_this else ''}"
                output = f" > {logs_dir}/{attack_method}_res.log 2>&1"
                if env == 'dexhands':
                    if '--headless' not in command:
                        command += ' --headless'
                    else:
                        command = ' '.join(command.split('--headless')) + '--headless'
                command = command + output
                f.write(command + "\n")
            f.write("\n")
        f.write("\n")

    print(f"python parallel.py -s {file_name} -o {out_dir} -n {num_workers}")
    if stage == 1:
        #print(f"python generate.py eval -e {env} -s {scenario} -a {algo} --stage 2" + "--extra='--headless'" if extra=='headless' else extra if extra else '')
        pass
    
    return file_name


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        type=str,
        choices=["train", "eval"],
        help="mode: train or eval",
    )
    parser.add_argument("-e", "--env", type=str, default="smac", help="env name")
    parser.add_argument(
        "-s", "--scenario", type=str, default="2s3z", help="scenario or map name"
    )
    parser.add_argument("-a", "--algo", type=str, default="mappo", help="algo name")
    parser.add_argument(
        "--stage",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="stage_0: eval all; stage_one: only eval default model in adaptive_action and traitor; stage_two:load adv model to eval.",
    )
    parser.add_argument("-o", "--out", type=str, default="out", help="out dir")
    parser.add_argument("--slice", action="store_true", help="whether to slice eval")
    parser.add_argument("--config_path", type=str, default=None, help="default config path")
    parser.add_argument("--trick_config_path", type=str, default=None, help="default trick config path")
    parser.add_argument("-t", "--trick", type=str, default=None, help="only generate the specified trick scripts")
    parser.add_argument("-m", "--method", type=str, default=None, help="only generate the specified attack algo scripts")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="specify gpu number")
    parser.add_argument("--log_dir", type=str, default="./results", help="results dir")
    parser.add_argument("--attack_config", type=str, default=None, help="adv algo config path")
    parser.add_argument(
        "-n",
        "--num_workers",
        default=2,
        type=int,
        help="Number of workers to use for parallel execution.",
    )
    parser.add_argument("--extra", type=str, default="", help="extra parameters at the end of command, like `--headless` in dexhands.")
    parser.add_argument("--recover_resilience_finetune", action="store_true", default=True, help="Whether to choose the best tricks by finetuning the resilience results.")
    args, _ = parser.parse_known_args()

    if args.mode == "train":
        args.recover_resilience_finetune = False
        generate_train_scripts(args)
    elif args.mode == "eval":
        generate_eval_scripts(args)
