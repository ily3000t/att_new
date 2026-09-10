import argparse
import json
import os

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
    args.extra = "" if not hasattr(args, "extra") else args.extra
    args.out = "./out" if not hasattr(args, "out") else args.out



def generate_train_scripts(args):
    # 预处理args
    pre_process_args(args)

    out_dir = args.out
    env = args.env
    scenario = args.scenario
    algo = args.algo
    config_path = args.config_path
    trick_config_path = args.trick_config_path
    gpu = args.gpu
    extra = args.extra

    logs_dir = os.path.join(out_dir, "logs", env, scenario, algo)
    settings_dir = os.path.join("settings", env, scenario)

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(settings_dir, exist_ok=True)

    # baseline配置文件
    config_path = os.path.join(settings_dir, algo + ".json") if config_path is None else config_path
    config_path = os.path.abspath(config_path)
    # tricks配置文件
    tricks_path = os.path.join("settings", "tricks.json") if trick_config_path is None else trick_config_path
    tricks_path = os.path.abspath(tricks_path)
    # 读取tricks.json
    with open(tricks_path, "r") as f:
        TRICKS = json.load(f)
    
    resilience_reward_path = os.path.join(out_dir, 'data', env, scenario, algo, "resilience_finetune.csv")
    resilience_reward = {}
    
    def insert_reward_to_dict(dict_trick, attack_method, coop_rew, adv_rew, resi_rew):
        dict_trick['coop_rewards'].append(coop_rew)
        if attack_method in dict_trick['adv_rewards']:
            dict_trick['adv_rewards'][attack_method].append(adv_rew)
        else:
            dict_trick['adv_rewards'][attack_method] = [adv_rew]
        if attack_method in dict_trick['resi_rewards']:
            dict_trick['resi_rewards'][attack_method].append(resi_rew)
        else:
            dict_trick['resi_rewards'][attack_method] = [resi_rew]
            
    def get_total_rewards(dict_trick):
        coop_rew_mean = sum(dict_trick['coop_rewards']) / len(dict_trick['coop_rewards'])
        adv_rew_mean = []
        for reward_list in dict_trick['adv_rewards'].values():
            adv_rew_mean.append(sum(reward_list) / len(reward_list))
        resi_rew_mean = []
        for reward_list in dict_trick['resi_rewards'].values():
            resi_rew_mean.append(sum(reward_list) / len(reward_list))
        return coop_rew_mean + sum(adv_rew_mean) + sum(resi_rew_mean)
    
    with open(resilience_reward_path, 'r', encoding='utf-8') as file:
        for line in file:
            units = line.strip().split(',')
            if len(units) != 5:
                continue
            if units[0] == 'trick':
                continue
            coop_rew = float(units[2])
            adv_rew = float(units[3])
            resi_rew = float(units[4])
            if units[0] not in resilience_reward:
                resilience_reward[units[0]] = {'coop_rewards': [], 'adv_rewards': {}, 'resi_rewards': {}}
            insert_reward_to_dict(resilience_reward[units[0]], units[1], coop_rew, adv_rew, resi_rew)
    
    resilience_reward = [(key, get_total_rewards(value)) for key, value in resilience_reward.items()]
    sorted_resilience_reward = sorted(resilience_reward, key=lambda x: x[1], reverse=True)
    
    trick_str_trick_dict = {}
    for key, value in TRICKS[algo].items():
        trick_str = ""
        trick_name = ""
        if isinstance(value, dict):
            trick_name = f'{key}'
            for k, v in value.items():
                trick_str += f" --algo.{k} {v}"
            trick_str_trick_dict[trick_name] = trick_str
        elif isinstance(value, list):
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
                trick_name = f"{key}_{v}"
                trick_str_trick_dict[trick_name] = trick_str
    
    trick_str_top3 = ""
    trick_name_top3 = []
    trick_name_top3_c =[]
    for trick_name, _ in sorted_resilience_reward:
        if trick_name == 'default':
            break
        assert trick_name in trick_str_trick_dict, f'{trick_name} is not valid!'
        trick_str = trick_str_trick_dict[trick_name]
        if trick_str.strip().split()[0] not in trick_name_top3_c:
            trick_name_top3.append(trick_name)
            trick_str_top3 += trick_str + ' '
            trick_name_top3_c.append(trick_str.strip().split()[0])
        if len(trick_name_top3) == 3:
            break
    exp_name = "_".join(trick_name_top3) if len(trick_name_top3) > 0 else 'default'
    ld = os.path.join(logs_dir, exp_name)
    os.makedirs(ld, exist_ok=True)
    command = f"CUDA_VISIBLE_DEVICES={gpu} python -u {TRIAN_ENTRY_PATH} --load_config {config_path} --exp_name {exp_name} {trick_str_top3} {'--headless' if extra=='headless' else extra if extra else ''} > {ld}/train.log 2>&1"
    print(command)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--env", type=str, default="smac", help="env name")
    parser.add_argument(
        "-s", "--scenario", type=str, default="2s3z", help="scenario or map name"
    )
    parser.add_argument("-a", "--algo", type=str, default="mappo", help="algo name")
    parser.add_argument("-o", "--out", type=str, default="out", help="out dir")
    parser.add_argument("--config_path", type=str, default=None, help="default config path")
    parser.add_argument("--trick_config_path", type=str, default=None, help="default trick config path")
    parser.add_argument("-g", "--gpu", type=int, default=0, help="specify gpu number")
    parser.add_argument("--extra", type=str, default="", help="extra parameters at the end of command, like `--headless` in dexhands.")
    args, _ = parser.parse_known_args()
    generate_train_scripts(args)
