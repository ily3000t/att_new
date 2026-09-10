import os


def get_env_scenario_algos(args):
    # 往下walk三级目录，返回一个(env, scenario, algo)三元组的列表
    envs = []
    for env_name in os.listdir(os.path.join(args.out, "data")):
        if not os.path.isdir(os.path.join(args.out, "data", env_name)):
            continue
        if args.env is not None and args.env != "None" and args.env != env_name:
            continue

        for scenario_name in os.listdir(os.path.join(args.out, "data", env_name)):
            if not os.path.isdir(os.path.join(args.out, "data", env_name, scenario_name)):
                continue
            if args.scenario is not None and args.scenario != "None" and args.scenario != scenario_name:
                continue

            for algo_name in os.listdir(os.path.join(args.out, "data", env_name, scenario_name)):
                if not os.path.isdir(os.path.join(args.out, "data", env_name, scenario_name, algo_name)):
                    continue
                if args.algo is not None and args.algo != "None" and args.algo != algo_name:
                    continue

                envs.append((env_name, scenario_name, algo_name))
    return envs


def get_esa_via_results(args, result_path):
    # 往下walk三级目录，返回一个(env, scenario, algo)三元组的列表
    esa = []
    for env_name in os.listdir(os.path.join(result_path)):
        if not os.path.isdir(os.path.join(result_path, env_name)):
            continue
        if args.env is not None and args.env != "None" and args.env != env_name:
            continue

        for scenario_name in os.listdir(os.path.join(result_path, env_name)):
            if not os.path.isdir(os.path.join(result_path, env_name, scenario_name)):
                continue
            if args.scenario is not None and args.scenario != "None" and args.scenario != scenario_name:
                continue

            for algo_name in os.listdir(os.path.join(result_path, env_name, scenario_name, "single")):
                if not os.path.isdir(os.path.join(result_path, env_name, scenario_name, "single", algo_name)):
                    continue
                if args.algo is not None and args.algo != "None" and args.algo != algo_name:
                    continue

                esa.append((env_name, scenario_name, algo_name))
    return esa