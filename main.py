import argparse
import json
import os

from amb.utils.config_utils import get_one_yaml_args, parse_timestep, update_args


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=["train", "eval", "render"],
        help="mode: train, eval, render",
    )
    parser.add_argument(
        "--env",
        type=str,
        default="pettingzoo_mpe",
        choices=[
            "smac",
            "mamujoco",
            "pettingzoo_mpe",
            "gym",
            "football",
            "smacv2",
            "toy",
            "metadrive",
            "quads",
            "dexhands",
            "network",
            "voltage",
        ],
        help="Environment name. Choose from: smac, mamujoco, pettingzoo_mpe, gym, football, smacv2.",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="mappo",
        choices=[
            "maddpg",
            "mappo",
            "qmix",
            "vdn",
            "iql",
            "qtran",
            "coma",
        ],
        help="Algorithm name. Choose from: maddpg, mappo, igs.",
    )
    parser.add_argument("--exp_name", type=str, default="installtest", help="Experiment name.")
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    parser.add_argument(
        "--attack",
        type=str,
        default="",
        choices=[
            "random_noise",
            "iterative_perturbation",
            "adaptive_action",
            "random_policy",
            "traitor",
        ],
        help="Attack method name. Choose from: random_noise, iterative_perturbation, adaptive_action, random_policy, traitor.",
    )
    parser.add_argument(
        "--attack_algo",
        type=str,
        default="mappo",
        choices=["mappo", "maddpg", "qmix"],
        help="Attack method name. Choose from: random_noise, iterative_perturbation, adaptive_action, random_policy, traitor.",
    )
    parser.add_argument(
        "--victim",
        type=str,
        default="",
        choices=["maddpg", "mappo", "qmix"],
        help="Victim algorithm name. Choose from: maddpg, mappo, qmix.",
    )
    parser.add_argument(
        "--load_victim",
        type=str,
        default="",
        help="If set, load existing victim config file and checkpoint file instead of reading from yaml config file.",
    )
    parser.add_argument(
        "--load_external",
        type=str,
        default="",
        help="If set, load external agent with checkpoint.",
    )
    parser.add_argument(
        "--black_query",
        type=str,
        default="",
        help="If set, load external agent by blackbox query api.",
    )

    args, unparsed_args = parser.parse_known_args()

    def process(arg):
        try:
            return eval(arg)
        except:
            return arg

    keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
    values = [process(v) for v in unparsed_args[1::2]]
    unparsed_dict = {k: v for k, v in zip(keys, values)}
    args = vars(args)  # convert to dict

    if args["mode"] == "train":
        if args["attack"] != "":
            raise ValueError("You should not use attack method in train mode.")

    if args["mode"] == "eval" or args["mode"] == "render":
        if args["load_victim"] == "" and args["load_external"] == "" and args["black_query"] == "":
            raise ValueError("You should specify load_victim or load_external when use eval.")

    if args["load_config"] != "":  # load config from existing config file
        with open(args["load_config"], encoding="utf-8") as file:
            all_config = json.load(file)
        args["algo"] = all_config["main_args"]["algo"]
        args["env"] = all_config["main_args"]["env"]
        args["run"] = all_config["main_args"]["run"]
        args["exp_name"] = (
            all_config["main_args"]["exp_name"]
            if "exp_name" in all_config["main_args"] and all_config["main_args"]["exp_name"] != ""
            else args["exp_name"]
        )

        algo_args = all_config["algo_args"]["train"]
        victim_args = all_config["algo_args"]["victim"]
        env_args = all_config["env_args"]

    else:  # load config from corresponding yaml file
        if args["mode"] == "eval" or args["mode"] == "render":
            if args["load_victim"] != "":
                with open(os.path.join(args["load_victim"], "config.json"), encoding="utf-8") as file:
                    victim_config = json.load(file)
                args["victim"] = victim_config["main_args"]["algo"]
                args["env"] = victim_config["main_args"]["env"]
                victim_config["algo_args"]["train"]["model_dir"] = os.path.join(args["load_victim"], "models")

                victim_args = victim_config["algo_args"]["train"]
                env_args = victim_config["env_args"]

                if args["attack"] == "":  # 有load_victim，但是没有attack，说明是eval vanilla
                    algo_args = victim_config["algo_args"]["train"]

                else:  # 有load_victim，也有attack，说明是eval attack
                    # attack algo的配置从默认文件中读取
                    algo_args = get_one_yaml_args(args["attack_algo"] + "_traitor")

            elif args["load_external"] != "":
                algo_args = get_one_yaml_args(args["algo"])
                env_args = get_one_yaml_args(args["env"], type="env")
                victim_args = algo_args

                if args["attack"] != "":
                    algo_args = get_one_yaml_args(args["attack_algo"] + "_traitor")

            elif args["black_query"] != "":
                algo_args = {}
                env_args = get_one_yaml_args(args["env"], type="env")
                victim_args = {}
                if args["attack"] != "":
                    algo_args = get_one_yaml_args(args["attack_algo"] + "_traitor")


        else:  # train, 没有load_config的情况下，从默认文件中读取
            algo_args = get_one_yaml_args(args["algo"])
            env_args = get_one_yaml_args(args["env"], type="env")
            victim_args = {}

    update_args(unparsed_dict, algo=algo_args, env=env_args, victim=victim_args)  # update args from command line
    algo_args = {"train": algo_args, "victim": victim_args}

    # envs
    if args["env"] == "dexhands":
        import isaacgym

        algo_args["train"]["episode_length"] = env_args["hands_episode_length"]

    # attack
    if "perturb_timesteps" in algo_args["train"]:
        algo_args["train"]["perturb_timesteps"] = parse_timestep(
            algo_args["train"]["perturb_timesteps"], algo_args["train"]["episode_length"]
        )

    if args["attack"] == "random_noise":
        # --run perturbation --algo.num_env_steps 0 --algo.perturb_iters 0 --algo.adaptive_alpha False --algo.targeted_attack False
        args["run"] = "perturbation"
        algo_args["train"]["num_env_steps"] = 0
        algo_args["train"]["perturb_iters"] = 0
        algo_args["train"]["adaptive_alpha"] = False
        algo_args["train"]["targeted_attack"] = False
        args["exp_name"] += "_random_noise"

    elif args["attack"] == "iterative_perturbation":
        # --run perturbation --algo.num_env_steps 0  --algo.perturb_iters 10 --algo.adaptive_alpha True --algo.targeted_attack False
        args["run"] = "perturbation"
        algo_args["train"]["num_env_steps"] = 0
        algo_args["train"]["perturb_iters"] = (
            algo_args["train"]["perturb_iters"] if algo_args["train"]["perturb_iters"] else 10
        )
        algo_args["train"]["adaptive_alpha"] = True
        algo_args["train"]["targeted_attack"] = False
        args["exp_name"] += "_iterative_perturbation"

    elif args["attack"] == "adaptive_action":
        # --run perturbation --algo.num_env_steps 5000000 --algo.perturb_iters 10 --algo.adaptive_alpha True --algo.targeted_attack True
        args["run"] = "perturbation"
        algo_args["train"]["num_env_steps"] = (
            algo_args["train"]["num_env_steps"] if algo_args["train"]["num_env_steps"] else 5000000
        )
        algo_args["train"]["perturb_iters"] = (
            algo_args["train"]["perturb_iters"] if algo_args["train"]["perturb_iters"] else 10
        )
        algo_args["train"]["adaptive_alpha"] = True
        algo_args["train"]["targeted_attack"] = True
        args["exp_name"] += "_adaptive_action"

    elif args["attack"] == "random_policy":
        # --run traitor --algo.num_env_steps 0
        args["run"] = "traitor"
        algo_args["train"]["num_env_steps"] = 0
        args["exp_name"] += "_random_policy"

    elif args["attack"] == "traitor":
        # --run traitor --algo.num_env_steps 5000000
        args["run"] = "traitor"
        algo_args["train"]["num_env_steps"] = (
            algo_args["train"]["num_env_steps"] if algo_args["train"]["num_env_steps"] else 5000000
        )
        args["exp_name"] += "_traitor"

    # mode
    if args["mode"] == "train":
        args["run"] = "single"
        algo_args["train"]["use_render"] = False
        if args["env"] == "dexhands" or args["env"] == "metadrive":
            algo_args["train"]["use_eval"] = False

    elif args["mode"] == "eval":
        algo_args["train"]["use_eval"] = True
        if args["attack"] == "":
            args["run"] = "single"
            algo_args["train"]["num_env_steps"] = 0

    elif args["mode"] == "render":
        algo_args["train"]["use_eval"] = True
        algo_args["train"]["use_render"] = True

    if args["load_external"] != "":
        from amb.runners.evaluate.external_runner import ExternalRunner

        runner = ExternalRunner(args, algo_args, env_args)
    elif args["black_query"] != "":
        from amb.runners.evaluate.blackbox_runner import BlackboxRunner

        runner = BlackboxRunner(args, algo_args, env_args)
    else:
        from amb.runners import get_single_runner

        runner = get_single_runner(args["run"], args["algo"])(args, algo_args, env_args)

    if algo_args["train"]["use_render"]:  # render, not train
        runner.render()
    else:
        runner.run()

    runner.close()
