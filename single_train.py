import os
import argparse
import json

import numpy as np
import yaml
from amb.utils.config_utils import get_one_yaml_args, update_args, parse_timestep
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# import torch
# # show tensor shape in vscode debugger
# def custom_repr(self):
#     return f'{{Tensor:{tuple(self.shape)}}} {original_repr(self)}'

# original_repr = torch.Tensor.__repr__
# torch.Tensor.__repr__ = custom_repr

def main():
    """Main function."""

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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
            "happo",
        ],
        help="Algorithm name. Choose from: maddpg, mappo, igs.",
    )
    parser.add_argument(
        "--run",
        type=str,
        default="single",
        choices=[
            "single",
            "perturbation",
            "traitor",
        ],
        help="Runner pipeline name. Choose from: single, perturbation, traitor.",
    )
    parser.add_argument(
        "--victim",
        type=str,
        default="mappo",
        choices=[
            "maddpg",
            "mappo",
            "qmix"
        ],
        help="Victim algorithm name. Choose from: maddpg, mappo, qmix.",
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
    parser.add_argument("--exp_name", type=str, default="installtest", help="Experiment name.")
    parser.add_argument(
        "--load_config",
        type=str,
        default="",
        help="If set, load existing experiment config file instead of reading from yaml config file.",
    )
    parser.add_argument(
        "--attack_config",
        type=str,
        default="",
        help="If set, overwrite the train config by attack_config.",
    )
    parser.add_argument(
        "--load_victim",
        type=str,
        default="",
        help="If set, load existing victim config file and checkpoint file instead of reading from yaml config file.",
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

    if args["load_config"] != "":  # load config from existing config file
        with open(args["load_config"], encoding='utf-8') as file:
            all_config = json.load(file)
        args["algo"] = all_config["main_args"]["algo"]
        args["env"] = all_config["main_args"]["env"]
        args["run"] = all_config["main_args"]["run"]
        args["exp_name"] = all_config["main_args"]["exp_name"] if "exp_name" in all_config["main_args"] and all_config["main_args"]["exp_name"] != "" else args["exp_name"]

        algo_args = all_config["algo_args"]["train"]
        victim_args = all_config["algo_args"]["victim"]
        env_args = all_config["env_args"]
        if "algo.hidden_sizes" in unparsed_args:
            algo_args["hedden_sizes"] = unparsed_args["algo.hidden_sizes"]
    else:  # load config from corresponding yaml file
        if args["run"] == "single":
            algo_args = get_one_yaml_args(args["algo"])
        elif args["run"] == "perturbation" or args["run"] == "traitor":
            algo_args = get_one_yaml_args(args["algo"] + "_traitor")

        if args["load_victim"] != "":
            with open(os.path.join(args["load_victim"], "config.json"), encoding='utf-8') as file:
                victim_config = json.load(file)
            args["victim"] = victim_config["main_args"]["algo"]
            args["env"] = victim_config["main_args"]["env"]
            victim_config["algo_args"]["train"]["model_dir"] = os.path.join(args["load_victim"], "models")

            victim_args = victim_config["algo_args"]["train"]
            env_args = victim_config["env_args"]

            if args["attack_config"] != "":
                with open(args["attack_config"], encoding='utf-8') as file:
                    if args["attack_config"].endswith(".json"):
                        attack_args = json.load(file)
                        for k, v in attack_args.items():
                            if k in algo_args:
                                algo_args[k] = v
                    if args["attack_config"].endswith(".yaml"):
                        attack_args = yaml.load(file, Loader=yaml.FullLoader)
                        for k, v in attack_args.items():
                            if k in algo_args:
                                algo_args[k] = v
        else:
            victim_args = {}
            if args["run"] == "perturbation" or args["run"] == "traitor":
                victim_args = get_one_yaml_args(args["victim"])
            env_args = get_one_yaml_args(args["env"], type="env")
            
    # note: isaac gym does not support multiple instances, thus cannot eval separately
    if args["env"] == "dexhands":
        import isaacgym  # isaacgym has to be imported before PyTorch
        algo_args["use_eval"] = False
        # 要求episode_length是比hands_episode_length更大一点的数，但同时是data_chunk_length的倍数
        # 判断algo_args是否存在data_chunk_length，如果不存在，就不需要修改episode_length
        if "data_chunk_length" in algo_args:
            algo_args["episode_length"] = algo_args["data_chunk_length"] * (env_args["hands_episode_length"] // algo_args["data_chunk_length"] + 1)
        else:
            algo_args["episode_length"] = env_args["hands_episode_length"]

    update_args(unparsed_dict, algo=algo_args, env=env_args, victim=victim_args)  # update args from command line

    algo_args = {"train": algo_args, "victim": victim_args, "extra": unparsed_dict}

    if "perturb_timesteps" in algo_args["train"]:
        algo_args["train"]["perturb_timesteps"] = parse_timestep(algo_args["train"]["perturb_timesteps"], algo_args["train"]["episode_length"])

    # env-based randomization needs to be set before creating the environment
    # and before creating the runner
    if "env_randomization" in algo_args["extra"] and algo_args["extra"]["env_randomization"]:
        # update env_args with randomization
        # check env_noise_params in algo_args["extra"], split by ,
        env_noise_params = algo_args["extra"].get("env_noise_params", "").split(",")
        # get gussian noise mean and std
        env_noise_mean = algo_args["extra"].get("env_noise_mean", "")
        env_noise_std = algo_args["extra"].get("env_noise_std", "")
        print("==>env_noise_params", env_noise_params)
        print("==>env_noise_mean", env_noise_mean)
        print("==>env_noise_std", env_noise_std)

        def add_noise(cfgs, params, mean, std):
            noise = np.random.normal(mean, std)
            clipped_noise = np.clip(noise, -1, 1)
            for param in params:
                if param in cfgs:
                    print(f"==>Before: {param}={cfgs[param]}")
                    cfgs[param] = cfgs[param] + clipped_noise * cfgs[param]
                    print(f"==>After: {param}={cfgs[param]}")
                elif "=" in param:
                    key, value = param.split("=")
                    cfgs[key] = float(value) + clipped_noise
                    print(f"==>Set: {key}={cfgs[key]}")
            return cfgs

        # specific for every env
        if args["env"] == "quads":
            env_noise_params = env_noise_params if env_noise_params != [''] \
                else ['quads_collision_hitbox_radius', 'quads_collision_falloff_radius', 'quads_obst_density', 'quads_obst_size', 'quads_collision_reward', 'quads_collision_smooth_max_penalty', 'quads_episode_duration', 'quads_obst_collision_reward', 'quads_obst_density', 'replay_buffer_sample_prob']
            env_noise_mean = env_noise_mean if env_noise_mean else 0.0
            env_noise_std = env_noise_std if env_noise_std else 0.3
            add_noise(env_args["conf"], env_noise_params, env_noise_mean, env_noise_std)
        elif args["env"] == "voltage":
            env_noise_params = env_noise_params if env_noise_params != [''] \
                else ['pv_scale', 'demand_scale', 'v_upper', 'v_lower', 'q_weight', 'action_bias', 'action_scale', 'demand_scale']
            env_noise_mean = env_noise_mean if env_noise_mean else 0.0
            env_noise_std = env_noise_std if env_noise_std else 0.3
            add_noise(env_args["env_args"], env_noise_params, env_noise_mean, env_noise_std)
        elif args["env"] == "dexhands":
            env_noise_params = env_noise_params if env_noise_params != [''] \
                else ['startPositionNoise', 'transition_scale', 'orientation_scale', 'rotRewardScale', 'startRotationNoise','resetPositionNoise','resetRotationNoise','resetDofPosRandomInterval','resetDofVelRandomInterval']
            env_noise_mean = env_noise_mean if env_noise_mean else 0.0
            env_noise_std = env_noise_std if env_noise_std else 0.3
            env_args["override_args"] = {
                'startPositionNoise': 0.005,
                'transition_scale': 0.5,
                'orientation_scale': 0.1,
                'rotRewardScale': 1.0,
                
                'startRotationNoise': 0.0,
                'resetPositionNoise': 0.005,
                'resetRotationNoise': 0.0,
                'resetDofPosRandomInterval': 0.2,
                'resetDofVelRandomInterval': 0.0,
            }
            add_noise(env_args["override_args"], env_noise_params, env_noise_mean, env_noise_std)
        elif args["env"] == "network":
            env_noise_params = env_noise_params if env_noise_params != [''] \
                else ['coop_gamma','headway_min','headway_st','headway_go','speed_max','accel_max','accel_min']
            env_noise_mean = env_noise_mean if env_noise_mean else 0.0
            env_noise_std = env_noise_std if env_noise_std else 0.3
            env_args["override_args"] = {
                'coop_gamma' : 0.8,
                'headway_min' : 1,
                'headway_st' : 5,
                'headway_go' : 35,
                'speed_max' : 30,
                'accel_max' : 2.5,
                'accel_min' : -2.5,
            }
            add_noise(env_args["override_args"], env_noise_params, env_noise_mean, env_noise_std)

    if "traitor_eps" in algo_args["extra"] and algo_args["extra"]["traitor_eps"]:
        algo_args["train"]["traitor_eps"] = algo_args["extra"]["traitor_eps"]
    else:
        algo_args["train"]["traitor_eps"] = 0
        
    if "adv_all" in algo_args["extra"] and algo_args["extra"]["adv_all"]:
        algo_args["train"]["adv_all"] = algo_args["extra"]["adv_all"]
    else:
        algo_args["train"]["adv_all"] = False

    if "perturbation_eps" in algo_args["extra"] and algo_args["extra"]["perturbation_eps"]:
        algo_args["train"]["perturbation_eps"] = algo_args["extra"]["perturbation_eps"]
    else:
        algo_args["train"]["perturbation_eps"] = 0
        
    if "adv_eps" in algo_args["extra"] and algo_args["extra"]["adv_eps"]:
        algo_args["train"]["perturb_epsilon"] = algo_args["extra"]["adv_eps"]

    if "use_mad_perform" in algo_args["extra"] and algo_args["extra"]["use_mad_perform"]:
        algo_args["train"]["use_mad_perform"] = algo_args["extra"]["use_mad_perform"]

    if "algo.use_regulation" in algo_args["extra"] and algo_args["extra"]["algo.use_regulation"]:
        print("Use ERINE regulation")
        algo_args["train"]["use_regulation"] = True
        algo_args["train"]["ernie_lambda"] = 0.4
        algo_args["train"]["ernie_epsilon"] = 0.1


    # start training
    from amb.runners import get_single_runner
    runner = get_single_runner(args["run"], args["algo"])(args, algo_args, env_args)
    run_error = None
    try:
        if algo_args["train"]['use_render']:
            runner.render()
        else:
            runner.run()
    except BaseException as exc:
        run_error = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        try:
            runner.close()
        except BaseException as exc:
            run_error = run_error or {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            if hasattr(runner, "run_dir"):
                from amb.utils.run_manifest import finish_run_manifest

                finish_run_manifest(runner.run_dir, run_error)

    print(">>>END Experiment finished successfully. END<<<")


if __name__ == "__main__":
    main()
