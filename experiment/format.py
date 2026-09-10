import argparse
import os


def update_dir_name(dir_path):
    for dir in os.listdir(dir_path):
        new_dir = (
            dir.replace("activate_", "activation_func_")
            .replace("criticlr_", "critic_lr_")
            .replace("entropycoef_", "entropy_coef_")
            .replace("rnn_", "use_recurrent_policy_")
            .replace("norm_", "use_feature_normalization_")
            .replace("share_", "share_param_")
            .replace("hiddensize_", "hidden_sizes_")
            .replace("init_uniform", "initialization_method_xavier_uniform_")
            .replace("nstep_", "n_step_")
            .replace("gae_", "use_gae_")
            .replace("popart_", "use_popart_")
            .replace("false", "False")
            .replace("eps_anneal_time_", "epsilon_anneal_time_")
            .replace("eps_finish_", "epsilon_finish_")
            .replace("batch_", "batch_size_")
            .replace("true", "True")
            .replace("0.00005", "5e-05")
        )
        os.rename(os.path.join(dir_path, dir), os.path.join(dir_path, new_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dir_path", type=str, help="dir path")
    args, _ = parser.parse_known_args()
    update_dir_name(args.dir_path)
