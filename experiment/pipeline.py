

import argparse
import os
import subprocess

current_dir = os.path.dirname(os.path.abspath(__file__))


def execute_command(command, capture_output=False):
    print(f"\033[32m==> {command}\033[0m")
    process = subprocess.run(command, shell=True, capture_output=capture_output, text=True)
    # print(f"\033[31m{process}\033[0m")
    return process


def train(args):
    process = execute_command(f"python {current_dir}/generate.py train {args.extra}", capture_output=True)
    execute_command(process.stdout)


def eval_all(args):
    process = execute_command(f"python {current_dir}/generate.py eval {args.extra}", capture_output=True)
    execute_command(process.stdout)


def eval_stage_1(args):
    process = execute_command(f"python {current_dir}/generate.py eval {args.extra} --stage 1", capture_output=True)
    execute_command(process.stdout)


def eval_stage_2(args):
    process = execute_command(f"python {current_dir}/generate.py eval {args.extra} --stage 2", capture_output=True)
    execute_command(process.stdout)


def export(args):
    execute_command(f"python {current_dir}/export.py {args.extra}")


def plot(args):
    execute_command(f"python {current_dir}/plot.py {args.extra}")


def rsync(output_dir):
    execute_command(f"python {current_dir}/rsync.py {args.extra}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", type=str, default="train", choices=["train", "eval", "export", "plot", "rsync"], help="start phase: train, eval, export, plot, rsync")
    parser.add_argument("-o", "--out", type=str, default="out", help="output dir")
    parser.add_argument("--fast", action="store_true", help="use fast mode for eval (stage 1 -> stage 2)")
    parser.add_argument("--stage", type=int, default=0, choices=[0, 1, 2], help="stage_0: eval all; stage_one: only eval default model in adaptive_action and traitor; stage_two:load adv model to eval.")
    parser.add_argument('--rsync', action='store_true', help='Whether to rsync the output dir to remote server')
    parser.add_argument('--end', action='store_true', help='Run continuously from the current execution until end.')
    args, unknown_args = parser.parse_known_args()
    # 将未知参数转换为字符串
    extra = " ".join(unknown_args)
    extra += f" --out {args.out}"
    args.extra = extra

    if args.phase == "train":
        # train
        train(args)

        if args.end:
            # eval
            if args.fast:
                # eval stage 1
                eval_stage_1(args)
                # eval stage 2
                eval_stage_2(args)

            else:
                eval_all(args)

            # export
            export(args)
            
            # plot
            plot(args)
    
    elif args.phase == "eval":
        if args.stage == 1 or args.fast:
            eval_stage_1(args)
            eval_stage_2(args)

        elif args.stage == 2:
            eval_stage_2(args)

        else:
            eval_all(args)
        
        if args.end:
            # export
            export(args)

            # plot
            plot(args)
    
    elif args.phase == "export":
        # export
        export(args)

        if args.end:
            # plot
            plot(args)

    elif args.phase == "plot":
        # plot
        plot(args)

    elif args.phase == "rsync":
        # rsync
        rsync(args)

    # sync to remote when --rsync and not in rsync phase
    if args.rsync and args.phase != "rsync":
        rsync(args)