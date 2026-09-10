import argparse
import os
import subprocess


def rsync(args):
    # 读取.env中的配置项
    from dotenv import load_dotenv

    # 加载.env文件
    load_dotenv()

    # 读取环境变量
    remote_server = os.getenv('REMOTE_SERVER')
    port = os.getenv('REMOTE_PORT')
    user = os.getenv('REMOTE_USER')
    remote_path = os.getenv('REMOTE_PATH')

    # -a 递归传输文件，带信息
    # -v 显示传输过程
    # --delete 删除接收端没有的文件
    # --exclude 排除文件
    # -e 指定ssh端口
    # -n 模拟传输过程
    command = f"rsync -av {'--delete' if args.delete else ''} -e 'ssh -p {port}' {args.out} {user}@{remote_server}:{remote_path}/{args.remote}"
    print(f"\033[32m==> {command}\033[0m")
    subprocess.run(command, shell=True, text=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, default="./out", help="rsync local dir")
    parser.add_argument("--remote", type=str, default="", help="remote dir")
    parser.add_argument("--delete", action="store_true", help="rsync --delete mode")
    args, _ = parser.parse_known_args()

    rsync(args)