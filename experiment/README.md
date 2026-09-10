## Trick Experiment Pipline

### 生成训练命令
    
```bash
python generate.py train --env <env_name> --scenario <scenario or map_name> --algo <algo_name>
```

### 执行训练命令，训练default和trick模型

方法一：使用`xargs`命令来并行执行命令，注意修改`-P`参数来控制并行数

```bash
cat <train_env_scenario_algo.sh> | xargs -I {} -P 4 bash -c "{} || echo '{}' >> errors.txt"
```

方法二：使用`run.py`来并行执行命令，注意修改`-n`参数来控制并行数

```bash
python run.py -s <train_env_scenario_algo.sh> -n 4
```

### 生成测试命令

由于本项目中的5种攻击算法，有三种（random_noise, iterative_perturbation, random_policy）是无需训练，另外两种（adaptive_action和traitor）是需要训练过程的，由于训练时间较长，如果针对要测试的trick模型再跑一遍攻击算法的训练，时间会更长。故实现了两种评测方式：

第一种：一次性生成所有测试命令，然后再执行所有测试命令。

```bash
python generate.py eval --env <env_name> --scenario <scenario or map_name> --algo <algo_name>
```

第二种：分成两个阶段执行，阶段一生成所有无需训练的评测命令，以及adaptive_action和traitor的default模型的评测命令，

```bash
python generate.py eval --env <env_name> --scenario <scenario or map_name> --algo <algo_name> --stage 1
```

在执行完阶段一的命令后，利用训练好的两个对抗模型来对所有其他trick模型进行评测。

```bash
python generate.py eval --env <env_name> --scenario <scenario or map_name> --algo <algo_name> --stage 2
```

### 汇总实验结果

将实验日志中的结果提取出来，汇总到一个csv文件中，默认输出目录`./data`。

```bash
python export.py --env <env_name> --scenario <scenario or map_name> --algo <algo_name> --method <all, random_noise, iterative_perturbation, adaptive_action, random_policy, traitor>
```

### 画图

将汇总的csv文件画图，输出到`plots`目录。



### 其他工具
将之前旧的目录格式化为新目录格式

```bash
python format.py <dir_path>
```