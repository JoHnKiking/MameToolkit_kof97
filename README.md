# Reinforcement Learning for playing arcade fighting game KOF'97

## Introduction
In this project, we intend to develop an RL Agent which can not only win the game: KOF’97
The following works are done in this project
<ol>
<li>We made a gym envrionment of KOF 97. see <a href="https://github.com/duhuaiyu/AIbotForKof97/blob/master/Kof97EnvironmentSR.py" title="Kof97EnvironmentSR.py">Kof97EnvironmentSR.py</a></li>
<li>Train models for the specific opponent. see <a href="https://github.com/duhuaiyu/AIbotForKof97/blob/master/trainVecEnvV2.py" title="trainVecEnvV2.py">trainVecEnvV2.py</a> and <a href="https://github.com/duhuaiyu/AIbotForKof97/blob/master/trainVecEnvLSTM.py" title="trainVecEnvLSTM.py">trainVecEnvLSTM.py</a></li>
<li>Evaluate the performance of those models. see <a href="https://github.com/duhuaiyu/AIbotForKof97/blob/master/Kof97_Evaluation.ipynb" title="Kof97_Evaluation.ipynb">Kof97_Evaluation.ipynb</a></li>
<li>visualise the evaluation result. see <a href="https://github.com/duhuaiyu/AIbotForKof97/blob/master/visualize.ipynb" title="visualize.ipynb">visualize.ipynb</a></li>
</ol>


## 项目说明
能够训练任意kof97角色在任意难度下的强化学习模型

## 项目要求
系统：ubuntu

## 启动准备
1. 建立虚拟环境，安装相应的库文件
2. 在 `configure.py` 文件中设置您需要的配置（角色名、当前训练编号、训练环境数量、记录频率等）。
3. 将完整的 kof97 roms文件夹放进仓库根目录下的 `./roms` 中
4. 将存档文件放在根目录上一级目录下的 `..\KOF97Saves\CH_{角色名}\` 中
5. 在项目根目录下建立 `opt_2` 文件夹，放入预训练模型压缩文件，命名格式为 `trail_{角色名}_{难度等级}_last_model.zip`
6. 执行 `pip install tensorflow` 命令


## 如何建立游戏存档
1. 进入当前虚拟环境的 mametoolkit 安装目录，路径示意：**miniconda3/envs/env_isaaclab/lib/python3.10/site-packages/MAMEToolkit/**，并进入 **/emulator/mame**
2. 在终端输入 **exec ./mame -rompath '/home/workspace/AIbotForkof97/roms' -pluginspath plugins -skip_gameinfo -window -nomaximize -console  -frameskip 0 -sound none**
3. 选择并双击 kof97 ，版本任意
4. 在动画加载界面点击 F2 进入BIOS，能够修改对战难度等级(soft -> kof97)以及对战模式（选择个人对战）(轻拳选择)
5. 进入游戏后选择目标角色，进入战斗
6. 在战斗准备（ready go）的时候点击 **shift + F7** 出现存档界面
7. 点击任一字母或数字作为存档名称即可保存，重复已有存档名称会覆盖
8. 进入 `~/.mame/sta` 目录，游戏存档均位于此处

## 配置文件可自定义内容
1. 训练并行环境数量 - cpu_num
2. 当前训练编号 - trail  
3. 当前训练角色名 - CH
4. 当前对战难度 - level
5. 是否需要渲染模拟器画面 - render
6. 模型记录频率 - model_zip_freq
7. 胜率记录频率 - win_rate_freq
8. 总体训练频次 - total_freq

## 共享盘
在共享盘根目录下 `./shared/kof97/`
- 存档 - sta
- 游戏roms - roms
- 已训练模型 - model


## 快速启动
python trainVecEnvV2.py

## 项目结果
1. 在 `./logs` 中会出现当前训练编号的所有训练过程模型
2. 在 `./win_rate_data` 文件夹中会出现相应的胜率图和胜率日志
3. 在当前目录下会出现当前训练的最终训练模型 
4. 在 `./ts-log/V2_env` 文件夹中会出现当前训练的日志
5. 在终端运行 `tensorboard --logdir=./logs --port=6006`，访问浏览器 `http://localhost:6006`


