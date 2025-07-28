import gym
import os
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv,VecMonitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.preprocessing import is_image_space
from multiprocessing import Process, freeze_support, set_start_method
from Kof97EnvironmentSR import Kof98EnvironmentV2,ComboRewardCalculatorV3
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.torch_layers import CombinedExtractor
from Callback import VideoRecorderCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.save_util import AsyncCompressor
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# import logging
from loguru import logger
import sys
from configure import get_config
config = get_config()

# logger = logging.getLogger("win_rate")
# logger.setLevel(logging.DEBUG)

# file_handler = logging.FileHandler("./win_rate_data/win_rate.log")
# file_handler.setLevel(logging.DEBUG)
# formatter = logging.Formatter("%(asctime)s -%(name)s - %(levelname)s - %(message)s")
# file_handler.setFormatter(formatter)

# console_handler = logging.StreamHandler()
# console_handler.setLevel(logging.INFO)

# logger.addHandler(file_handler)
# logger.addHandler(console_handler)

# 添加文件日志（使用默认格式）
logger.add(
    "./win_rate_data/win_rate.log",  # 日志文件路径
    level="DEBUG",  # 记录 DEBUG 及以上级别的日志
    rotation="10 MB",  # 可选：日志文件轮转（如 "100 MB"、"1 GB"）
    retention="7 days",  # 可选：日志保留时间
    encoding="utf-8",  # 确保 UTF-8 编码
)

# 添加控制台日志（使用默认格式 + 颜色）
logger.add(
    sys.stdout,
    level="INFO",  # 控制台只输出 INFO 及以上级别
    colorize=True,  # 启用颜色（默认已启用，可省略）
)



class WinRateCallback(BaseCallback):
    """
    胜率统计回调
    """
    def __init__(self, save_freq: int, save_path: str = "win_rate_data", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.win_rates = []  # 存储当前窗口胜率值
        self.total_win_rates = []  # 存储整体胜率值
        self.timesteps = []  # 存储对应的时间步
        self.win_counts = []  # 存储胜场数
        self.loss_counts = []  # 存储败场数
        self.episode_counts = []  # 存储总回合数
        self.last_win = 0
        self.last_loss = 0
        self.last_total = 0
        
        # 创建保存目录
        os.makedirs(save_path, exist_ok=True)

        plt.ion()  # 开启交互模式
        self.fig = plt.figure(figsize=(12, 6))
        self.ax1 = self.fig.add_subplot(1, 2, 1)
        self.ax2 = self.fig.add_subplot(1, 2, 2)
    
    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            vec_env = self.training_env
            env_nums = vec_env.num_envs
            total_win = 0
            total_loss = 0
            total_count = 0
            for i in range(env_nums):
                status = vec_env.env_method("get_status", indices=i)[0]
                # 累加统计数据
                total_win += status["win_count"]
                total_loss += status["loss_count"]
                total_count += status["episode_count"]
            
            if total_count > 0:
                # 计算当前窗口胜率
                current_win = total_win - self.last_win
                current_loss = total_loss - self.last_loss
                current_total = total_count - self.last_total
                win_rate = current_win / current_total * 100 if current_total > 0 else 0
                
                # 计算整体胜率
                total_win_rate = total_win / total_count * 100
                
                logger.debug(f"current freq: {self.n_calls}")
                logger.debug(f"win_rate: {win_rate:.2f}% (win_count: {current_win}, loss_count: {current_loss}, current_total_count: {current_total})")
                logger.debug(f"total_win_rate: {total_win_rate:.2f}% (total_win_count: {total_win}, total_loss_count: {total_loss}, total_count: {total_count})")

                self.last_win = total_win
                self.last_loss = total_loss
                self.last_total = total_count
                
                # 记录数据
                self.win_rates.append(win_rate)
                self.total_win_rates.append(total_win_rate)
                self.timesteps.append(self.num_timesteps)
                self.win_counts.append(total_win)
                self.loss_counts.append(total_loss)
                self.episode_counts.append(total_count)
            
            # 定期保存数据
            self._update_plot()
            if len(self.win_rates) % 10 == 0:
                self._save_data()
        return True
    
    def _update_plot(self):
        """更新实时绘图"""
        # 清除原有图形
        self.ax1.clear()
        self.ax2.clear()
        
        if len(self.win_rates) > 0:
            # 绘制双胜率曲线
            self.ax1.plot(self.timesteps, self.win_rates, 'b-', linewidth=2, label='current_win_rate')
            self.ax1.plot(self.timesteps, self.total_win_rates, 'r-', linewidth=2, label='total_win_rate')
            self.ax1.set_xlabel('step')
            self.ax1.set_ylabel('win_rate (%)')
            self.ax1.set_title('Changes in win rate during training')
            self.ax1.legend()
            self.ax1.grid(True)
            
            # 自动调整x轴范围
            if len(self.timesteps) > 1:
                self.ax1.set_xlim(left=self.timesteps[0], right=self.timesteps[-1]*1.05)
            
            # 绘制累计胜场和败场
            self.ax2.plot(self.timesteps, self.win_counts, 'g-', label='win_count')
            self.ax2.plot(self.timesteps, self.loss_counts, 'r-', label='loss_count')
            self.ax2.set_xlabel('step')
            self.ax2.set_ylabel('count')
            self.ax2.set_title('Cumulative battle results')
            self.ax2.legend()
            self.ax2.grid(True)
            
            # 自动调整x轴范围
            if len(self.timesteps) > 1:
                self.ax2.set_xlim(left=self.timesteps[0], right=self.timesteps[-1]*1.05)
        
        # 刷新图形
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
    
    def _on_training_end(self) -> None:
        """训练结束时调用的方法"""
        self._save_data()
        logger.debug(f"胜率数据已保存至 {self.save_path}")
        plt.ioff()  # 关闭交互模式
        plt.close()  # 关闭图形窗口
    
    def _save_data(self):
        """保存数据到CSV文件"""
        data = {
            "timestep": self.timesteps,
            "window_win_rate": self.win_rates,
            "total_win_rate": self.total_win_rates,
            "win_count": self.win_counts,
            "loss_count": self.loss_counts,
            "episode_count": self.episode_counts
        }
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.save_path, "win_rate_history.csv"), index=False)
    
    def _plot_win_rate(self):
        """绘制胜率曲线图"""
        if len(self.win_rates) < 2:
            logger.debug("数据点不足，无法绘制曲线图")
            return
        
        plt.figure(figsize=(12, 6))
        
        # 绘制双胜率曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.timesteps, self.win_rates, 'b-', linewidth=2, label='current_win_rate')
        plt.plot(self.timesteps, self.total_win_rates, 'r-', linewidth=2, label='total_win_rate')
        plt.xlabel('step')
        plt.ylabel('win_rate (%)')
        plt.title('Changes in win rate during training')
        plt.legend()
        plt.grid(True)
        
        # 绘制累计胜场和败场
        plt.subplot(1, 2, 2)
        plt.plot(self.timesteps, self.win_counts, 'g-', label='win_count')
        plt.plot(self.timesteps, self.loss_counts, 'r-', label='loss_count')
        plt.xlabel('step')
        plt.ylabel('count')
        plt.title('Cumulative battle results')
        plt.legend()
        plt.grid(True)
        
        # 保存图像
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, "win_rate_plot.png"))
        plt.close()
        logger.debug(f"胜率图像已保存至 {self.save_path}")

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = create_env()
        # env = gym.make(env_id)
        env.seed(seed + rank)
        # env = Monitor(env, filename=sub_dir)
        return env
    set_random_seed(seed)
    return _init

def create_env():
    env_res = Kof98EnvironmentV2(CH="CH_"+config.v2.CH, env_param={"render": config.v2.render})
    return env_res

env_str = "V2_env"

if __name__ == '__main__':

    # compressor = AsyncCompressor(10)
    # env = gym.make("kof97:kof97-v0")
    # env.reset()
    set_start_method('forkserver', force=True)
    # env_id = "kof97:kof97-v1"
    env_id = config.v2.env_id
    num_cpu = config.v2.cpu_num # Number of use
    trail = config.v2.trail

    # Create the vectorized environment
    env = VecMonitor(SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)]))
    log_dir = config.v2.paths.log_dir
    os.makedirs(log_dir, exist_ok=True)
    # evl_env = Monitor(create_env())
    # params = config.v2.params
    # model = PPO('MultiInputPolicy', env, verbose=0,tensorboard_log=log_dir,**params)
    model = PPO.load(config.v2.paths.model_input_path + config.v2.names.model_input_name, env=env, tensorboard_log="./tensorboard_logs")
    model.learning_rate = config.v2.target_learning_rate
    checkpoint_callback = CheckpointCallback(save_freq=config.v2.freq.model_zip_freq, save_path=config.v2.paths.model_output_path, name_prefix=config.v2.names.model_output_name)
    
    win_rate_callback = WinRateCallback(save_freq=config.v2.freq.win_rate_freq, save_path=config.v2.paths.win_rate_data)
    model.learn(total_timesteps=config.v2.freq.total_freq, tb_log_name=config.v2.names.tb_log_name, reset_num_timesteps=True, callback=[checkpoint_callback, win_rate_callback])
    # model.learn(total_timesteps=3000,tb_log_name="PPO_V2_t3_transfer_CH_bsa", reset_num_timesteps=True, callback=win_rate_callback)
    # model.learn(total_timesteps=30000,tb_log_name="PPO_V2_t3_transfer_CH_bsa", reset_num_timesteps=True, callback=checkpoint_callback)
    print("finish learn")
    env.close()
    # evl_env.close()
    print("finish close")

    model.save("Kof97_PPO_V2_t3_transfer_CH_bsa")
    # obs = env.reset()
    # for _ in range(10000):
    #     action, _states = model.predict(obs)
    #     obs, rewards, dones, info = env.step(action)
    #     env.render()

