from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from stable_baselines3.common.monitor import Monitor
import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, ProgressBarCallback
from stable_baselines3.common.evaluation import evaluate_policy
from environments.vertisim_env import VertiSimEnvWrapper
import gymnasium as gym
import torch
from utils.learning_rate_schedule import linear_schedule
from utils.helpers import mask_fn
from stable_baselines3.common.logger import configure
import time


def maskable_ppo(log_dir, tensorboard_log_dir):

    policy_kwargs = dict(activation_fn=torch.nn.ReLU,
                         net_arch=[128, 128])    
                             
    # # Create the environment
    env = gym.make('vertisim', rl_model="MaskablePPO")
    # Wrap the environment in a monitor wrapper    
    env = Monitor(env)    
    # Wrap the environment in an action mask wrapper
    env = ActionMasker(env, mask_fn)
    # Create the model
    model = MaskablePPO(policy=MaskableActorCriticPolicy, 
                        env=env, 
                        verbose=1, 
                        batch_size=64,
                        ent_coef=0.05,
                        n_steps=1024,
                        learning_rate=linear_schedule(0.001),                
                        policy_kwargs=policy_kwargs,
                        tensorboard_log=tensorboard_log_dir) 
    # Print the model
    print(f"Using environment: {env} with {MaskablePPO} policy")       

    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(save_freq=5000, save_path=log_dir,
                                             name_prefix="checkpoint")

    model.save("./model/masked_ppo_vertisim")

    model.learn(total_timesteps=1024*1000, callback=[checkpoint_callback], log_interval=1)