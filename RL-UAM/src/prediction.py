from sb3_contrib import MaskablePPO
import gymnasium as gym
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor
from utils.helpers import mask_fn


model = MaskablePPO.load("./model/masked_ppo_vertisim")

#  Create the environment
env = gym.make('vertisim', rl_model="MaskablePPO")
# Wrap the environment in a monitor wrapper    
env = Monitor(env)    
# Wrap the environment in an action mask wrapper
env = ActionMasker(env, mask_fn)

print("Making prediction")
obs, _ = env.reset()
terminated = False
while not terminated:
    # Retrieve current action mask
    action_masks = env.action_mask()
    action, _states = model.predict(obs, action_masks=action_masks)
    new_state, reward, terminated, truncated, _ = env.step(action)