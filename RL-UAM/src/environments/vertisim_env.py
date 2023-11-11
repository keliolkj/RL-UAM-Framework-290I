from typing import Any, SupportsFloat, Dict, Tuple
import gymnasium as gym
import numpy as np
from utils.helpers import extract_dict_values
import requests
import time
from requests.exceptions import RequestException

SERVICE_ORCHESTRATOR_API_URL = "http://service_orchestrator:6000"
VERTISIM_API_URL = "http://vertisim_service:5001"

class VertiSimEnvWrapper(gym.Env):
    def __init__(self, rl_model: str) -> None:
        self.rl_model = rl_model
        self.params = self._fetch_params()
        self.action_space = self.get_action_space(self.params['n_actions'], 
                                                  self.params['n_aircraft'])
        self.observation_space = self.get_observation_space(self.params['n_vertiports'],
                                                            self.params['n_aircraft'],
                                                            self.params['n_vertiport_state_variables'],
                                                            self.params['n_aircraft_state_variables'],
                                                            self.params['n_environmental_state_variables'],
                                                            self.params['n_additional_state_variables'])
        self.mask = np.ones((self.params['n_aircraft'], self.params['n_actions']), dtype=np.float64)

    def _fetch_params(self):
        response = requests.get(f"{VERTISIM_API_URL}/get_space_params", timeout=30)
        if response.status_code == 200:
            return response.json()
        else:
            raise ConnectionError(f"Failed to fetch parameters from VertiSim. Status code: {response.status_code}, Response text: {response.text}")

    def get_action_space(self, n_actions, n_aircraft):
        if self.rl_model in ["DQN"]:
            return gym.spaces.Discrete(n_actions**n_aircraft)
        elif self.rl_model in ["PPO", "MaskablePPO"]:
            return gym.spaces.MultiDiscrete([n_actions] * n_aircraft)

    def get_observation_space(self, 
                              n_vertiports, 
                              n_aircraft, 
                              n_vertiport_state_variables, 
                              n_aircraft_state_variables, 
                              n_environmental_state_variables,
                              n_additional_state_variables):
        total_state_variables = (
            n_vertiports * n_vertiport_state_variables +
            n_aircraft * n_aircraft_state_variables +
            n_vertiports * (n_environmental_state_variables+1) +
            n_additional_state_variables
        )

        return gym.spaces.Box(low=0, high=np.inf, shape=(total_state_variables,), dtype=np.float64)
    
    def step(self, action):
        # Retrying logic parameters
        max_retries = 6
        backoff_factor = 2
        initial_delay = 1  # Initial delay in seconds

        # Convert actions
        action = self._convert_actions(action)
        # print(f"RL called step with action: {action}. Type: {type(action)}")

        for attempt in range(max_retries):
            try:
                # Send the action to VertiSim via HTTP and receive the new state
                response = requests.post(f"{VERTISIM_API_URL}/step", json={"actions": action}, timeout=120)
                if response.status_code == 200:
                    # Process the successful response
                    new_state, reward, terminated, truncated, self.mask = VertiSimEnvWrapper.process_step_response(response)
                    return new_state, reward, terminated, truncated, {}
                else:
                    raise ConnectionError(f"Failed to fetch step from VertiSim. Status code: {response.status_code}, Response text: {response.text}")

            except RequestException as e:
                if attempt >= max_retries - 1:
                    raise ConnectionError(
                        f"Failed to fetch step from VertiSim after {max_retries} tries. Error: {str(e)}"
                    ) from e
                delay = initial_delay * backoff_factor ** attempt
                print(f"Failed to fetch step from VertiSim. Error: {str(e)}. Retrying in {delay} seconds...")
                time.sleep(delay)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        print("RL called reset")
        # Send a reset request to VertiSim and receive the initial state
        response = requests.post(f"{SERVICE_ORCHESTRATOR_API_URL}/reset_instance", timeout=120)
        if response.status_code == 200:
            try:
                obs_tensor = np.array(extract_dict_values(response.json()['initial_state'])).reshape(-1)
                info = {}
                return obs_tensor, info
            except:
                raise ValueError("Failed to extract initial state from reset response.")
        else:
            raise ConnectionError(f"Failed to fetch reset from Service Orchestrator. Status code: {response.status_code}, Response text: {response.text}")
    
    def render(self, mode='human'):
        pass
    
    def close(self):
        # Handle any cleanup or shutdown tasks here
        pass

    def action_mask(self):
        return np.array(self.mask, dtype=np.float64).reshape(self.params['n_aircraft'], self.params['n_actions'])

    def _convert_actions(self, action):
        if self.rl_model in ["DQN"]:
            action = VertiSimEnvWrapper.discrete_to_multidiscrete(action, dimensions=self.params['n_aircraft'], n_actions=self.params['n_actions'])
            action = [int(item) if isinstance(item, np.int64) else item for item in action]
        elif self.rl_model in ["PPO", "MaskablePPO"]:
            action = action.tolist()
        return action

    @staticmethod
    def process_step_response(response):
        # Extract new state, reward, done, and info from response data
        data = response.json()
        new_state = np.array(extract_dict_values(data['new_state']))
        reward = data['reward']
        terminated = data['terminated']
        truncated = data['truncated']
        action_mask = data['action_mask']
        return new_state, reward, terminated, truncated, action_mask

    def seed(self, seed=None):
        # self.enseed(seed)
        # return super(VertiSimEnvWrapper, self).seed(seed)
        pass

    @staticmethod
    def multidiscrete_to_discrete(action, n_actions=3):
        """Converts a MultiDiscrete action to a Discrete action."""
        discrete_action = 0
        for i, a in enumerate(reversed(action)):
            discrete_action += a * (n_actions ** i)
        return discrete_action

    @staticmethod
    def discrete_to_multidiscrete(action, dimensions=4, n_actions=3):
        """Converts a Discrete action back to a MultiDiscrete action."""
        multidiscrete_action = []
        for _ in range(dimensions):
            multidiscrete_action.append(action % n_actions)
            action = action // n_actions
        return list(reversed(multidiscrete_action))  