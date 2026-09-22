import gymnasium as gym
import numpy as np
from gymnasium import spaces

from UE_client import UnrealClient


class UnrealEnv(gym.Env):

    metadata = {
        "render_modes": []
    }

    def __init__(self):
        super().__init__()

        self.ue = UnrealClient()

        # Доступные действия агента:
        #
        # 0 = +Y
        # 1 = -Y
        # 2 = +X
        # 3 = -X
        #
        # 4 Исключительно для reset().
        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low=-2000.0,
            high=2000.0,
            shape=(4,),
            dtype=np.float32
        )

    def _make_observation(self, response):
        return np.array(
            [
                response["agentX"],
                response["agentY"],
                response["targetX"],
                response["targetY"],
            ],
            dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        #Сбрасывает среду UE

        super().reset(seed=seed)

        response = self.ue.step(4)

        observation = self._make_observation(
            response
        )

        return observation, {}

    def step(self, action):
        action = int(action)

        response = self.ue.step(action)

        observation = self._make_observation(
            response
        )

        reward = float(
            response["reward"]
        )

        terminated = bool(
            response["terminated"]
        )

        truncated = bool(
            response["truncated"]
        )

        distance = float(
            np.sqrt(
                (response["targetX"] - response["agentX"]) ** 2
                + (response["targetY"] - response["agentY"]) ** 2
            )
        )

        info = {
            "distance": distance,
            "step": int(
                response["currentStep"]
            )
        }

        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )

    def close(self):
        self.ue.close()
        super().close()