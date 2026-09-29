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
        self.opposites = {
            0: 1,
            1: 0,
            2: 3,
            3: 2,
            4: None
        }
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
            low=np.array([
                -2000,
                -2000,
                0,
                0,
                0,
                0
            ], dtype=np.float32),

            high=np.array([
                2000,
                2000,
                300,
                300,
                300,
                300
            ], dtype=np.float32),

            dtype=np.float32
        )

    def _make_observation(self, response):
        return np.array(
            [
                response["targetX"] - response["agentX"],
                response["targetY"] - response["agentY"],

                response["lidarFront"],
                response["lidarBack"],
                response["lidarRight"],
                response["lidarLeft"],
            ],
            dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        #Сбрасывает среду UE

        super().reset(seed=seed)

        response = self.ue.step(4)
        self.last_step = 4

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

        lidars = [response['lidarFront'], response['lidarBack'], response['lidarRight'], response['lidarLeft'],]

        lidars_valid = min(lidars)


        if response["terminated"]:
            reward = 100
        elif lidars_valid >= 100:
            reward = response["distanceDelta"] / 100.0 - 0.05
        elif 50 <= lidars_valid < 100:
            reward = -0.5
        else:
            reward = -1

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