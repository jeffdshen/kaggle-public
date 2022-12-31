import numpy as np
import gym
from gym import spaces

import pygame

from .utils import ARMS, ArmHelper, get_cost
from .levels import LEVELS


class SantaGameEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 4}

    def __init__(
        self,
        levels=LEVELS,
        size=(64, 64),
        arms=ARMS,
        unseen_weight=2.0,
        target_weight=0.02,
        render_mode=None,
    ):
        self.size = np.array(size)
        self.levels = levels
        self.arm_helper = ArmHelper(arms)
        self.unseen_weight = unseen_weight
        self.target_weight = target_weight
        num_angles = self.arm_helper.num_angles

        self.pix_square_size = 10
        self.initial_window_size = (self.size + 1) * self.pix_square_size
        self.final_window_size = np.array([1028, 1028])

        self.observation_space = spaces.Dict(
            {
                "colors": spaces.Box(0.0, 1.0, shape=(*self.size, 3), dtype=np.float32),
                "seen": spaces.Box(0, 1, shape=(*self.size,), dtype=np.uint8),
                "arm": spaces.Box(
                    np.zeros_like(num_angles), num_angles, dtype=np.int64
                ),
                "loc": spaces.Box(np.zeros((2,)), self.size - 1, dtype=np.int64),
                "target": spaces.Box(np.zeros((2,)), self.size - 1, dtype=np.int64),
            }
        )
        self.action_space = spaces.MultiDiscrete([3] * 8)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        return {
            "colors": self._colors,
            "seen": self._seen,
            "arm": self._arm,
            "loc": self._loc,
            "target": self._target,
        }

    def _get_info(self):
        return {
            "level": self._level_index,
            "remaining": self._remaining,
            "cost": self._cost,
            "total_cost": self._total_cost,
        }

    def _set_to_target(self):
        self._seen.fill(1)
        self._loc = np.copy(self._target)
        self._remaining = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._level_index = self.np_random.integers(0, len(self.levels))
        self._level = self.levels[self._level_index]
        self._colors = np.array(self._level.colors)
        self._seen = np.copy(self._level.seen)
        self._arm = np.copy(self._level.arm)
        self._loc = np.copy(self._level.start)
        self._target = np.copy(self._level.target)
        self._remaining = self._level.remaining
        self._cost = 0.0
        self._total_cost = 0.0
        self._loc_history = [self._loc]

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        if np.all(np.array(action) == 0):
            self._set_to_target()
            return self._get_obs(), -1000, True, False, self._get_info()

        prev_abs_loc = self.arm_helper.to_loc(self._arm)
        self._arm = np.array(self.arm_helper.rotate(self._arm, action))
        new_abs_loc = self.arm_helper.to_loc(self._arm)
        new_loc = self._loc + np.subtract(new_abs_loc, prev_abs_loc)

        if not np.all((new_loc >= 0) & (new_loc < self.size)):
            self._set_to_target()
            return self._get_obs(), -1000, True, False, self._get_info()

        old_color = self._colors[tuple(self._loc)]
        old_dist = np.abs(self._loc - self._target).sum()
        self._loc = new_loc
        new_color = self._colors[tuple(self._loc)]
        new_dist = np.abs(self._loc - self._target).sum()
        delta_color = np.subtract(new_color, old_color)
        self._loc_history.append(self._loc)

        unseen = 1 - self._seen[tuple(self._loc)]
        self._remaining -= unseen

        self._seen[tuple(self._loc)] = 1
        self._cost = get_cost(delta_color, action)
        reward = unseen - self._cost - (new_dist - old_dist)
        self._total_cost += self._cost
        terminated = np.array_equal(self._loc, self._target) and self._remaining == 0
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, False, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        canvas = pygame.Surface(self.initial_window_size)

        canvas.fill((255, 255, 255))
        pix_square_size = self.pix_square_size

        for x, y in np.ndindex(*self.size):
            pygame.draw.rect(
                canvas,
                tuple(self._colors[x, y] * 255),
                pygame.Rect(np.array((x, y, 1, 1)) * pix_square_size),
            )
            if not self._seen[x, y]:
                pygame.draw.circle(
                    canvas,
                    (0, 0, 0),
                    np.array((x + 0.5, y + 0.5)) * pix_square_size,
                    0.25 * pix_square_size,
                )

        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._loc + 0.5) * pix_square_size,
            0.25 * pix_square_size,
        )
        pygame.draw.circle(
            canvas,
            (255, 0, 0),
            (self._target + 0.5) * pix_square_size,
            0.25 * pix_square_size,
        )

        draw_canvas = pygame.transform.flip(
            pygame.transform.scale(canvas, self.final_window_size), False, True
        )
        return np.transpose(
            np.array(pygame.surfarray.pixels3d(draw_canvas)), axes=(1, 0, 2)
        )
