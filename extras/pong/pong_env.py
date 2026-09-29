from __future__ import annotations

import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import pygame
except ImportError:
    pygame = None


class ToyPongEnv(gym.Env):
    """
    Deliberately tiny Pong environment for learning RL.

    True internal state:
      - player_y
      - opponent_y
      - ball_x, ball_y
      - ball_vx, ball_vy
      - player_score, opponent_score

    Observation can intentionally hide some variables to demonstrate
    state-representation / partial-observability experiments.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None, representation="full", max_steps=2500):
        super().__init__()
        self.render_mode = render_mode
        self.representation = representation
        self.max_steps = max_steps

        self.width = 800
        self.height = 500
        self.paddle_h = 90
        self.paddle_w = 14
        self.ball_r = 8
        self.player_x = self.width - 45
        self.opponent_x = 45

        # Actions: 0 up, 1 stay, 2 down
        self.action_space = spaces.Discrete(3)

        obs_dims = {
            "full": 6,
            "no_velocity": 4,
            "minimal": 3,
        }
        if representation not in obs_dims:
            raise ValueError(f"Unknown representation: {representation}")

        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(obs_dims[representation],),
            dtype=np.float32,
        )

        self.window = None
        self.clock = None

    def _obs(self):
        # Normalize to roughly [-1, 1].
        player_y = (self.player_y / self.height) * 2 - 1
        opponent_y = (self.opponent_y / self.height) * 2 - 1
        ball_x = (self.ball_x / self.width) * 2 - 1
        ball_y = (self.ball_y / self.height) * 2 - 1
        ball_vx = self.ball_vx / 8.0
        ball_vy = self.ball_vy / 8.0

        if self.representation == "full":
            vals = [player_y, ball_x, ball_y, ball_vx, ball_vy, opponent_y]
        elif self.representation == "no_velocity":
            vals = [player_y, ball_x, ball_y, opponent_y]
        else:
            vals = [player_y, ball_x, ball_y]

        return np.asarray(vals, dtype=np.float32)

    def _reset_ball(self, direction=None):
        self.ball_x = self.width / 2
        self.ball_y = self.height / 2 + self.np_random.uniform(-80, 80)
        if direction is None:
            direction = self.np_random.choice([-1, 1])
        angle = self.np_random.uniform(-0.45, 0.45)
        speed = 5.5
        self.ball_vx = direction * speed * math.cos(angle)
        self.ball_vy = speed * math.sin(angle)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.player_y = self.height / 2
        self.opponent_y = self.height / 2
        self.player_score = 0
        self.opponent_score = 0
        self.steps = 0
        self._reset_ball()

        obs = self._obs()
        info = self._info()

        if self.render_mode == "human":
            self.render()

        return obs, info

    def _info(self):
        return {
            "player_score": self.player_score,
            "opponent_score": self.opponent_score,
            "internal_state": {
                "player_y": float(self.player_y),
                "opponent_y": float(self.opponent_y),
                "ball_x": float(self.ball_x),
                "ball_y": float(self.ball_y),
                "ball_vx": float(self.ball_vx),
                "ball_vy": float(self.ball_vy),
            },
        }

    def step(self, action):
        self.steps += 1

        # Player movement
        paddle_speed = 7.0
        if action == 0:
            self.player_y -= paddle_speed
        elif action == 2:
            self.player_y += paddle_speed
        self.player_y = float(np.clip(self.player_y, self.paddle_h / 2, self.height - self.paddle_h / 2))

        # Simple opponent controller
        target = self.ball_y
        delta = np.clip(target - self.opponent_y, -5.0, 5.0)
        self.opponent_y = float(np.clip(
            self.opponent_y + delta,
            self.paddle_h / 2,
            self.height - self.paddle_h / 2
        ))

        # Ball
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # Top/bottom bounce
        if self.ball_y <= self.ball_r:
            self.ball_y = self.ball_r
            self.ball_vy *= -1
        elif self.ball_y >= self.height - self.ball_r:
            self.ball_y = self.height - self.ball_r
            self.ball_vy *= -1

        reward = 0.0

        # Opponent paddle collision
        if (
            self.ball_vx < 0
            and self.ball_x - self.ball_r <= self.opponent_x + self.paddle_w / 2
            and abs(self.ball_y - self.opponent_y) <= self.paddle_h / 2
        ):
            self.ball_x = self.opponent_x + self.paddle_w / 2 + self.ball_r
            self.ball_vx = abs(self.ball_vx) * 1.01
            offset = (self.ball_y - self.opponent_y) / (self.paddle_h / 2)
            self.ball_vy += 1.2 * offset

        # Player paddle collision: tiny positive shaping reward
        if (
            self.ball_vx > 0
            and self.ball_x + self.ball_r >= self.player_x - self.paddle_w / 2
            and abs(self.ball_y - self.player_y) <= self.paddle_h / 2
        ):
            self.ball_x = self.player_x - self.paddle_w / 2 - self.ball_r
            self.ball_vx = -abs(self.ball_vx) * 1.01
            offset = (self.ball_y - self.player_y) / (self.paddle_h / 2)
            self.ball_vy += 1.2 * offset
            reward += 0.05

        # Score
        if self.ball_x < 0:
            self.player_score += 1
            reward += 1.0
            self._reset_ball(direction=1)

        elif self.ball_x > self.width:
            self.opponent_score += 1
            reward -= 1.0
            self._reset_ball(direction=-1)

        # Episode ends when someone reaches 7 points.
        terminated = self.player_score >= 7 or self.opponent_score >= 7
        truncated = self.steps >= self.max_steps

        obs = self._obs()
        info = self._info()

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if pygame is None:
            raise RuntimeError("pygame is required for rendering")

        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Toy Pong RL Lab")
        if self.clock is None:
            self.clock = pygame.time.Clock()

        surf = self.window if self.render_mode == "human" else pygame.Surface((self.width, self.height))
        surf.fill((18, 18, 22))

        # Midline
        for y in range(0, self.height, 30):
            pygame.draw.rect(surf, (80, 80, 90), (self.width // 2 - 2, y, 4, 15))

        # Paddles + ball
        pygame.draw.rect(
            surf, (230, 230, 235),
            (self.opponent_x - self.paddle_w / 2, self.opponent_y - self.paddle_h / 2, self.paddle_w, self.paddle_h)
        )
        pygame.draw.rect(
            surf, (230, 230, 235),
            (self.player_x - self.paddle_w / 2, self.player_y - self.paddle_h / 2, self.paddle_w, self.paddle_h)
        )
        pygame.draw.circle(surf, (240, 240, 245), (int(self.ball_x), int(self.ball_y)), self.ball_r)

        font = pygame.font.SysFont("Arial", 36)
        score = font.render(f"{self.opponent_score}     {self.player_score}", True, (210, 210, 220))
        surf.blit(score, (self.width / 2 - score.get_width() / 2, 20))

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None

        arr = pygame.surfarray.array3d(surf)
        return np.transpose(arr, (1, 0, 2))

    def close(self):
        if self.window is not None and pygame is not None:
            pygame.quit()
            self.window = None
            self.clock = None
