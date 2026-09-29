"""
Optional real Atari Pong experiment.

Install:
    pip install "gymnasium[atari,accept-rom-license]" ale-py stable-baselines3[extra]

Note: Atari Pong from pixels is much heavier than Toy Pong.
"""

import argparse
from pathlib import Path

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--envs", type=int, default=4)
    args = parser.parse_args()

    Path("models").mkdir(exist_ok=True)
    Path("tensorboard").mkdir(exist_ok=True)

    env = make_atari_env("PongNoFrameskip-v4", n_envs=args.envs, seed=0)
    env = VecFrameStack(env, n_stack=4)

    model = A2C(
        "CnnPolicy",
        env,
        verbose=1,
        tensorboard_log="tensorboard/",
    )
    model.learn(total_timesteps=args.steps)
    model.save("models/atari_pong_a2c")
    env.close()


if __name__ == "__main__":
    main()
