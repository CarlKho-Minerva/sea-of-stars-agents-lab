import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from pong_env import ToyPongEnv


class RewardHistoryCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.timesteps = []
        self.rewards = []

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                self.timesteps.append(self.num_timesteps)
                self.rewards.append(ep["r"])
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=120_000)
    parser.add_argument("--representation", choices=["full", "no_velocity", "minimal"], default="full")
    args = parser.parse_args()

    Path("models").mkdir(exist_ok=True)
    Path("logs").mkdir(exist_ok=True)
    Path("plots").mkdir(exist_ok=True)
    Path("tensorboard").mkdir(exist_ok=True)

    env = Monitor(ToyPongEnv(representation=args.representation), filename="logs/monitor.csv")

    callback = RewardHistoryCallback()
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gamma=0.99,
        tensorboard_log="tensorboard/",
        policy_kwargs=dict(net_arch=[64, 64]),
    )

    model.learn(total_timesteps=args.steps, callback=callback)
    model.save("models/pong_ppo")

    if callback.rewards:
        df = pd.DataFrame({
            "timestep": callback.timesteps,
            "episode_reward": callback.rewards,
        })
        df["rolling_reward"] = df["episode_reward"].rolling(20, min_periods=1).mean()
        df.to_csv("logs/training_rewards.csv", index=False)

        plt.figure(figsize=(9, 5))
        plt.plot(df["timestep"], df["rolling_reward"])
        plt.xlabel("Environment steps")
        plt.ylabel("20-episode rolling reward")
        plt.title(f"Toy Pong PPO learning curve ({args.representation})")
        plt.tight_layout()
        plt.savefig("plots/training_curve.png", dpi=160)
        print("Saved plots/training_curve.png")

    env.close()
    print("Saved model to models/pong_ppo.zip")


if __name__ == "__main__":
    main()
