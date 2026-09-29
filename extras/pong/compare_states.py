import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from pong_env import ToyPongEnv


def evaluate(model, representation, episodes=30):
    env = ToyPongEnv(representation=representation)
    rewards, lengths, wins = [], [], []

    for _ in range(episodes):
        obs, info = env.reset()
        total = 0.0
        steps = 0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total += reward
            steps += 1
            if terminated or truncated:
                break

        rewards.append(total)
        lengths.append(steps)
        wins.append(int(info["player_score"] > info["opponent_score"]))

    env.close()
    return {
        "mean_reward": float(np.mean(rewards)),
        "win_rate": float(np.mean(wins)),
        "mean_episode_steps": float(np.mean(lengths)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=50_000)
    args = parser.parse_args()

    Path("comparison_models").mkdir(exist_ok=True)
    Path("plots").mkdir(exist_ok=True)

    rows = []
    for rep in ["full", "no_velocity", "minimal"]:
        print(f"\n=== Training representation: {rep} ===")
        env = Monitor(ToyPongEnv(representation=rep))
        model = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            n_steps=1024,
            batch_size=64,
            policy_kwargs=dict(net_arch=[64, 64]),
        )
        model.learn(args.steps)
        model.save(f"comparison_models/{rep}")
        metrics = evaluate(model, rep)
        rows.append({"representation": rep, **metrics})
        env.close()

    df = pd.DataFrame(rows)
    df.to_csv("state_representation_comparison.csv", index=False)
    print("\n", df.to_string(index=False))

    plt.figure(figsize=(7, 4))
    plt.bar(df["representation"], df["win_rate"])
    plt.ylim(0, 1)
    plt.ylabel("Win rate")
    plt.title("State representation comparison")
    plt.tight_layout()
    plt.savefig("plots/state_representation_comparison.png", dpi=160)
    print("\nSaved state_representation_comparison.csv")
    print("Saved plots/state_representation_comparison.png")


if __name__ == "__main__":
    main()
