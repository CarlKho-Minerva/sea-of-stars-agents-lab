import time
from stable_baselines3 import PPO
from pong_env import ToyPongEnv

env = ToyPongEnv(render_mode="human", representation="full")
model = PPO.load("models/pong_ppo")

obs, info = env.reset()
episode_reward = 0.0

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(int(action))
    episode_reward += reward

    if terminated or truncated:
        print(
            f"episode reward={episode_reward:.2f}, "
            f"score={info['player_score']}-{info['opponent_score']}"
        )
        episode_reward = 0.0
        obs, info = env.reset()

env.close()
