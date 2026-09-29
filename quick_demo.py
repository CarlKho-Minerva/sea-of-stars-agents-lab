import pandas as pd
from eldermist.env import ElderMistReplicaEnv
from eldermist.policies import random_policy, sensible_rule_policy
from eldermist.qlearning import train_q, q_policy

def evaluate(policy, seeds=range(100)):
    rows = []
    for seed in seeds:
        env = ElderMistReplicaEnv()
        obs, _ = env.reset(seed=seed)
        total = 0
        while True:
            a = policy(env, obs)
            obs, r, term, trunc, _ = env.step(a)
            total += r
            if term or trunc:
                break
        rows.append({
            "win": int(env.state.boss_hp <= 0),
            "turns": env.state.turn,
            "counter_slashes": env.state.counter_slashes,
            "return": total,
        })
    return pd.DataFrame(rows)

print("Random baseline")
print(evaluate(random_policy).mean(numeric_only=True))

print("\nReadable rule policy")
print(evaluate(sensible_rule_policy).mean(numeric_only=True))

print("\nTraining Q-learning...")
q, train_rows = train_q(lambda: ElderMistReplicaEnv(), episodes=5000, seed=42)

def learned(env, obs):
    return q_policy(q, env)

print("\nLearned policy")
print(evaluate(learned, range(1000, 1100)).mean(numeric_only=True))
