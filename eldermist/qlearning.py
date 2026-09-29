from collections import defaultdict
import random
import numpy as np
from .env import ACTION_TO_ID

class QTable:
    def __init__(self, alpha=0.15, gamma=0.97):
        self.alpha = alpha
        self.gamma = gamma
        self.q = defaultdict(dict)

    def value(self, s, a):
        return self.q[s].get(a, 0.0)

    def choose(self, s, valid_action_ids, epsilon=0.1):
        if random.random() < epsilon:
            return random.choice(valid_action_ids)
        vals = [self.value(s, a) for a in valid_action_ids]
        best = max(vals)
        choices = [a for a, v in zip(valid_action_ids, vals) if v == best]
        return random.choice(choices)

    def update(self, s, a, r, s2, valid2, done):
        old = self.value(s, a)
        future = 0.0 if done or not valid2 else max(self.value(s2, x) for x in valid2)
        target = r + self.gamma * future
        self.q[s][a] = old + self.alpha * (target - old)

def valid_ids(env):
    return [ACTION_TO_ID[a] for a in env.valid_actions()]

def train_q(
    env_factory,
    episodes=5000,
    alpha=0.15,
    gamma=0.97,
    eps_start=0.45,
    eps_end=0.03,
    seed=0,
):
    random.seed(seed)
    np.random.seed(seed)
    q = QTable(alpha, gamma)
    rows = []

    for ep in range(episodes):
        env = env_factory()
        env.reset(seed=seed + ep)
        total = 0.0
        counters_before = env.state.counter_slashes

        frac = ep / max(1, episodes - 1)
        eps = eps_start + frac * (eps_end - eps_start)

        for t in range(env.max_player_turns):
            s = env.compact_state()
            vids = valid_ids(env)
            a = q.choose(s, vids, eps)

            _, r, terminated, truncated, _ = env.step(a)
            total += r
            done = terminated or truncated
            s2 = env.compact_state()
            vids2 = [] if done else valid_ids(env)
            q.update(s, a, r, s2, vids2, done)

            if done:
                break

        rows.append({
            "episode": ep,
            "return": total,
            "win": int(env.state.boss_hp <= 0),
            "turns": env.state.turn,
            "counter_slashes": env.state.counter_slashes - counters_before,
            "specials_cancelled": env.state.specials_cancelled,
            "epsilon": eps,
        })

    return q, rows

def q_policy(q, env):
    s = env.compact_state()
    vids = valid_ids(env)
    vals = [q.value(s, a) for a in vids]
    return vids[int(np.argmax(vals))]


def explain_q_choice(q, env):
    """
    Return all legal actions sorted by learned Q-value, plus the chosen action.
    """
    s = env.compact_state()
    vids = valid_ids(env)
    rows = [
        {
            "action_id": a,
            "action": env_action_name(a),
            "q_value": q.value(s, a),
        }
        for a in vids
    ]
    rows.sort(key=lambda x: x["q_value"], reverse=True)
    return {
        "represented_state": s,
        "chosen_action": rows[0]["action"] if rows else None,
        "ranked_actions": rows,
    }

def env_action_name(action_id):
    from .env import ACTIONS
    return ACTIONS[action_id]
