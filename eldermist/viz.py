import matplotlib.pyplot as plt
from .env import ACTIONS

def _bar(ax, x, y, w, h, frac, label):
    frac = max(0, min(1, frac))
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, linewidth=1.5))
    ax.add_patch(plt.Rectangle((x, y), w * frac, h, alpha=0.35))
    ax.text(x, y + h + 0.03, label, fontsize=10, va="bottom")

def plot_state(env, title=None):
    s = env.state
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    _bar(ax, 0.8, 4.8, 8.4, 0.35, s.boss_hp / 3995, f"Elder Mist {s.boss_hp}/3995")
    _bar(ax, 0.8, 4.05, 8.4, 0.28, (s.sword_hp / 800 if s.sword_active else 0),
         f"Sword {'ACTIVE' if s.sword_active else 'DISABLED'} {s.sword_hp}/800")

    _bar(ax, 0.8, 2.8, 3.7, 0.32, s.zale_hp / 300, f"Zale HP {s.zale_hp}/300 | MP {s.zale_mp}/10")
    _bar(ax, 5.5, 2.8, 3.7, 0.32, s.valere_hp / 300, f"Valere HP {s.valere_hp}/300 | MP {s.valere_mp}/10")

    locks = env.lock_dict()
    special = s.special or "none"
    ax.text(0.8, 1.75, f"Actor: {env.actor_name()}    Combo: {s.combo}/3", fontsize=12)
    ax.text(
        0.8, 1.25,
        f"Charged move: {special}   countdown={s.special_countdown}   "
        f"locks S:{locks['sword']} B:{locks['blunt']} ☀:{locks['sun']} ☾:{locks['moon']}",
        fontsize=11,
    )
    ax.text(0.8, 0.55, env.last_event, fontsize=9, wrap=True)

    ax.set_title(title or f"Turn {s.turn}", fontsize=14)
    plt.tight_layout()
    return fig

def rollout(env, policy, max_steps=100):
    obs, _ = env.reset()
    history = []
    for _ in range(max_steps):
        history.append({
            "state": env.state.__dict__.copy(),
            "event": env.last_event,
            "actor": env.actor_name(),
            "valid": list(env.valid_actions()),
        })
        action = policy(env, obs)
        obs, r, terminated, truncated, info = env.step(action)
        history[-1]["action"] = ACTIONS[action]
        history[-1]["reward"] = r
        if terminated or truncated:
            history.append({
                "state": env.state.__dict__.copy(),
                "event": env.last_event,
                "actor": env.actor_name(),
                "valid": list(env.valid_actions()),
                "action": None,
                "reward": 0,
            })
            break
    return history
