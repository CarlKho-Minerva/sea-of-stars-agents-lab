import random
from .env import ACTIONS, ACTION_TO_ID

def random_policy(env, obs=None):
    return ACTION_TO_ID[random.choice(env.valid_actions())]

def explain_sensible_rule_policy(env):
    s = env.state
    valid = env.valid_actions()
    actor = env.actor_name()

    if s.combo >= 3 and min(s.zale_hp, s.valere_hp) < 125 and "mending_light" in valid:
        return "mending_light", (
            "Party HP is dangerously low and combo=3, so the policy spends the combo "
            "on Mending Light instead of taking more damage."
        )

    if actor == "Valere" and min(s.zale_hp, s.valere_hp) < 180 and "lunar_shield" in valid:
        return "lunar_shield", (
            "Valere is acting, somebody is below 180 HP, and Lunar Shield is affordable. "
            "The policy chooses protection because surviving the next boss action has high value."
        )

    if s.sword_active:
        if actor == "Zale" and "sunball_sword" in valid:
            return "sunball_sword", (
                "The Sword is active. Directly hitting Elder Mist's body would trigger Counter Slash. "
                "Sunball targets the Sword while splashing damage onto the boss, so this baseline prefers it."
            )
        if actor == "Valere" and "crescent_sword" in valid:
            return "crescent_sword", (
                "The Sword is active. Crescent Sword safely damages the Sword and splashes the boss, "
                "avoiding the direct-body Counter Slash rule."
            )
        fallback = "normal_sword" if actor == "Zale" else "normal_blunt"
        return fallback, (
            "The Sword is active but the stronger splash skill is unavailable/too expensive, "
            "so the policy uses the actor's normal attack on the Sword."
        )

    if s.special:
        locks = env.lock_dict()
        if actor == "Zale":
            if locks["sun"] > 0 and "sunball_sword" in valid:
                return "sunball_sword", (
                    f"{s.special} is charging and still has a Sun lock. "
                    "Zale can break that lock with Sunball."
                )
            if locks["sword"] > 0:
                return "normal_body", (
                    f"{s.special} is charging and still has a Sword lock. "
                    "With the Sword target disabled, Zale's normal body hit breaks that lock."
                )
        else:
            if locks["moon"] > 0 and "crescent_sword" in valid:
                return "crescent_sword", (
                    f"{s.special} is charging and still has a Moon lock. "
                    "Valere can break that lock with Crescent."
                )
            if locks["blunt"] > 0:
                return "normal_body", (
                    f"{s.special} is charging and still has a Blunt lock. "
                    "Valere's normal hit can break it."
                )

    return "normal_body", (
        "The Sword is disabled and there is no higher-priority survival/lock-breaking need, "
        "so the policy directly damages Elder Mist."
    )

def sensible_rule_policy(env, obs=None):
    action, _ = explain_sensible_rule_policy(env)
    return ACTION_TO_ID[action]
