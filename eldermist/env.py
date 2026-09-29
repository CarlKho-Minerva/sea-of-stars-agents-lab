from __future__ import annotations
from dataclasses import dataclass, asdict
from collections import Counter
import random
import numpy as np

# Optional Gymnasium compatibility. The simulator still works without Gymnasium,
# which keeps it lightweight on Windows-on-ARM laptops.
try:
    import gymnasium as gym
    from gymnasium import spaces
    BaseEnv = gym.Env
except ImportError:
    class BaseEnv:
        def reset(self, seed=None, options=None):
            if seed is not None:
                np.random.seed(seed)

    class _Discrete:
        def __init__(self, n):
            self.n = n
        def sample(self):
            return int(np.random.randint(self.n))

    class _Box:
        def __init__(self, low, high, shape, dtype):
            self.low, self.high, self.shape, self.dtype = low, high, shape, dtype

    class spaces:
        Discrete = _Discrete
        Box = _Box


LOCK_TYPES = ("sword", "blunt", "sun", "moon")

ACTIONS = [
    "normal_sword",       # Zale: sword hit, restore MP
    "sunball_sword",      # Zale: sun splash targeted at sword
    "normal_body",        # Zale/Valere: direct body hit
    "normal_blunt",       # Valere: blunt hit, restore MP
    "crescent_sword",     # Valere: moon/blunt-ish splash targeted at sword
    "lunar_shield",       # Valere only
    "mending_light",      # combo heal
]

ACTION_TO_ID = {a: i for i, a in enumerate(ACTIONS)}


@dataclass
class BattleState:
    boss_hp: int = 3995
    sword_hp: int = 800
    sword_active: bool = True
    sword_disabled_boss_turns: int = 0

    zale_hp: int = 300
    valere_hp: int = 300
    zale_mp: int = 10
    valere_mp: int = 10
    combo: int = 0
    shield_zale: int = 0
    shield_valere: int = 0

    actor: int = 0  # 0=Zale, 1=Valere
    player_actions_since_boss: int = 0
    boss_turn: int = 0
    turn: int = 0

    special: str = "torblado"
    special_countdown: int = 3
    locks_sword: int = 1
    locks_blunt: int = 1
    locks_sun: int = 1
    locks_moon: int = 2

    counter_slashes: int = 0
    specials_cancelled: int = 0


class ElderMistReplicaEnv(BaseEnv):
    """
    Educational, mechanics-inspired replica of Elder Mist Round 2.

    The environment is intentionally small enough for tabular Q-learning.
    Real-game values are used for boss/sword HP where known, while player-side
    combat numbers are normalized for fast experimentation.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        representation="full",
        reward_mode="shaped",
        seed=None,
        max_player_turns=100,
    ):
        super().__init__()
        self.representation = representation
        self.reward_mode = reward_mode
        self.max_player_turns = max_player_turns
        self.action_space = spaces.Discrete(len(ACTIONS))
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(16,), dtype=np.float32
        )
        self.rng = random.Random(seed)
        self.state = BattleState()
        self.last_event = ""

    # ---------- Public API ----------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)
        self.state = BattleState()
        self.last_event = "The rematch begins. Elder Mist is charging Torblado."
        return self._obs(), self._info()

    def step(self, action):
        action = int(action)
        if action < 0 or action >= len(ACTIONS):
            raise ValueError(action)

        s = self.state
        if self._done():
            raise RuntimeError("Episode is over. Call reset().")

        action_name = ACTIONS[action]
        valid = self.valid_actions()

        if action_name not in valid:
            # Invalid strategic action wastes the turn.
            event = f"{self.actor_name()} cannot use {action_name}; the turn is wasted."
            reward = -0.08
        else:
            event, reward = self._apply_player_action(action_name)

        # Charged spell countdown advances after every player action.
        if s.special:
            s.special_countdown -= 1

        s.turn += 1
        s.player_actions_since_boss += 1

        # Alternate player actor, skipping KO'd hero.
        self._advance_actor()

        # Special resolves as soon as its countdown reaches zero.
        if s.special and s.special_countdown <= 0:
            e2, r2 = self._resolve_special()
            event += " | " + e2
            reward += r2

        # Elder Mist gets a normal/boss turn after both heroes have had a chance.
        if not self._done() and s.player_actions_since_boss >= 2:
            s.player_actions_since_boss = 0
            e2, r2 = self._boss_turn()
            event += " | " + e2
            reward += r2

        terminated = self._done()
        truncated = s.turn >= self.max_player_turns and not terminated

        if terminated:
            if s.boss_hp <= 0:
                reward += 1.0
                event += " | VICTORY"
            else:
                reward -= 1.0
                event += " | PARTY DEFEATED"
        elif truncated:
            reward -= 0.4
            event += " | TURN LIMIT"

        if self.reward_mode == "sparse" and not terminated:
            reward = 0.0

        self.last_event = event
        return self._obs(), float(reward), terminated, truncated, self._info()

    # ---------- State / representation ----------

    def actor_name(self):
        return "Zale" if self.state.actor == 0 else "Valere"

    def valid_actions(self):
        s = self.state
        actor = self.actor_name()
        out = []

        if actor == "Zale":
            out += ["normal_sword", "normal_body"]
            if s.zale_mp >= 3:
                out += ["sunball_sword"]
        else:
            out += ["normal_blunt", "normal_body"]
            if s.valere_mp >= 3:
                out += ["crescent_sword"]
            if s.valere_mp >= 4:
                out += ["lunar_shield"]

        if s.combo >= 3:
            out += ["mending_light"]

        return out

    def compact_state(self, representation=None):
        rep = representation or self.representation
        s = self.state

        def hp_bucket(h):
            if h <= 0:
                return 0
            if h < 80:
                return 1
            if h < 160:
                return 2
            if h < 240:
                return 3
            return 4

        def boss_bucket(h, mx):
            if h <= 0:
                return 0
            frac = h / mx
            return min(5, max(1, int(np.ceil(frac * 5))))

        lock_total = self._lock_total()
        base = {
            "actor": s.actor,
            "z_hp": hp_bucket(s.zale_hp),
            "v_hp": hp_bucket(s.valere_hp),
            "boss_hp": boss_bucket(s.boss_hp, 3995),
            "sword": int(s.sword_active),
            "sword_hp": boss_bucket(s.sword_hp, 800) if s.sword_active else 0,
            "special": {"":0, "torblado":1, "thunder":2, "healing_mist":3}.get(s.special, 0),
            "special_cd": max(0, s.special_countdown),
            "locks": min(lock_total, 5),
            "combo": min(s.combo, 3),
            "z_mp": min(s.zale_mp // 3, 3),
            "v_mp": min(s.valere_mp // 3, 3),
            "shield": int(s.shield_zale > 0) + 2 * int(s.shield_valere > 0),
        }

        if rep == "full":
            keys = [
                "actor","z_hp","v_hp","boss_hp","sword","sword_hp",
                "special","special_cd","locks","combo","z_mp","v_mp","shield"
            ]
        elif rep == "no_sword":
            keys = [
                "actor","z_hp","v_hp","boss_hp",
                "special","special_cd","locks","combo","z_mp","v_mp","shield"
            ]
        elif rep == "no_locks":
            keys = [
                "actor","z_hp","v_hp","boss_hp","sword","sword_hp",
                "combo","z_mp","v_mp","shield"
            ]
        elif rep == "minimal":
            keys = ["actor","z_hp","v_hp","boss_hp"]
        else:
            raise ValueError(f"Unknown representation: {rep}")

        return tuple(base[k] for k in keys)

    def _obs(self):
        # Fixed-length numeric observation for neural methods.
        s = self.state
        lock_total = self._lock_total()
        special_id = {"":0, "torblado":1, "thunder":2, "healing_mist":3}.get(s.special, 0)
        arr = np.array([
            s.boss_hp / 3995,
            (s.sword_hp / 800) if s.sword_active else 0,
            float(s.sword_active),
            s.zale_hp / 300,
            s.valere_hp / 300,
            s.zale_mp / 10,
            s.valere_mp / 10,
            min(s.combo, 3) / 3,
            float(s.actor),
            special_id / 3,
            max(s.special_countdown, 0) / 3,
            lock_total / 5,
            s.locks_sword / 5,
            s.locks_blunt / 5,
            s.locks_sun / 5,
            s.locks_moon / 5,
        ], dtype=np.float32)
        return arr

    def _info(self):
        return {
            "state": asdict(self.state),
            "compact_state": self.compact_state(),
            "valid_actions": self.valid_actions(),
            "event": self.last_event,
            "actor": self.actor_name(),
        }

    # ---------- Player actions ----------

    def _apply_player_action(self, action):
        s = self.state
        actor = self.actor_name()
        reward = -0.01  # prefer shorter solutions
        event = f"{actor}: {action}"

        # Normal attacks restore MP and combo.
        if action == "normal_sword":
            s.zale_mp = min(10, s.zale_mp + 2)
            s.combo = min(3, s.combo + 1)
            if s.sword_active:
                dmg = 185
                s.sword_hp -= dmg
                reward += 0.02
                event += f" -> Sword -{dmg}"
            else:
                dmg = 165
                s.boss_hp -= dmg
                reward += 0.02
                event += f" -> Boss -{dmg}"
            self._break_locks({"sword": 1})

        elif action == "normal_blunt":
            s.valere_mp = min(10, s.valere_mp + 2)
            s.combo = min(3, s.combo + 1)
            if s.sword_active:
                dmg = 175
                s.sword_hp -= dmg
                reward += 0.02
                event += f" -> Sword -{dmg}"
            else:
                dmg = 160
                s.boss_hp -= dmg
                reward += 0.02
                event += f" -> Boss -{dmg}"
            self._break_locks({"blunt": 1})

        elif action == "sunball_sword":
            s.zale_mp -= 3
            if s.sword_active:
                sword_dmg, boss_dmg = 260, 115
                s.sword_hp -= sword_dmg
                s.boss_hp -= boss_dmg
                reward += 0.05
                event += f" -> Sword -{sword_dmg}, splash Boss -{boss_dmg} (no Counter Slash)"
            else:
                dmg = 245
                s.boss_hp -= dmg
                reward += 0.04
                event += f" -> Boss -{dmg}"
            self._break_locks({"sun": 1})

        elif action == "crescent_sword":
            s.valere_mp -= 3
            if s.sword_active:
                sword_dmg, boss_dmg = 235, 105
                s.sword_hp -= sword_dmg
                s.boss_hp -= boss_dmg
                reward += 0.05
                event += f" -> Sword -{sword_dmg}, splash Boss -{boss_dmg} (no Counter Slash)"
            else:
                dmg = 225
                s.boss_hp -= dmg
                reward += 0.04
                event += f" -> Boss -{dmg}"
            self._break_locks({"moon": 1, "blunt": 1})

        elif action == "normal_body":
            dmg = 205 if actor == "Zale" else 195
            s.boss_hp -= dmg
            reward += 0.025
            event += f" -> Boss -{dmg}"

            if actor == "Zale":
                s.zale_mp = min(10, s.zale_mp + 2)
                self._break_locks({"sword": 1})
            else:
                s.valere_mp = min(10, s.valere_mp + 2)
                self._break_locks({"blunt": 1})
            s.combo = min(3, s.combo + 1)

            if s.sword_active and s.boss_hp > 0:
                self._damage_party(60, both=True)
                s.counter_slashes += 1
                reward -= 0.18
                event += " -> COUNTER SLASH! both -60"

        elif action == "lunar_shield":
            s.valere_mp -= 4
            s.shield_zale = 1
            s.shield_valere = 1
            s.zale_hp = min(300, s.zale_hp + 25)
            s.valere_hp = min(300, s.valere_hp + 25)
            reward += 0.025
            event += " -> shields both + small heal"
            self._break_locks({"moon": 1})

        elif action == "mending_light":
            s.combo -= 3
            s.zale_hp = min(300, s.zale_hp + 115)
            s.valere_hp = min(300, s.valere_hp + 115)
            reward += 0.04
            event += " -> both heal +115"

        self._check_sword_disabled()
        s.boss_hp = max(0, s.boss_hp)
        return event, reward

    # ---------- Boss logic ----------

    def _boss_turn(self):
        s = self.state
        s.boss_turn += 1
        reward = 0.0

        if not s.sword_active and s.sword_disabled_boss_turns > 0:
            s.sword_disabled_boss_turns -= 1
            if s.sword_disabled_boss_turns == 0 and s.boss_hp > 0:
                s.sword_active = True
                s.sword_hp = 489
                revive = " Sword revives at 489 HP."
            else:
                revive = ""
        else:
            revive = ""

        # If no special is charging, normal fist and maybe start next special.
        target = "Zale" if self.rng.random() < 0.5 else "Valere"
        self._damage_target(target, 48)
        event = f"Elder Mist Fist Smash -> {target} -48.{revive}"

        if not s.special and self.rng.random() < 0.62:
            self._start_random_special()
            event += f" Starts charging {s.special} ({self._lock_total()} locks, {s.special_countdown} player turns)."

        return event, reward

    def _start_random_special(self):
        s = self.state
        move = self.rng.choices(
            ["torblado", "thunder", "healing_mist"],
            weights=[0.38, 0.30, 0.32],
        )[0]
        s.special = move

        if move == "torblado":
            # 5 locks drawn from Sword/Blunt/Sun/Moon.
            c = Counter(self.rng.choice(LOCK_TYPES) for _ in range(5))
            s.special_countdown = 3
        elif move == "thunder":
            c = Counter({"moon": 3})
            s.special_countdown = 2
        else:
            # Teaching approximation of the real mixed 3-lock heal patterns.
            patterns = [
                ["sword", "blunt", "sword"],
                ["sword", "sun", "sword"],
                ["blunt", "sword", "blunt"],
                ["blunt", "moon", "blunt"],
            ]
            c = Counter(self.rng.choice(patterns))
            s.special_countdown = 2

        s.locks_sword = c["sword"]
        s.locks_blunt = c["blunt"]
        s.locks_sun = c["sun"]
        s.locks_moon = c["moon"]

    def _resolve_special(self):
        s = self.state
        move = s.special
        remaining = self._lock_total()

        if remaining <= 0:
            s.specials_cancelled += 1
            event = f"{move} CANCELLED by breaking all locks"
            reward = 0.12
        else:
            # Fewer remaining locks => reduced effect.
            if move == "torblado":
                dmg = int(125 * (0.45 + 0.11 * remaining))
                target = "Zale" if self.rng.random() < 0.5 else "Valere"
                self._damage_target(target, dmg)
                event = f"Torblado fires with {remaining} locks -> {target} -{dmg}"
                reward = -0.04 * remaining

            elif move == "thunder":
                dmg = int(58 * (0.45 + 0.18 * remaining))
                self._damage_party(dmg, both=True)
                event = f"Thunder fires with {remaining} locks -> both -{dmg}"
                reward = -0.035 * remaining

            else:
                # Prefer healing whichever component is more damaged.
                boss_missing = 3995 - s.boss_hp
                sword_missing = (800 - s.sword_hp) if s.sword_active else 0
                scale = 0.55 + 0.15 * remaining
                if s.sword_active and sword_missing > boss_missing * 0.35:
                    amount = int(210 * scale)
                    s.sword_hp = min(800, s.sword_hp + amount)
                    event = f"Healing Mist restores Sword +{amount}"
                else:
                    amount = int(150 * scale)
                    s.boss_hp = min(3995, s.boss_hp + amount)
                    event = f"Healing Mist restores Elder Mist +{amount}"
                reward = -0.03 * remaining

        s.special = ""
        s.special_countdown = 0
        s.locks_sword = s.locks_blunt = s.locks_sun = s.locks_moon = 0
        return event, reward

    # ---------- Helpers ----------

    def _break_locks(self, damage_types):
        s = self.state
        if not s.special:
            return

        mapping = {
            "sword": "locks_sword",
            "blunt": "locks_blunt",
            "sun": "locks_sun",
            "moon": "locks_moon",
        }

        for typ, count in damage_types.items():
            field = mapping[typ]
            current = getattr(s, field)
            setattr(s, field, max(0, current - count))

        if self._lock_total() == 0:
            # Cancellation is resolved immediately; countdown stays but special
            # resolution will happen after the action in step().
            s.special_countdown = 0

    def _lock_total(self):
        s = self.state
        return s.locks_sword + s.locks_blunt + s.locks_sun + s.locks_moon

    def lock_dict(self):
        s = self.state
        return {
            "sword": s.locks_sword,
            "blunt": s.locks_blunt,
            "sun": s.locks_sun,
            "moon": s.locks_moon,
        }

    def _check_sword_disabled(self):
        s = self.state
        if s.sword_active and s.sword_hp <= 0:
            s.sword_active = False
            s.sword_hp = 0
            s.sword_disabled_boss_turns = 3

    def _advance_actor(self):
        s = self.state
        if s.zale_hp <= 0 and s.valere_hp <= 0:
            return
        if s.actor == 0:
            s.actor = 1 if s.valere_hp > 0 else 0
        else:
            s.actor = 0 if s.zale_hp > 0 else 1

    def _damage_target(self, target, amount):
        s = self.state
        if target == "Zale":
            if s.zale_hp <= 0:
                target = "Valere"
            elif s.shield_zale > 0:
                s.shield_zale -= 1
                return
        if target == "Valere":
            if s.valere_hp <= 0:
                target = "Zale"
            elif s.shield_valere > 0:
                s.shield_valere -= 1
                return

        if target == "Zale":
            s.zale_hp = max(0, s.zale_hp - amount)
        else:
            s.valere_hp = max(0, s.valere_hp - amount)

    def _damage_party(self, amount, both=True):
        self._damage_target("Zale", amount)
        self._damage_target("Valere", amount)

    def _done(self):
        s = self.state
        return s.boss_hp <= 0 or (s.zale_hp <= 0 and s.valere_hp <= 0)

    # ---------- Text render ----------

    def render(self):
        s = self.state
        locks = self.lock_dict()
        lines = [
            f"Turn {s.turn} | Actor: {self.actor_name()}",
            f"Elder Mist: {s.boss_hp:4d}/3995",
            f"Sword: {'ACTIVE' if s.sword_active else 'DOWN':6s} {s.sword_hp:3d}/800",
            f"Zale:   HP {s.zale_hp:3d}/300  MP {s.zale_mp:2d}/10  shield={s.shield_zale}",
            f"Valere: HP {s.valere_hp:3d}/300  MP {s.valere_mp:2d}/10  shield={s.shield_valere}",
            f"Combo: {s.combo}/3",
            f"Special: {s.special or '-'}  cd={s.special_countdown}  locks={locks}",
            f"Valid actions: {', '.join(self.valid_actions())}",
            f"Last event: {self.last_event}",
        ]
        return "\n".join(lines)
