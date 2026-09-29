# Sea of Stars Agents Lab

A step-by-step reinforcement-learning curriculum built around a lightweight replica of **Elder Mist Round 2**.

**No Sea of Stars install is required for the main lab. No GPU is required.**

Start here:

➡️ **[START_HERE.md](START_HERE.md)**

## What you will actually learn

This repo is ordered so each abstraction becomes concrete before the next one appears.

```text
00 Gym basics
   reset() / step() / reward / policy
        ↓
Elder Mist manual play
   YOU are the policy
        ↓
Rule policy
   a policy can just be if-statements
        ↓
Q-learning
   behavior changes because Q-values change from reward
        ↓
State representation
   hide Sword/locks and measure what breaks
        ↓
POMDP + belief state
   hidden truth + noisy observations + history
        ↓
Weight-level neural learning
   forward pass + loss + backprop + actual weight updates
        ↓
Fatimah / Observable Intuition research design
   screenshot + accessibility tree + history → candidate state → harness → action
```

## Recommended notebook order

1. `notebooks/00_Gym_Basics.ipynb`
2. `notebooks/Elder_Mist_RL_Lab.ipynb`
3. `notebooks/04_POMDP_Belief_State.ipynb`
4. `notebooks/05_Weight_Level_Learning.ipynb`
5. `notebooks/06_State_Representation_Research.ipynb`

Optional later:

6. `notebooks/01_Toy_Pong.ipynb`
7. `notebooks/02_Atari_Pong.ipynb`
8. `notebooks/03_Bridge_to_Sea_of_Stars.ipynb`

## Elder Mist mechanics represented

The simulator preserves the strategically important structure of the second Elder Mist encounter:

- Zale + Valere
- Elder Mist: 3,995 HP
- Sword: 800 HP
- Sword can be disabled and returns later
- direct body attacks while Sword is active can trigger Counter Slash
- splash attacks aimed at Sword can damage Sword + body without that counter
- Torblado / Thunder / Healing Mist are represented as charged lock-based moves

Damage, MP, and some timing values are normalized for teaching rather than claimed to be a frame-perfect reverse engineering of the commercial game.

See `SOURCES.md`.

## Useful docs

- `docs/CONCEPT_MAP.md` — every RL term in plain language
- `docs/WHY_MOVES_HAPPEN.md` — why a rule policy, Q-policy, or neural policy chose an action
- `docs/FATIMAH_STATE_REP.md` — direct bridge to state-representation research
- `START_HERE.md` — exact install/run instructions

## One-line mental model

```text
Environment = reacts.
State representation = what the chooser gets to know.
Policy = chooses.
Reward = grades outcomes.
Learning = changes future choosing from experience.
```
