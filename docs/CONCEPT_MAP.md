# RL concept map using Elder Mist

## Environment

The environment is **everything outside the policy that reacts to actions**.

In this lab:

```text
ElderMistReplicaEnv
```

owns:
- boss HP
- Sword HP / active state
- player HP / MP
- charged moves
- locks
- boss attacks
- Counter Slash
- victory / defeat

The policy does **not** decide how much damage Sunball causes. The environment does.

## `reset()`

```python
obs, info = env.reset()
```

means:

> create a new episode and give the policy its first observation.

It does not mean “there is no state.”

## `step(action)`

```python
next_obs, reward, terminated, truncated, info = env.step(action)
```

means:

> apply one action to the environment, let the world change, then report what the policy can observe next.

## State

The complete state is everything needed for the simulator to determine future dynamics.

Example:

```text
boss_hp=2600
sword_active=True
sword_hp=310
zale_hp=180
valere_hp=220
current_actor=Valere
special=Thunder
moon_locks=2
...
```

## Observation

What the agent receives right now.

A state can contain facts that an observation hides.

That distinction becomes crucial in a POMDP.

## State representation

The information we actually give the decision-maker.

Examples:

```text
FULL:
actor + HP + Sword + locks + MP + shields

MINIMAL:
actor + HP + boss HP
```

Both can describe the same true world, but one throws away more information.

## Policy π

The mapping:

```text
information available now -> action
```

Examples:
- you clicking a move
- an `if` statement
- a Q-table
- a neural network
- an LLM + prompt + RAG + tools + planner
- an entire harness

A policy is a **role**, not a particular implementation.

## Transition T

How the world changes when an action occurs.

In a commercial game, the game engine implements this.

In this replica, `env.step()` implements it.

## Reward R

A numeric training signal.

Sparse:

```text
win  +1
lose -1
else  0
```

Shaped:

```text
small reward for progress
small penalty for damage / wasted turns
big terminal reward for victory
```

## Value

`V(s)` asks:

> How good is it to be in this state, considering likely future reward?

`Q(s,a)` asks:

> How good is taking this action in this state, considering likely future reward?

## Learning

Learning means the behavior-producing parameters change from experience.

Q-learning:
- Q-table numbers change

Neural learning:
- network weights change

Prompting/RAG:
- model weights do **not** necessarily change; the policy's inputs/configuration change instead.
