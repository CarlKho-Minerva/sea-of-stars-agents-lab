# Why did that move happen?

There are three different reasons an agent can choose a move. Keep them separate.

## 1. Rule policy

If the action came from `sensible_rule_policy`, the explanation is human-readable.

Example:

```text
Sword active + Zale + Sunball affordable
→ choose Sunball on Sword
```

Why?

Because **we wrote that rule**.

There is no learning.

## 2. Q-learning policy

For a learned Q-table:

```text
current compact state:
(actor=Zale, sword=1, party_hp=mid, boss_hp=high, ...)

legal action      Q-value
--------------------------------
sunball_sword      0.83
normal_sword       0.54
normal_body       -0.42
```

The policy chooses `sunball_sword` because it currently has the largest estimated long-term return.

Why does it have 0.83?

Because repeated trajectories updated it using:

```text
reward now + discounted estimate of future reward
```

The Q-table is the learned memory.

## 3. Neural policy

A neural policy receives numbers and computes:

```text
input
→ weighted sums
→ hidden activations
→ output logits
→ action probabilities
```

The explanation is no longer a single explicit rule.

Training changes weights so actions associated with better training signals become more likely.

## Important distinction

```text
"We chose Sunball because the rule says Sword active"
```

is not the same kind of explanation as:

```text
"Sunball has the largest learned Q-value in this represented state"
```

and neither is the same as:

```text
"a neural policy assigns Sunball 72% probability"
```

They may produce the same move using completely different mechanisms.
