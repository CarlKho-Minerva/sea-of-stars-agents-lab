# RL Pong Lab

A tiny hands-on RL lab for learning the loop:

**play it yourself → train headless → watch the learned policy → inspect learning curves**

This repo has two paths:

1. **Toy Pong** — recommended first. Numeric state, fast training, easy to understand.
2. **Real Atari Pong** — optional. Pixel observations and a CNN policy; much slower.

## 0. Setup

```bash
python -m venv .venv
source .venv/bin/activate     # macOS/Linux
# .venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

## 1. Play Toy Pong yourself

```bash
python play.py
```

Controls:

- Up / W = move paddle up
- Down / S = move paddle down
- R = restart episode
- Esc = quit

Watch the terminal. Every step is:

```text
observation -> your action -> environment transition -> new observation -> reward
```

## 2. Train an agent headlessly

```bash
python train.py --steps 120000
```

This uses PPO with an MLP policy. There is no rendering during training, so it runs quickly.

Outputs:

- `models/pong_ppo.zip`
- `logs/monitor.csv`
- `plots/training_curve.png`

## 3. Watch the trained agent

```bash
python watch.py
```

The policy now maps:

```text
[player_y, ball_x, ball_y, ball_vx, ball_vy, opponent_y]
                         ↓
                       MLP
                         ↓
                 up / stay / down
```

## 4. Inspect the representation experiment

```bash
python compare_states.py --steps 50000
```

This trains the same PPO algorithm with different observation/state representations:

- `full`: all available numeric state
- `no_velocity`: removes ball velocity
- `minimal`: only paddle Y + ball X/Y

The script compares:

- evaluation reward
- points won
- episode length

This is the same experimental pattern you can later use for τ²-bench or screenshot/accessibility-tree state representations.

## 5. TensorBoard

During or after training:

```bash
tensorboard --logdir tensorboard
```

Open the local URL it prints.

## 6. Real Atari Pong (optional)

Install Atari dependencies:

```bash
pip install "gymnasium[atari,accept-rom-license]" ale-py
```

Then:

```bash
python atari_pong_train.py --steps 1000000
```

This is substantially heavier. The observations are raw frames and the policy uses a CNN.

Toy Pong is intentionally better for learning the concepts because you can directly inspect every state variable.

## What `reset()` means

```python
obs, info = env.reset()
```

means:

1. create a fresh episode with a real internal state,
2. return the first observation visible to the agent.

It does **not** mean the world has no state.

## What `step()` means

```python
obs, reward, terminated, truncated, info = env.step(action)
```

means:

1. apply `action`,
2. update the hidden/internal environment state,
3. compute reward,
4. return the next observation.

That is the RL loop.
