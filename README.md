# Sea of Stars Agents Lab

A visual-first, step-by-step reinforcement-learning curriculum built around a lightweight replica of the **Elder Mist (second encounter)** boss fight from *Sea of Stars*.

**No Sea of Stars install. No GPU. No PyTorch.** Just NumPy, Pandas, Matplotlib, Gymnasium, ipywidgets and JupyterLab.

```text
Environment = reacts.
State representation = what the chooser gets to know.
Policy = chooses.
Reward = grades outcomes.
Learning = changes future choosing from experience.
```

## Setup (Windows)

Open PowerShell in this folder and run:

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
```

It deletes any old `.venv`, creates a clean one, installs `requirements.txt`, runs `verify_setup.py` (wait for `SETUP OK`), then opens JupyterLab. After that, double-click `launch.bat` to reopen the lab.

<details>
<summary>Manual steps / other OS</summary>

```bash
python -m venv .venv
# Windows: .\.venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python verify_setup.py
python -m jupyter lab
```

Python 3.11 or 3.12 recommended.
</details>

## Notebook order

Everything lives in `notebooks/`. Run cells top to bottom with **Shift+Enter**.

### Part 0 — Gymnasium micro-course (one concept per notebook)

| # | Notebook | Concept |
|---|---|---|
| 00.01 | `00_01_State_Action_Q` | state, action, Q-table |
| 00.02 | `00_02_Transition_SPrime_Reward` | `env.step()`, s′, reward, terminated vs truncated |
| 00.03 | `00_03_Return_Gamma_Bellman` | return, γ, Bellman target |
| 00.04 | `00_04_Alpha_TD_Update` | α, TD error, the Q-update |
| 00.05 | `00_05_Epsilon_Explore_Exploit` | ε, explore vs exploit |
| 00.06 | `00_06_One_Episode_Line_By_Line` | one full episode as transitions |
| 00.07 | `00_07_Training_0_to_300` | Q-table filling in over training |
| 00.08 | `00_08_Q_Learning_GUI` | sliders for α / γ / ε |

Then `00_Gym_Basics` is the recap of all of the above.

### Part 1 — from Pong to Sea of Stars

1. `01_Toy_Pong` — a visual state vector
2. `02_Atari_Pong` — pixels as observations (intuition only; no emulator needed)
3. `03_Bridge_to_Sea_of_Stars`
4. `Elder_Mist_RL_Lab` — play it yourself, then rule policy → Q-learning → *why* a move was chosen → state-representation ablations
5. `04_POMDP_Belief_State` — hidden state, noisy observations, history, belief
6. `05_Weight_Level_Learning` — forward pass, loss, backprop, actual weight changes
7. `06_State_Representation_Research` — raw sources → candidate representations → harness → held-out evaluation → Pareto frontier

### How to read each cell

Every teaching cell is followed by a figure. Read/run the cell, look at the figure directly under it, say out loud what changed, then continue. At every step ask:

1. What information exists right now?
2. Who chooses the action?
3. What does the environment change?
4. What observation comes back?
5. Where does reward appear?
6. What actually learns?

If one of those is unclear, stop at that cell.

## Elder Mist mechanics represented

- Zale + Valere vs. Elder Mist (3,995 HP) and its Sword (800 HP)
- the Sword can be disabled and returns later
- direct body attacks while the Sword is active can trigger Counter Slash
- splash attacks aimed at the Sword damage Sword + body without that counter
- Torblado / Thunder / Healing Mist as charged, lock-based moves

Damage, MP and some timing values are normalized for teaching, not a frame-perfect reverse-engineering. No game art or assets are used. See [`docs/SOURCES.md`](docs/SOURCES.md).

## Repo layout

```text
notebooks/          the curriculum (above)
eldermist/          the Gymnasium env, policies, Q-learning, tiny MLP, plotting
lineworld.py        6-cell LineWorld env for the micro-course
learning_cards.py   color key + figures for the micro-course
visuals.py          diagram helpers used by the main notebooks
docs/               CONCEPT_MAP, WHY_MOVES_HAPPEN, FATIMAH_STATE_REP, ROADMAP, SOURCES
extras/pong/        optional real Pong / PPO / Atari scripts (own requirements.txt)
figures/            exported PNGs of every notebook figure
quick_demo.py       random vs rule vs learned policy, from the command line
tools/              verify_notebook_visuals.py (checks every teaching cell has a visual)
```

## Useful docs

- [`docs/CONCEPT_MAP.md`](docs/CONCEPT_MAP.md) — every RL term in plain language
- [`docs/WHY_MOVES_HAPPEN.md`](docs/WHY_MOVES_HAPPEN.md) — why a rule, Q, or neural policy chose an action
- [`docs/ROADMAP.md`](docs/ROADMAP.md) — what you should be able to explain after each part
- [`docs/FATIMAH_STATE_REP.md`](docs/FATIMAH_STATE_REP.md) — bridge to state-representation research
