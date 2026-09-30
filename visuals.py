"""Shared visuals for every notebook in the lab.

One color per RL concept, used everywhere: the chip legend, the concept cards,
the flow diagrams and the plots. If a box, chip or bar is blue it is STATE,
orange it is ACTION, green it is REWARD, in every notebook.

Importing this module also applies one matplotlib style, so all figures match.
"""
import re

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from IPython.display import HTML, display

COLORS = {
    # micro-course concepts (00.01-00.08)
    "STATE": "#4C78A8",
    "ACTION": "#F58518",
    "Q": "#ECA82C",
    "TRANSITION": "#B279A2",
    "S PRIME": "#5B8FF9",
    "REWARD": "#54A24B",
    "TERMINATED": "#6B7280",
    "TRUNCATED": "#9CA3AF",
    "INFO": "#64748B",
    "GAMMA": "#00A6A6",
    "TARGET": "#22A06B",
    "TD ERROR": "#C026D3",
    "ALPHA": "#E45756",
    "EPSILON": "#FF6B9A",
    "EXPLORE": "#9C6ADE",
    "EXPLOIT": "#2F80ED",
    "EPISODE": "#8B5CF6",
    # concepts the later notebooks add
    "ENVIRONMENT": "#334155",
    "OBSERVATION": "#17BECF",
    "POLICY": "#0F766E",
    "REPRESENTATION": "#1F3A93",
    "HISTORY": "#8C564B",
    "BELIEF": "#A3A01B",
    "WEIGHTS": "#E377C2",
    "LOSS": "#C026D3",
    "LEARNING": "#E45756",
    "KEY IDEA": "#64748B",
}

# First matching rule wins, so the specific concepts come before the broad ones.
_RULES = [
    ("TD ERROR", r"TD ERROR"),
    ("BELIEF", r"BELIEF|BAYES|POSTERIOR|PRIOR"),
    ("HISTORY", r"HISTORY|MEMORY|FRAME T|O[₁₂₃₄]"),
    ("REPRESENTATION", r"REPRESENTATION|Zₜ|TRANSFORM|ALIAS|\bFULL\b|NO SWORD|NO LOCKS|MINIMAL|NO VELOCITY"
                       r"|KEEP WHAT MATTERS|NUMBERS|NUMERIC|STATE VECTOR|ENCODER|FEATURE|LATENT|HIDE A|ONLY REP"),
    ("EXPLOIT", r"ARGMAX|GREEDY"),
    ("Q", r"\bQ\b|Q-"),
    ("LOSS", r"LOSS|GRADIENT|BACKPROP"),
    ("WEIGHTS", r"W[₁₂12]|WEIGHT|\bW BEFORE|\bW AFTER|HIDDEN UNITS|NETWORK|LOGITS|SOFTMAX|\bINPUT|FORWARD PASS|NEURAL"),
    ("S PRIME", r"NEXT STATE|NEW STATE|NEW OBS|NEXT BATTLE|S′"),
    ("REWARD", r"REWARD|WIN RATE|SUCCESS|OUTCOME|RETURN|MEASURE|COMPARE|HELD-OUT|PARETO|TOKENS|LATENCY|ERRORS|ASSERT"),
    ("OBSERVATION", r"OBSERVATION|OBSERVAB|AMBIGUOUS|\bOBS\b|OBSERVE|PIXELS|SCREEN|A11Y|SENSOR|NOISY|\bRAW\b|SOURCES|TELEMETRY"),
    ("ACTION", r"\bACTIONS?\b|\bACT\b|\bMOVE|UP / STAY|SAFE / RISKY"),
    ("TRANSITION", r"TRANSITION|STEP|WORLD CHANGES|REACTS|CONSEQUENCE|↓"),
    ("POLICY", r"POLICY|Π|YOUR BRAIN|RULES|IF/ELSE|HARNESS|PLANNER|\bLLM\b|LEARNER|CHOOS|TEACHER|CLONING|DEMONSTRATION"),
    ("STATE", r"STATE|BATTLE|POSITION"),
    ("EPISODE", r"EPISODE|REPEAT|FIGHTS|EXPERIENCE|ROLLOUT|LOOP"),
    ("LEARNING", r"LEARN|UPDATE|TRAIN"),
    ("ENVIRONMENT", r"ENVIRONMENT|WORLD|ELDER MIST|PONG|ATARI|LINEWORLD|LIFEOS|GAME|PADDLE|BALL|TASK|RESET|GYM"),
]


def concept_of(text, default="INFO"):
    """Which concept a label is about, e.g. concept_of("NEXT STATE s′") -> "S PRIME"."""
    t = str(text).upper()
    for name, pattern in _RULES:
        if re.search(pattern, t):
            return name
    return default


def C(concept):
    """Color for a concept name, or for free text via concept_of."""
    return COLORS.get(concept, COLORS[concept_of(concept)])


# ---------------------------------------------------------------- style

def apply_style():
    order = ["STATE", "ACTION", "REWARD", "Q", "POLICY", "OBSERVATION", "EPISODE", "BELIEF", "TRANSITION"]
    plt.rcParams.update({
        "figure.dpi": 110,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "font.size": 11,
        "axes.prop_cycle": cycler(color=[COLORS[k] for k in order]),
        "image.cmap": "viridis",
    })


apply_style()


# ------------------------------------------------------------- markdown

MICRO_KEYS = list(COLORS)[:list(COLORS).index("EPISODE") + 1]


def legend(keys=None):
    """Show the color key as a row of chips (the micro-course concepts by default)."""
    chip = ('<span style="display:inline-block;background:{c};color:white;font-weight:800;'
            'padding:4px 8px;border-radius:5px;margin:3px">{k}</span>')
    keys = keys or MICRO_KEYS
    display(HTML("<div>" + "".join(chip.format(c=COLORS[k], k=k) for k in keys) + "</div>"))


# --------------------------------------------------------------- figures

def lineworld(position=0, n=6, title="LineWorld", arrow=None):
    """Draw the LineWorld track. arrow=(from_cell, to_cell, label) draws an action arrow."""
    goal = n - 1
    fig, ax = plt.subplots(figsize=(10, 2.6))
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(-0.8, 1.3)
    ax.axis("off")
    for i in range(n):
        ec = COLORS["REWARD"] if i == goal else "black"
        ax.add_patch(plt.Rectangle((i - 0.42, -0.25), 0.84, 0.5, fill=False, lw=2, ec=ec))
        ax.text(i, -0.55, f"s={i}", ha="center")
    ax.text(goal, 0.42, "GOAL +1", ha="center", weight="bold", color=COLORS["REWARD"])
    ax.scatter([position], [0], s=420, color=COLORS["STATE"], zorder=3)
    ax.text(position, 0, "A", ha="center", va="center", color="white", weight="bold", zorder=4)
    if arrow is not None:
        a, b, label = arrow
        ax.annotate("", xy=(b, 0.8), xytext=(a, 0.8),
                    arrowprops=dict(arrowstyle="->", lw=2.5, color=COLORS["ACTION"]))
        ax.text((a + b) / 2, 1.0, label, ha="center", weight="bold", color=COLORS["ACTION"])
    ax.set_title(title, fontsize=14)
    plt.tight_layout()
    return fig


def gridworld(position=0, goal=5, n=6, title="LineWorld"):
    """Older name for lineworld(), kept for 00_Gym_Basics."""
    return lineworld(position, n=max(n, goal + 1), title=title)


def q_heatmap(Q, title="Q-table", action_labels=("LEFT", "RIGHT")):
    """Q-table as a heatmap: rows are states, columns are actions."""
    Q = np.asarray(Q, dtype=float)
    fig, ax = plt.subplots(figsize=(6, max(3, Q.shape[0] * 0.55)))
    ax.grid(False)
    im = ax.imshow(Q, aspect="auto", cmap="viridis", vmin=min(0, Q.min()), vmax=max(1, Q.max()))
    labels = list(action_labels) + [str(j) for j in range(len(action_labels), Q.shape[1])]
    ax.set_xticks(range(Q.shape[1]), [f"a={j} {l}" for j, l in enumerate(labels[:Q.shape[1]])])
    ax.set_yticks(range(Q.shape[0]), [f"s={i}" for i in range(Q.shape[0])])
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            ax.text(j, i, f"{Q[i, j]:.2f}", ha="center", va="center", color="white", weight="bold")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Q(s, a)")
    plt.tight_layout()
    return fig


def qtable_heatmap(Q, title="Q-values"):
    """Older name for q_heatmap(), kept for 00_Gym_Basics."""
    return q_heatmap(Q, title)


def pipeline(labels, title="Flow", figsize=(11, 3.2)):
    """A left-to-right flow of labeled boxes, each colored by its concept."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    n = len(labels)
    xs = np.linspace(0.08, 0.92, n)
    for i, (x, label) in enumerate(zip(xs, labels)):
        ax.text(x, 0.50, label, ha="center", va="center", color="white", weight="bold", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.45", fc=C(label), ec=C(label), lw=1.5))
        if i < n - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.055, 0.50), xytext=(x + 0.055, 0.50),
                        arrowprops=dict(arrowstyle="->", lw=1.6))
    ax.set_title(title, fontsize=16, pad=16)
    plt.tight_layout()
    return fig


def card(title, lines, figsize=(9, 4.8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.05, 0.90, title, fontsize=18, weight="bold", va="top", color=C(title))
    y = 0.78
    for line in lines:
        ax.text(0.07, y, line, fontsize=12, va="top", family="monospace")
        y -= 0.10
    plt.tight_layout()
    return fig


def probability_bar(labels, probs, title="Probabilities"):
    probs = np.asarray(probs, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(labels, probs, color=COLORS["ACTION"])
    ax.set_ylim(0, max(1.0, probs.max() * 1.15 if len(probs) else 1.0))
    ax.set_ylabel("Probability")
    ax.set_title(title)
    plt.xticks(rotation=20)
    plt.tight_layout()
    return fig


def compare_bars(labels, values, title, ylabel):
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.bar(labels, values, color=[C(l) for l in labels])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(values) * 1.2 if values else 1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    return fig


def battle_schematic(state=None, title="Elder Mist battle state"):
    if state is None:
        state = {
            "boss_hp": 3995, "sword_hp": 800, "sword_active": True,
            "zale_hp": 300, "valere_hp": 300, "actor": "Zale",
            "special": "Torblado", "locks": "S:1 B:1 Sun:1 Moon:2"
        }
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    env_c, st_c = COLORS["ENVIRONMENT"], COLORS["STATE"]
    ax.text(5, 5.55, "ELDER MIST", ha="center", fontsize=17, weight="bold", color=env_c)
    ax.text(5, 5.05, f"HP {state.get('boss_hp', '?')}/3995", ha="center", fontsize=12)
    ax.text(7.7, 4.25, "SWORD", ha="center", fontsize=13, weight="bold", color=env_c)
    ax.text(7.7, 3.85, f"{'ACTIVE' if state.get('sword_active', True) else 'DOWN'} | HP {state.get('sword_hp','?')}", ha="center")
    ax.text(2.0, 2.1, "ZALE", ha="center", fontsize=15, weight="bold", color=st_c)
    ax.text(2.0, 1.7, f"HP {state.get('zale_hp','?')}", ha="center")
    ax.text(8.0, 2.1, "VALERE", ha="center", fontsize=15, weight="bold", color=st_c)
    ax.text(8.0, 1.7, f"HP {state.get('valere_hp','?')}", ha="center")
    ax.text(5, 0.8, f"Actor: {state.get('actor','?')} | Special: {state.get('special','-')} | Locks: {state.get('locks','-')}", ha="center", fontsize=11)
    ax.annotate("", xy=(5, 4.5), xytext=(2.4, 2.5), arrowprops=dict(arrowstyle="->", lw=1.5, color=COLORS["ACTION"]))
    ax.annotate("", xy=(7.2, 4.0), xytext=(7.8, 2.5), arrowprops=dict(arrowstyle="->", lw=1.5, color=COLORS["ACTION"]))
    ax.set_title(title, fontsize=16)
    plt.tight_layout()
    return fig
