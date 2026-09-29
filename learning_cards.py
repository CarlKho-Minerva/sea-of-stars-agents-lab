"""Shared colors and figures for the 00_0x micro-course notebooks.

Each RL concept has one fixed color, used in both the markdown cards and the plots.
"""
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import HTML, display

COLORS = {
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
}


def legend(keys=None):
    """Show the color key as a row of chips."""
    chip = ('<span style="display:inline-block;background:{c};color:white;font-weight:800;'
            'padding:4px 8px;border-radius:5px;margin:3px">{k}</span>')
    keys = keys or list(COLORS)
    display(HTML("<div>" + "".join(chip.format(c=COLORS[k], k=k) for k in keys) + "</div>"))


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


def q_heatmap(Q, title="Q-table", action_labels=("LEFT", "RIGHT")):
    """Q-table as a heatmap: rows are states, columns are actions."""
    Q = np.asarray(Q, dtype=float)
    fig, ax = plt.subplots(figsize=(6, max(3, Q.shape[0] * 0.55)))
    im = ax.imshow(Q, aspect="auto", cmap="viridis", vmin=min(0, Q.min()), vmax=max(1, Q.max()))
    ax.set_xticks(range(Q.shape[1]), [f"a={j} {l}" for j, l in enumerate(action_labels)])
    ax.set_yticks(range(Q.shape[0]), [f"s={i}" for i in range(Q.shape[0])])
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            ax.text(j, i, f"{Q[i, j]:.2f}", ha="center", va="center", color="white", weight="bold")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Q(s, a)")
    plt.tight_layout()
    return fig
