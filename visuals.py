import matplotlib.pyplot as plt
import numpy as np

def card(title, lines, figsize=(9, 4.8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.text(0.05, 0.90, title, fontsize=18, weight="bold", va="top")
    y = 0.78
    for line in lines:
        ax.text(0.07, y, line, fontsize=12, va="top", family="monospace")
        y -= 0.10
    plt.tight_layout()
    return fig

def pipeline(labels, title="Flow", figsize=(11, 3.2)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    n = len(labels)
    xs = np.linspace(0.08, 0.92, n)
    for i, (x, label) in enumerate(zip(xs, labels)):
        ax.text(
            x, 0.50, label, ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="black", lw=1.5),
            fontsize=11,
        )
        if i < n - 1:
            ax.annotate(
                "", xy=(xs[i+1]-0.055, 0.50), xytext=(x+0.055, 0.50),
                arrowprops=dict(arrowstyle="->", lw=1.6)
            )
    ax.set_title(title, fontsize=16, pad=16)
    plt.tight_layout()
    return fig

def gridworld(position=0, goal=5, n=6, title="LineWorld"):
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.set_xlim(-0.5, n-0.5)
    ax.set_ylim(-0.7, 0.9)
    for i in range(n):
        ax.add_patch(plt.Rectangle((i-0.42, -0.25), 0.84, 0.50, fill=False, lw=1.5))
        ax.text(i, -0.52, str(i), ha="center")
    ax.scatter([position], [0], s=260, marker="o")
    ax.text(position, 0.42, "AGENT", ha="center", weight="bold")
    ax.scatter([goal], [0], s=260, marker="*")
    ax.text(goal, 0.42, "GOAL", ha="center", weight="bold")
    ax.axis("off")
    ax.set_title(title, fontsize=15)
    plt.tight_layout()
    return fig

def qtable_heatmap(Q, title="Q-values"):
    Q = np.asarray(Q)
    fig, ax = plt.subplots(figsize=(8, max(3, Q.shape[0]*0.55)))
    im = ax.imshow(Q, aspect="auto")
    ax.set_xlabel("Action")
    ax.set_ylabel("State")
    ax.set_title(title)
    for i in range(Q.shape[0]):
        for j in range(Q.shape[1]):
            ax.text(j, i, f"{Q[i,j]:.2f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Q")
    plt.tight_layout()
    return fig

def probability_bar(labels, probs, title="Probabilities"):
    probs = np.asarray(probs, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(labels, probs)
    ax.set_ylim(0, max(1.0, probs.max()*1.15 if len(probs) else 1.0))
    ax.set_ylabel("Probability")
    ax.set_title(title)
    plt.xticks(rotation=20)
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
    ax.text(5, 5.55, "ELDER MIST", ha="center", fontsize=17, weight="bold")
    ax.text(5, 5.05, f"HP {state.get('boss_hp', '?')}/3995", ha="center", fontsize=12)
    ax.text(7.7, 4.25, "SWORD", ha="center", fontsize=13, weight="bold")
    ax.text(7.7, 3.85, f"{'ACTIVE' if state.get('sword_active', True) else 'DOWN'} | HP {state.get('sword_hp','?')}", ha="center")
    ax.text(2.0, 2.1, "ZALE", ha="center", fontsize=15, weight="bold")
    ax.text(2.0, 1.7, f"HP {state.get('zale_hp','?')}", ha="center")
    ax.text(8.0, 2.1, "VALERE", ha="center", fontsize=15, weight="bold")
    ax.text(8.0, 1.7, f"HP {state.get('valere_hp','?')}", ha="center")
    ax.text(5, 0.8, f"Actor: {state.get('actor','?')} | Special: {state.get('special','-')} | Locks: {state.get('locks','-')}", ha="center", fontsize=11)
    ax.annotate("", xy=(5,4.5), xytext=(2.4,2.5), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.annotate("", xy=(7.2,4.0), xytext=(7.8,2.5), arrowprops=dict(arrowstyle="->", lw=1.5))
    ax.set_title(title, fontsize=16)
    plt.tight_layout()
    return fig

def compare_bars(labels, values, title, ylabel):
    fig, ax = plt.subplots(figsize=(8.5, 4))
    ax.bar(labels, values)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(values)*1.2 if values else 1)
    plt.xticks(rotation=15)
    plt.tight_layout()
    return fig
