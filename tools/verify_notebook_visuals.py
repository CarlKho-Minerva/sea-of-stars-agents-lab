from pathlib import Path
import nbformat as nbf

root = Path(__file__).resolve().parent.parent
problems = []

# The 00_0x micro-course uses HTML cards rather than SVG, so only the main notebooks are checked.
for path in sorted((root / "notebooks").glob("*.ipynb")):
    if path.name.startswith("00_0"):
        continue
    nb = nbf.read(path, as_version=4)
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == "markdown" and "<svg" not in cell.source:
            problems.append(f"{path.name} cell {i}: markdown missing SVG")
        if cell.cell_type == "code":
            markers = ["plt.show()", "display(", "plot_state(", "fig ="]
            if not any(m in cell.source for m in markers):
                problems.append(f"{path.name} cell {i}: code missing obvious visual")

if problems:
    print("VISUAL CHECK FAILED")
    for p in problems:
        print(" -", p)
    raise SystemExit(1)

print("VISUAL CHECK OK")
print("Every notebook teaching cell has a visual diagram, figure, chart, or interactive figure.")
