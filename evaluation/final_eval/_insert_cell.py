#!/usr/bin/env python3
"""Idempotently append the FINAL-EVAL cell to results_viz.ipynb (preserves outputs)."""
import json, shutil, uuid
from pathlib import Path

EVAL = Path("/leonardo_work/IscrC_AHNetBio/C-VAE/evaluation")
NB = EVAL / "results_viz.ipynb"
SRC = EVAL / "final_eval" / "_final_cell_src.py"
MARKER = "FINAL EVAL — paper tables (SELF-CONTAINED)"

src = SRC.read_text()
# nbformat wants source as a list of lines each ending in \n (except maybe last).
lines = src.splitlines(keepends=True)

nb = json.loads(NB.read_text())
cells = nb["cells"]

# find existing final-eval cell by marker
idx = None
for i, c in enumerate(cells):
    if c.get("cell_type") == "code" and MARKER in "".join(c.get("source", [])):
        idx = i
        break

new_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": lines,
    "id": uuid.uuid4().hex[:8],
}

if idx is not None:
    new_cell["id"] = cells[idx].get("id", new_cell["id"])
    cells[idx] = new_cell
    action = f"replaced existing cell at index {idx}"
else:
    shutil.copy2(NB, NB.with_suffix(".ipynb.bak"))
    cells.append(new_cell)
    action = f"appended new cell (backup → {NB.name}.bak)"

NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"OK: {action}; total cells now {len(cells)}; cell id {new_cell['id']}")
