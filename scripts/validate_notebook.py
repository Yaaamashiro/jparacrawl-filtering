"""Validate notebook structure and Python cell syntax without executing Colab operations."""

import ast
from pathlib import Path

import nbformat

path = Path(__file__).resolve().parents[1] / "notebooks/jparacrawl_filtering_colab.ipynb"
nb = nbformat.read(path, as_version=4)
nbformat.validate(nb)
for cell in nb.cells:
    if cell.cell_type == "code":
        ast.parse(cell.source)
        assert cell.execution_count is None and not cell.outputs
print("Notebook schema and Python cell syntax valid; no saved output.")
