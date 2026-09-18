"""Converts the `# %%` percent-format sources in this folder into Kaggle-ready .ipynb files."""
import json
import re
from pathlib import Path

HERE = Path(__file__).parent


def to_notebook(source):
    cells = []
    for chunk in re.split(r"^# %%", source, flags=re.M)[1:]:
        header, _, body = chunk.partition("\n")
        body = body.strip("\n")
        if header.strip() == "[markdown]":
            text = "\n".join(line[2:] if line.startswith("# ") else line.lstrip("#") for line in body.splitlines())
            cells.append({"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)})
        else:
            cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                          "source": body.splitlines(keepends=True)})
    return {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
                         "language_info": {"name": "python"}}}


for path in sorted(HERE.glob("0*.py")):
    out = HERE.parent / "notebooks" / path.with_suffix(".ipynb").name
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(to_notebook(path.read_text()), indent=1) + "\n")
    print("wrote", out)
