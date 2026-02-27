"""Convert Python tutorial scripts to Jupyter notebooks.

Splits each .py file into:
  - Markdown cell: module docstring
  - Code cell: imports and setup
  - For each Step N: markdown header cell + code cell

Run: python examples/tutorials/_convert_to_ipynb.py
"""
import json
import os
import re
import glob
import textwrap


def make_nb(cells):
    """Create a minimal Jupyter notebook dict."""
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11.0",
            },
        },
        "cells": cells,
    }


def make_md_cell(source_lines):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines,
    }


def make_code_cell(source_lines):
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source_lines,
        "execution_count": None,
        "outputs": [],
    }


def parse_py_to_cells(py_path):
    """Parse a Python tutorial file into notebook cells."""
    with open(py_path, "r", encoding="utf-8") as f:
        content = f.read()

    cells = []

    # 1. Extract the docstring
    docstring_match = re.match(r'^"""(.*?)"""', content, re.DOTALL)
    if docstring_match:
        docstring = docstring_match.group(1).strip()
        # Convert to markdown
        lines = docstring.split("\n")
        # First line is the title
        title = lines[0].strip()
        rest = "\n".join(lines[1:]).strip()

        md_source = [f"# {title}\n"]
        if rest:
            md_source.append("\n")
            md_source.append(rest + "\n")
        cells.append(make_md_cell(md_source))

        # Remove docstring from content
        content = content[docstring_match.end():].strip()

    # 2. Split into sections by the "# ===..." step banners
    # Pattern: lines like print("=" * 60) ... print("  Step N: Title") ... print("=" * 60)
    # Or the code block between step banners

    # Strip comment-only step headers (# ==== / # Step N / # ====)
    content = re.sub(
        r'^# =+\n#\s+Step \d+.*\n# =+\n',
        '',
        content,
        flags=re.MULTILINE,
    )

    # Match step banners: either print("=" * 60) or print("\n" + "=" * 60)
    # followed by print("  Step N: Title") followed by print("=" * 60)
    banner_pattern = re.compile(
        r'((?:print\(["\']\\n["\'] \+ ["\']=["\'] \* 60\)'
        r'|print\(["\']=["\'] \* 60\))\n'
        r'print\(["\']  Step (\d+):? (.*?)["\']\)\n'
        r'print\(["\']=["\'] \* 60\))',
        re.MULTILINE,
    )

    # Find all step boundaries
    all_steps = list(banner_pattern.finditer(content))

    if not all_steps:
        # No step structure found — put everything in one code cell
        # But first separate imports
        lines = content.split("\n")
        import_end = 0
        for i, line in enumerate(lines):
            if (line.startswith("import ") or line.startswith("from ") or
                line.startswith("sys.path") or line.startswith("np.set_print") or
                line.strip() == "" or line.startswith("#")):
                import_end = i + 1
            else:
                break

        if import_end > 0:
            setup_code = "\n".join(lines[:import_end]).strip()
            # Fix sys.path for notebook (go up differently)
            setup_code = fix_paths_for_notebook(setup_code)
            cells.append(make_code_cell([setup_code + "\n"]))

            remaining = "\n".join(lines[import_end:]).strip()
            if remaining:
                cells.append(make_code_cell([remaining + "\n"]))
        else:
            cells.append(make_code_cell([content + "\n"]))

        return cells

    # Extract code before first step (imports/setup)
    pre_step = content[:all_steps[0].start()].strip()
    if pre_step:
        pre_step = fix_paths_for_notebook(pre_step)
        cells.append(make_code_cell([pre_step + "\n"]))

    # Process each step
    for i, match in enumerate(all_steps):
        step_num = match.group(2)
        step_title = match.group(3).strip().rstrip("'\"")

        # Create markdown header for this step
        cells.append(make_md_cell([f"## Step {step_num}: {step_title}\n"]))

        # Get code for this step (from after the banner to next step or end)
        code_start = match.end()
        if i + 1 < len(all_steps):
            code_end = all_steps[i + 1].start()
        else:
            code_end = len(content)

        step_code = content[code_start:code_end].strip()

        # Remove the print banner at the start (already captured in markdown)
        # The banner is the three print() lines we matched

        if step_code:
            step_code = fix_paths_for_notebook(step_code)
            cells.append(make_code_cell([step_code + "\n"]))

    return cells


def fix_paths_for_notebook(code):
    """Fix __file__-based paths for notebook context.

    Python scripts live in tutorials/python_scripts/ and use __file__ to resolve
    relative paths. Notebooks live in tutorials/ and have no __file__, so we
    replace with pathlib.Path.cwd()-based equivalents.

    From tutorials/:
      ../../src           -> project root / src
      ../../data          -> wrong (scripts use ../data from tutorials/)
      ../data             -> actually correct *from tutorials/* but scripts
                             now at python_scripts/ use ../../data
    """
    # sys.path: scripts use "..", "..", "..", "src" from python_scripts/
    # notebooks at tutorials/ need  "..", "..", "src"
    code = code.replace(
        'sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))',
        'import pathlib\nsys.path.insert(0, str(pathlib.Path.cwd().parent.parent / "src"))',
    )
    # data_path: scripts use "..", "..", "data" from python_scripts/
    # notebooks at tutorials/ need "..", "data"
    code = code.replace(
        'os.path.join(os.path.dirname(__file__), "..", "..", "data", "TRAVELMODE.csv")',
        'str(pathlib.Path.cwd().parent / "data" / "TRAVELMODE.csv")',
    )
    return code


def convert_file(py_path, out_dir):
    """Convert a single .py file to .ipynb."""
    basename = os.path.splitext(os.path.basename(py_path))[0]
    nb_path = os.path.join(out_dir, basename + ".ipynb")

    cells = parse_py_to_cells(py_path)
    nb = make_nb(cells)

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    return nb_path


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    py_dir = os.path.join(script_dir, "python_scripts")
    out_dir = script_dir  # notebooks go in tutorials/

    py_files = sorted(glob.glob(os.path.join(py_dir, "t*.py")))

    print(f"Converting {len(py_files)} Python tutorials to Jupyter notebooks...")
    for py_path in py_files:
        nb_path = convert_file(py_path, out_dir)
        print(f"  {os.path.basename(py_path)} -> {os.path.basename(nb_path)}")

    print(f"\nDone! {len(py_files)} notebooks created in {out_dir}")


if __name__ == "__main__":
    main()
