"""Convert Python tutorial scripts to Jupyter notebooks.

Splits each .py file into:
  - Markdown cell: module docstring
  - Code cell: imports and setup
  - For each Step N: markdown header cell + (code | prose-markdown)* cells

Top-level triple-quoted ``print(...)`` blocks (without an ``f`` prefix)
are lifted into their own markdown cells so prose renders as Markdown
rather than appearing inside a code cell as a literal print statement.
This matches the hand-edited notebook structure introduced in commit
bcbec92 and keeps a single source of truth (the .py script) for
future regeneration.

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
    """Legacy code-cell shape: cell_type → metadata → source → exec → outputs.

    Used for the pre-step setup cell (matches the original converter
    output that bcbec92 left untouched).
    """
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source_lines,
        "execution_count": None,
        "outputs": [],
    }


def _make_in_step_code_cell(text):
    """Canonical in-step code-cell shape (matches bcbec92 hand edits).

    Differs from ``make_code_cell`` in two ways:

    1. Source is split per-line via ``splitlines(keepends=True)``.
    2. Keys are ordered cell_type → exec_count → metadata → outputs → source.

    Both differences match the shape commit bcbec92 normalized inside
    each step, so re-running the converter doesn't churn that commit.
    """
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


def _per_line_md_source(text):
    """Build the per-line source list bcbec92 uses for lifted prose cells.

    Each interior line keeps its trailing ``\\n``; the final line has no
    trailing newline, matching ``str.splitlines(keepends=True)``.
    """
    return text.splitlines(keepends=True)


# Top-level triple-quoted print() prose blocks (no ``f`` prefix) — pure
# prose to lift into markdown cells.  f-string variants (print(f"...")
# with triple quotes) are intentionally NOT matched, since they carry
# runtime interpolation and must remain code.
#
# The body capture allows content on the opening line (some scripts
# write ``print("""text...``) and on the closing line — leading and
# trailing newlines are stripped after capture by ``textwrap.dedent``
# + ``.strip()``.
_PROSE_PRINT_RE = re.compile(
    r'^print\("""(.*?)"""\)\s*$',
    re.MULTILINE | re.DOTALL,
)

# Trailing ``print(f"  Next: ...")`` navigation hints (the f-prefix
# variant only).  These lift to a bold ``**Next:** ...`` markdown cell.
_NAV_PRINT_RE = re.compile(
    r'^print\(f"\s*Next:\s*(.*?)"\)\s*$',
    re.MULTILINE,
)

# Bare ``print("...")`` (no ``f`` prefix, single string argument).
# bcbec92 absorbed any trailing run of these lines into the preceding
# prose markdown cell — each line's argument becomes a final line of
# the prose (with leading visual-indent whitespace stripped).
_BARE_PRINT_LINE_RE = re.compile(
    r'^\s*print\("(.*?)"\)\s*$',
)

# Combined lift pattern: alternation of both prose- and nav-style
# matches.  A single ``finditer`` keeps the matches in document order
# so surrounding code is partitioned correctly even when both kinds
# appear in one step.
_LIFT_RE = re.compile(
    r'^(?:'
    r'print\("""(?P<prose>.*?)"""\)'
    r'|'
    r'print\(f"\s*Next:\s*(?P<nav>.*?)"\)'
    r')\s*$',
    re.MULTILINE | re.DOTALL,
)


def split_code_by_prose_prints(code, lift_nav=True):
    """Split a code chunk into a list of (cell_type, content) tuples.

    Two kinds of statements lift to markdown:

    1. Top-level triple-quoted ``print(...)`` prose blocks — leading
       indent is stripped via ``textwrap.dedent``.
    2. Trailing ``print(f"  Next: ...")`` navigation hints — rendered
       as ``**Next:** <text>``.  Only lifted when ``lift_nav=True``;
       bcbec92 only converted these in files it was already editing,
       so the caller passes ``lift_nav=bcbec92_mode``.

    Surrounding code becomes ``code`` entries.  Returns
    ``[("code", code)]`` unchanged if no lift candidates are present.
    """
    pattern = _LIFT_RE if lift_nav else _PROSE_PRINT_RE

    pieces = []
    last_end = 0
    for m in pattern.finditer(code):
        before = code[last_end:m.start()].strip()
        if before:
            pieces.append(("code", before))

        if pattern is _PROSE_PRINT_RE:
            md_text = textwrap.dedent(m.group(1)).strip()
        elif m.group("prose") is not None:
            md_text = textwrap.dedent(m.group("prose")).strip()
        else:
            md_text = f"**Next:** {m.group('nav').strip()}"

        if md_text:
            pieces.append(("markdown", md_text))
        last_end = m.end()

    if not pieces:
        return [("code", code)]

    rest = code[last_end:].strip()
    if rest:
        pieces.append(("code", rest))

    if lift_nav:
        pieces = _absorb_bare_next_into_prose(pieces)

    return pieces


def _absorb_bare_next_into_prose(pieces):
    """Fold trailing bare ``print("...")`` lines into the prior prose.

    bcbec92 merged any run of one or more bare ``print("...")`` calls
    (no ``f`` prefix, single string argument) at the end of a step
    into the preceding prose markdown cell, separated from the body
    by a blank line.  Each absorbed line's argument has its leading
    visual-indent whitespace stripped before joining.

    No-op when the trailing code piece doesn't consist purely of such
    bare prints, or when there is no preceding markdown piece.
    """
    if len(pieces) < 2:
        return pieces
    last_kind, last_content = pieces[-1]
    prev_kind, prev_content = pieces[-2]
    if last_kind != "code" or prev_kind != "markdown":
        return pieces

    absorbed_lines = []
    for line in last_content.strip().splitlines():
        if not line.strip():
            continue
        m = _BARE_PRINT_LINE_RE.match(line)
        if not m:
            return pieces  # mixed content — leave the code cell alone
        absorbed_lines.append(m.group(1).lstrip())

    if not absorbed_lines:
        return pieces

    merged = prev_content + "\n\n" + "\n".join(absorbed_lines)
    return pieces[:-2] + [("markdown", merged)]


def parse_py_to_cells(py_path):
    """Parse a Python tutorial file into notebook cells."""
    with open(py_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Files containing prose ``print(...)`` blocks were normalized to
    # bcbec92's canonical in-step code-cell shape when those blocks were
    # lifted to markdown.  Files without prose prints stayed in the
    # original legacy shape.  Match the per-file convention so a clean
    # regeneration produces a byte-equivalent notebook for both groups.
    bcbec92_mode = bool(_PROSE_PRINT_RE.search(content))

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
                _emit_split_cells(cells, remaining, bcbec92_mode)
        else:
            _emit_split_cells(cells, content, bcbec92_mode)

        return cells

    # Extract code before first step (imports/setup).  Pre-step setup
    # code stays in the legacy single-element shape regardless of mode
    # — bcbec92 left the imports cell untouched.
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
            _emit_split_cells(cells, step_code, bcbec92_mode)

    return cells


def _emit_split_cells(cells, code, bcbec92_mode):
    """Append cells from a step's code chunk.

    Each piece becomes a separate cell.  Prose lifts to a markdown cell
    with per-line source.  Code pieces produced by an actual split
    (i.e., a step that contained a prose print) use the canonical
    in-step shape; code in steps that had no prose print stays in the
    original legacy single-element shape.  This matches bcbec92, which
    only normalized cells it actually edited.

    The ``bcbec92_mode`` flag is honored for consistency but not
    strictly required — the per-step heuristic is sufficient.
    """
    pieces = split_code_by_prose_prints(code, lift_nav=bcbec92_mode)
    has_markdown_split = any(kind == "markdown" for kind, _ in pieces)

    for kind, content in pieces:
        if kind == "markdown":
            cells.append(make_md_cell(_per_line_md_source(content)))
        elif bcbec92_mode and has_markdown_split:
            cells.append(_make_in_step_code_cell(content + "\n"))
        else:
            cells.append(make_code_cell([content + "\n"]))


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

    # bcbec92 added a trailing newline to every notebook it edited, so
    # the bcbec92-touched files (those whose .py source has a prose
    # print block) end with "\n" and the others don't.  Match per-file
    # so a clean regen produces a byte-identical notebook either way.
    with open(py_path, "r", encoding="utf-8") as f:
        has_prose_prints = bool(_PROSE_PRINT_RE.search(f.read()))

    with open(nb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        if has_prose_prints:
            f.write("\n")

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
