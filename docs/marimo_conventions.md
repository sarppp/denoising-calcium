# marimo notebook conventions

Tags: `marimo`, `notebook_conventions`, `python`, `tooling`

When writing or editing marimo (`.py`) notebooks, follow these rules. marimo
is **not** Jupyter — cells share a global namespace and use a reactive
dataflow model.

## 1. Single-owner rule

Every top-level name in a cell becomes a global owned by that cell. Two
cells cannot both define the same bare name (e.g. `for p in ...` in two
cells → `MultipleDefinitionError`).

## 2. Underscore prefix for cell-locals

Prefix every variable that is private to a cell with `_` (loop indices
`_p`, `_i`, temporary arrays `_arr`, intermediate axes `_ax`, throwaway
figures `_fig`, etc.). marimo treats `_name` as cell-local and does not
check it for collisions.

## 3. Only expose shared names bare

Names used by other cells (`fits`, `DATA`, `TRAIN`, `noise_fig`) are bare
and listed in the cell's `return (x, y, z)`.

## 4. No bare `return`

A cell either returns a tuple of exports or omits `return` entirely.
`return` with no value or `return (single,)` for a non-exported value
triggers MB005 syntax errors after marimo's code generation.

## 5. No `plt.show()`

marimo renders the last expression or any figure returned from the cell.
Calling `plt.show()` is unnecessary and can double-render. For multiple
figures per cell, collect them into a list and return `mo.vstack(list)`
as the last expression.

## 6. Cell definition forms

Use `@app.cell` for decorated-function cells (standard). Use
`@app.function` for pure stateless helpers that should be callable from
multiple cells.

## 7. Promote heavy logic to `src/`

When a cell exceeds ~15 lines of real logic, move it to
`workspace/src/cidc/*.py` (or equivalent) and call the helper from a
short cell. This keeps cells readable and makes logic testable.

## 8. Lint after editing

Run `marimo check <file>` or call the `mcp0_lint_notebook` MCP tool
before declaring the notebook fixed. The marimo MCP server exposes
runtime state, errors, variable values, and dependency graphs — use
these tools when connected instead of asking the user to paste output.

## 9. Launch with MCP for Cascade visibility

```bash
uv run --with="marimo[mcp]" marimo edit notebook.py --mcp --no-token
```

exposes `http://localhost:PORT/mcp/server`, which Windsurf/Cascade
connects to via `~/.codeium/windsurf/mcp_config.json` using `"serverUrl"`
(not `"url"`):

```json
{
  "mcpServers": {
    "marimo": {
      "serverUrl": "http://localhost:2718/mcp/server"
    }
  }
}
```

## 10. Sharing

The `.py` file is git-friendly and runnable by anyone with Python. For
non-marimo audiences, export with

```bash
marimo export html     notebook.py -o out.html     # static, double-clickable
marimo export html-wasm notebook.py -o site/        # interactive, zero-install
marimo export ipynb    notebook.py -o out.ipynb    # for Jupyter users
marimo export md       notebook.py -o out.md       # for writeups
```
