# Copyright 2026 Marimo. All rights reserved.
# uv run --with="marimo[mcp]" marimo edit workspace/notebooks/01_eda.py --mcp --no-token --host 0.0.0.0 --port 2718
#
# ─── marimo notebook detection gotcha ────────────────────────────────────────
# The marimo GUI file browser only tags a `.py` file with the "m" icon (and
# lists it as a notebook) if the FIRST 512 BYTES contain BOTH the literals
# `import marimo` AND `marimo.App`. See
# marimo/_server/files/directory_scanner.py :: is_marimo_app (READ_LIMIT=512).
# Markdown notebooks are detected by `marimo-version:` in the first 512 bytes.
#
# Practical consequence: if a notebook has a long module-level docstring at
# the top, `app = marimo.App(...)` can get pushed past byte 512 and marimo
# will silently fail to recognise the file. Symptom: no `m` icon in the file
# tree, and `mcp0_get_active_notebooks` only sees the notebook explicitly
# passed to `marimo edit`, never its siblings.
#
# Fix: keep the top-of-file docstring short (a single line is safest) and
# move any longer preamble into the first `@app.cell` as `mo.md(...)`.
#
# Verification (the exact command I used):
#
#     grep -bn 'marimo.App\|import marimo' workspace/notebooks/06_proofs.py
#
# Output format is `<byte_offset>:<line>:<match>`. Both `import marimo` and
# `marimo.App` must have byte_offset < 512. Example good / bad:
#
#     06_proofs.py:107:app = marimo.App(width="medium")     # ✓ under 512
#     06_proofs.py:524:app = marimo.App(width="medium")     # ✗ not detected
#
# One-liner assertion for a whole folder:
#
#     for f in workspace/notebooks/*.py; do
#         python -c "import sys; d=open(sys.argv[1],'rb').read(512); \
#             print(sys.argv[1], 'OK' if b'marimo.App' in d and b'import marimo' in d else 'FAIL')" "$f"
#     done
#
# Also note: each `marimo edit` process exposes exactly one notebook over MCP,
# on its own port. To make multiple notebooks visible to Cascade, launch one
# process per notebook (different --port) and add one entry per port to
# ~/.codeium/windsurf/mcp_config.json using `"serverUrl"` (NOT `"url"`), then
# reload MCP servers in Windsurf.

#{
#  "mcpServers": {
#    "marimo": {
#      "serverUrl": "http://localhost:2718/mcp/server"
#    }
#  }
#}


# ─── LaTeX inside mo.md — what actually renders correctly ───────────────────
# marimo renders math with KaTeX. Most bugs come from two things:
#
# 1. String escaping. ALWAYS use a raw string `r""" ... """` for `mo.md`
#    content that contains `\`. Without the `r`, `\n`, `\t`, `\f`, `\v`,
#    `\a`, `\b` get eaten by Python before KaTeX ever sees them — e.g.
#    `\frac` survives but `\nu`, `\tau`, `\text` (no, that one's fine) can
#    silently break. Raw strings cost nothing; use them unconditionally.
#
# 2. Subscripts after bracketed indices. KaTeX (and LaTeX) parses `[...]`
#    right after a command as an OPTIONAL ARGUMENT, so
#
#        \operatorname{ACF}[1]_{\text{total}}       ← WRONG
#
#    binds `_{\text{total}}` to `]`, not to `ACF[1]`. Either:
#    - move the index into parentheses: `\mathrm{ACF}_{\text{total}}(1)`
#    - or brace the whole base: `{\mathrm{ACF}[1]}_{\text{total}}`
#
# 3. Don't mix unicode math glyphs with `$...$`. Inline `SNR_dB = 10·log₁₀(…)`
#    using unicode subscripts and `·` renders as plain text and looks broken
#    next to real math. Put the whole equation in `$$...$$` with `\log_{10}`,
#    `\cdot`, `\frac`, etc. Reserve unicode for prose, LaTeX for equations.
#
# 4. Indentation inside `mo.md(r""" ... """)`. CommonMark treats ≥4 leading
#    spaces as a code block. marimo dedents by the minimum indent of the
#    block, so keep every line at the same indentation level — don't indent
#    the `$$` further than the surrounding prose or it becomes `<pre>`.
#
# 5. Prefer `\mathrm{...}` over `\operatorname{...}` for short multi-letter
#    names that aren't really functions (ACF, Var, SNR). `\operatorname`
#    adds thin spacing that looks off for labels.
#
# 6. Good-to-know macros that KaTeX supports:
#    `\mathrm`, `\text`, `\frac`, `\log`, `\sqrt`, `\sum`, `\int`, `\cdot`,
#    `\approx`, `\leq`, `\geq`, `\left(` / `\right)`, `\!` (neg thin space),
#    `\,` (thin space), `\;` (med space). Things KaTeX does NOT support
#    by default: `\label`, `\ref`, `\newcommand` at render time, most
#    `amsmath` environments beyond `aligned` / `cases` / `matrix`.
#
# 7. Verify visually. If an equation renders as literal text with `\frac`
#    visible, it's almost always (a) missing the `r` prefix, (b) an
#    `[index]` optional-arg bug, or (c) extra indentation turning the
#    block into `<pre><code>`.
#
# Example (good):
#
#     mo.md(r"""
#     ## SNR from lag-1 autocorrelation
#
#     $$
#     \mathrm{SNR}_{\mathrm{dB}} = 10\,\log_{10}\!\left(
#         \frac{\mathrm{ACF}(1)}{1 - \mathrm{ACF}(1)}
#     \right).
#     $$
#     """)


# uv run --with="marimo[mcp]" marimo edit workspace/notebooks/06_proofs.py --mcp --no-token --host 0.0.0.0 --port 2719 --watch

# ─────────────────────────────────────────────────────────────────────────────

import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    mo.md("# Welcome to marimo! 🌊🍃")
    return (mo,)


@app.cell
def _(mo):
    slider = mo.ui.slider(1, 22)
    return (slider,)


@app.cell
def _(mo, slider):
    mo.md(f"""
    marimo is a **reactive** Python notebook.

    This means that unlike traditional notebooks, marimo notebooks **run
    automatically** when you modify them or
    interact with UI elements, like this slider: {slider}.

    {"##" + "🍃" * slider.value}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Tip: disabling automatic execution": mo.md(
                rf"""
            marimo lets you disable automatic execution: in the notebook
            footer, change "On Cell Change" to "lazy".

            When the runtime is lazy, after running a cell, marimo marks its
            descendants as stale instead of automatically running them. The
            lazy runtime puts you in control over when cells are run, while
            still giving guarantees about the notebook state.
            """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        Tip: This is a tutorial notebook. You can create your own notebooks
        by entering `marimo edit` at the command line.
        """
    ).callout()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Reactive execution

    A marimo notebook is made up of small blocks of Python code called
    cells.

    marimo reads your cells and models the dependencies among them: whenever
    a cell that defines a global variable  is run, marimo
    **automatically runs** all cells that reference that variable.

    Reactivity keeps your program state and outputs in sync with your code,
    making for a dynamic programming environment that prevents bugs before they
    happen.
    """)
    return


@app.cell(hide_code=True)
def _(changed, mo):
    (
        mo.md(
            f"""
            **✨ Nice!** The value of `changed` is now {changed}.

            When you updated the value of the variable `changed`, marimo
            **reacted** by running this cell automatically, because this cell
            references the global variable `changed`.

            Reactivity ensures that your notebook state is always
            consistent, which is crucial for doing good science; it's also what
            enables marimo notebooks to double as tools and  apps.
            """
        )
        if changed
        else mo.md(
            """
            **🌊 See it in action.** In the next cell, change the value of the
            variable  `changed` to `True`, then click the run button.
            """
        )
    )
    return


@app.cell
def _():
    changed = False
    return (changed,)


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Tip: execution order": (
                """
                The order of cells on the page has no bearing on
                the order in which cells are executed: marimo knows that a cell
                reading a variable must run after the cell that  defines it. This
                frees you to organize your code in the way that makes the most
                sense for you.
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    **Global names must be unique.** To enable reactivity, marimo imposes a
    constraint on how names appear in cells: no two cells may define the same
    variable.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Tip: encapsulation": (
                """
                By encapsulating logic in functions, classes, or Python modules,
                you can minimize the number of global variables in your notebook.
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.accordion(
        {
            "Tip: private variables": (
                """
                Variables prefixed with an underscore are "private" to a cell, so
                they can be defined by multiple cells.
                """
            )
        }
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. UI elements

    Cells can output interactive UI elements. Interacting with a UI
    element **automatically triggers notebook execution**: when
    you interact with a UI element, its value is sent back to Python, and
    every cell that references that element is re-run.

    marimo provides a library of UI elements to choose from under
    `marimo.ui`.
    """)
    return


@app.cell
def _(mo):
    mo.md("""
    **🌊 Some UI elements.** Try interacting with the below elements.
    """)
    return


@app.cell
def _(mo):
    icon = mo.ui.dropdown(["🍃", "🌊", "✨"], value="🍃")
    return (icon,)


@app.cell
def _(icon, mo):
    repetitions = mo.ui.slider(1, 16, label=f"number of {icon.value}: ")
    return (repetitions,)


@app.cell
def _(icon, repetitions):
    icon, repetitions
    return


@app.cell
def _(icon, mo, repetitions):
    mo.md("# " + icon.value * repetitions.value)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. marimo is just Python

    marimo cells parse Python (and only Python), and marimo notebooks are
    stored as pure Python files — outputs are _not_ included. There's no
    magical syntax.

    The Python files generated by marimo are:

    - easily versioned with git, yielding minimal diffs
    - legible for both humans and machines
    - formattable using your tool of choice,
    - usable as Python  scripts, with UI  elements taking their default
    values, and
    - importable by other modules (more on that in the future).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Running notebooks as apps

    marimo notebooks can double as apps. Click the app window icon in the
    bottom-right to see this notebook in "app view."

    Serve a notebook as an app with `marimo run` at the command-line.
    Of course, you can use marimo just to level-up your
    notebooking, without ever making apps.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. The `marimo` command-line tool

    **Creating and editing notebooks.** Use

    ```
    marimo edit
    ```

    in a terminal to start the marimo notebook server. From here
    you can create a new notebook or edit existing ones.


    **Running as apps.** Use

    ```
    marimo run notebook.py
    ```

    to start a webserver that serves your notebook as an app in read-only mode,
    with code cells hidden.

    **Convert a Jupyter notebook.** Convert a Jupyter notebook to a marimo
    notebook using `marimo convert`:

    ```
    marimo convert your_notebook.ipynb > your_app.py
    ```

    **Tutorials.** marimo comes packaged with tutorials:

    - `dataflow`: more on marimo's automatic execution
    - `ui`: how to use UI elements
    - `markdown`: how to write markdown, with interpolated values and
       LaTeX
    - `plots`: how plotting works in marimo
    - `sql`: how to use SQL
    - `layout`: layout elements in marimo
    - `fileformat`: how marimo's file format works
    - `markdown-format`: for using `.md` files in marimo
    - `for-jupyter-users`: if you are coming from Jupyter

    Start a tutorial with `marimo tutorial`; for example,

    ```
    marimo tutorial dataflow
    ```

    In addition to tutorials, we have examples in our
    [our GitHub repo](https://www.github.com/marimo-team/marimo/tree/main/examples).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6. The marimo editor

    Here are some tips to help you get started with the marimo editor.
    """)
    return


@app.cell
def _(mo, tips):
    mo.accordion(tips)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Finally, a fun fact
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    The name "marimo" is a reference to a type of algae that, under
    the right conditions, clumps together to form a small sphere
    called a "marimo moss ball". Made of just strands of algae, these
    beloved assemblages are greater than the sum of their parts.
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _():
    tips = {
        "Saving": (
            """
            **Saving**

            - _Name_ your app using the box at the top of the screen, or
              with `Ctrl/Cmd+s`. You can also create a named app at the
              command line, e.g., `marimo edit app_name.py`.

            - _Save_ by clicking the save icon on the bottom right, or by
              inputting `Ctrl/Cmd+s`. By default marimo is configured
              to autosave.
            """
        ),
        "Running": (
            """
            1. _Run a cell_ by clicking the play ( ▷ ) button on the top
            right of a cell, or by inputting `Ctrl/Cmd+Enter`.

            2. _Run a stale cell_  by clicking the yellow run button on the
            right of the cell, or by inputting `Ctrl/Cmd+Enter`. A cell is
            stale when its code has been modified but not run.

            3. _Run all stale cells_ by clicking the play ( ▷ ) button on
            the bottom right of the screen, or input `Ctrl/Cmd+Shift+r`.
            """
        ),
        "Console Output": (
            """
            Console output (e.g., `print()` statements) is shown below a
            cell.
            """
        ),
        "Creating, Moving, and Deleting Cells": (
            """
            1. _Create_ a new cell above or below a given one by clicking
                the plus button to the left of the cell, which appears on
                mouse hover.

            2. _Move_ a cell up or down by dragging on the handle to the
                right of the cell, which appears on mouse hover.

            3. _Delete_ a cell by clicking the trash bin icon. Bring it
                back by clicking the undo button on the bottom right of the
                screen, or with `Ctrl/Cmd+Shift+z`.
            """
        ),
        "Disabling Automatic Execution": (
            """
            Via the notebook settings (gear icon) or footer panel, you
            can disable automatic execution. This is helpful when
            working with expensive notebooks or notebooks that have
            side-effects like database transactions.
            """
        ),
        "Disabling Cells": (
            """
            You can disable a cell via the cell context menu.
            marimo will never run a disabled cell or any cells that depend on it.
            This can help prevent accidental execution of expensive computations
            when editing a notebook.
            """
        ),
        "Code Folding": (
            """
            You can collapse or fold the code in a cell by clicking the arrow
            icons in the line number column to the left, or by using keyboard
            shortcuts.

            Use the command palette (`Ctrl/Cmd+k`) or a keyboard shortcut to
            quickly fold or unfold all cells.
            """
        ),
        "Code Formatting": (
            """
            If you have [ruff](https://github.com/astral-sh/ruff) installed,
            you can format a cell with the keyboard shortcut `Ctrl/Cmd+b`.
            """
        ),
        "Command Palette": (
            """
            Use `Ctrl/Cmd+k` to open the command palette.
            """
        ),
        "Keyboard Shortcuts": (
            """
            Open the notebook menu (top-right) or input `Ctrl/Cmd+Shift+h` to
            view a list of all keyboard shortcuts.
            """
        ),
        "Configuration": (
            """
           Configure the editor by clicking the gears icon near the top-right
           of the screen.
           """
        ),
        "Exit & Shutdown": (
            """
           You can leave Marimo & shut down the server by clicking the
           circled X at the top right of the screen and responding
           to the prompt.

           :floppy_disk: _Be sure to save your work first!_
           """
        ),
    }
    return (tips,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
