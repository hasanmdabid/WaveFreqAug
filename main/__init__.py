# Makes `main` a regular package instead of an implicit namespace package.
# Without this, if `main/` ever ends up on sys.path by itself (e.g. via a
# PYTHONPATH env var pointing at it, which IDEs commonly set when a workspace
# is opened at this subfolder) — even behind the repo root in search order —
# Python's import system permanently resolves `import main` to the concrete
# module `main/main.py` instead of the `main/` package (PEP 420: a concrete
# module/package match anywhere on sys.path always wins over a namespace
# package, regardless of position), breaking every `from main.X import Y`
# with "ModuleNotFoundError: No module named 'main.utils'; 'main' is not a
# package". Verified via reproduction that this file fixes it even with a
# poisoned PYTHONPATH still present.
