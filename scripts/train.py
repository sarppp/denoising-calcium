"""Thin shim kept for backward compat. Use ``cidc train`` instead."""
from cidc.train import main
if __name__ == "__main__":
    main()
