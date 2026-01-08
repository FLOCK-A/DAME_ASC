"""Top-level CLI shim for DAME-ASC placeholder project."""

from typing import List
from dame_asc.cli import main


def _entry() -> int:
    # Delegate to package CLI. Use None to allow sys.argv.
    return main(None)


if __name__ == "__main__":
    raise SystemExit(_entry())

# End of file
