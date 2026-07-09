"""
etot_fit.py — DEPRECATED compatibility wrapper.

Use ``python main.py etot ...`` or ``from emtdb.etotfit import ETotFitter``
instead of calling this script directly.
"""

import sys
import warnings

warnings.warn(
    "etot_fit.py is deprecated; use 'python main.py etot' instead",
    DeprecationWarning,
    stacklevel=2,
)

from emtdb.etotfit import ETotFitter, E0FitResult  # noqa: F401

# Re-export public helpers for backward compatibility
_process_folder = None

# Direct CLI entry point — delegates to the module's main()
def main():
    """Deprecated entry point.  Redirects to ``python main.py etot``."""
    from emtdb.cli import main as cli_main

    # Prepend "etot" to argv so the CLI parser routes to cmd_etot
    sys.argv.insert(1, "etot")
    return cli_main()


if __name__ == "__main__":
    sys.exit(main())
