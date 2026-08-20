"""Entry point for ``python -m ragorc``.

Present so that both module forms work, since people reasonably try either:

    python -m ragorc query "..."        # this file
    python -m ragorc.cli query "..."    # the __main__ guard in cli.py
    ragorc query "..."                  # the console script from pyproject

All three reach the same Typer app.
"""

from ragorc.cli import app

app()
