"""Entry point for `python -m merge_cli`.

The console scripts are the documented way in, but on Windows they routinely
are not usable: a system-wide Python install puts merge.exe in ...\\Scripts\\
and pip warns that the directory is not on PATH, so `merge` is "not recognized"
even though the install succeeded. `python -m merge_cli` always works, because
it goes through the interpreter that has the package installed.

merge_cli.cli is runnable the same way for the same reason; this module just
gives it the shorter name.
"""
from .cli import cli

if __name__ == "__main__":
    cli()
