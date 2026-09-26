# ==============================================================
# Console utils
#
#   contains utils for printing messages to the console
# ==============================================================

import os as _os
from rich.console import Console

_console = Console(no_color=bool(_os.environ.get("NO_COLOR")))

def ok(msg: str, prefix: str = "") -> None:
    prefix_str = f"[[bold cyan]{prefix}[/bold cyan]] " if prefix else ""
    _console.print(f"{prefix_str}{msg}", style="bold green")

def warn(msg: str, prefix: str = "") -> None:
    prefix_str = f"[[bold cyan]{prefix}[/bold cyan]] " if prefix else ""
    _console.print(f"{prefix_str}{msg}", style="bold yellow")

def err(msg: str, prefix: str = "") -> None:
    prefix_str = f"[[bold cyan]{prefix}[/bold cyan]] " if prefix else ""
    _console.print(f"{prefix_str}{msg}", style="bold red")

def info(msg: str, prefix: str = "") -> None:
    prefix_str = f"[[bold cyan]{prefix}[/bold cyan]] " if prefix else ""
    _console.print(f"{prefix_str}{msg}", style="cyan")
