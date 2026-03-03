from rich.console import Console

_console = Console()

def ok(msg: str) -> None:
    _console.print(msg, style="bold green")

def warn(msg: str) -> None:
    _console.print(msg, style="bold yellow")

def err(msg: str) -> None:
    _console.print(msg, style="bold red")

def info(msg: str) -> None:
    _console.print(msg, style="cyan")
