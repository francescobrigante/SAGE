import torch
from rich.console import Console
from einops import rearrange

_console = Console()

def ok(msg: str) -> None:
    _console.print(msg, style="bold green")

def warn(msg: str) -> None:
    _console.print(msg, style="bold yellow")

def err(msg: str) -> None:
    _console.print(msg, style="bold red")

def info(msg: str) -> None:
    _console.print(msg, style="cyan")



def extract_model_config(model: torch.nn.Module) -> dict:
    """Raccoglie informazioni chiave sulla configurazione del modello per logging."""
    try:
        modules = [f"{name}:{m.__class__.__name__}" for name, m in model.named_modules() if name]
    except Exception:
        modules = []
    total_params = 0
    for p in model.parameters():
        if p.is_complex():
            total_params += p.numel() * 2
        else:
            total_params += p.numel()

    trainable_params = 0
    for p in model.parameters():
        if p.requires_grad:
            if p.is_complex():
                trainable_params += p.numel() * 2
            else:
                trainable_params += p.numel()


    def nbytes(model):
        return sum(p.nelement() * p.element_size() for p in model.parameters())

    tot_bytes = nbytes(model)

    return {
        "class": model.__class__.__name__,
        "model_bytes": tot_bytes,
        "num_parameters_total": int(total_params),
        "num_parameters_trainable": int(trainable_params),
        "model_total_bytes": tot_bytes,
        "modules": modules[:512],  # limita la lunghezza per non esagerare nei log
        "repr": repr(model),
    }

def trim_to_shortest(a: torch.Tensor, b: torch.Tensor):
    """Trim the longer of two tensors to the length of the shorter one."""
    if a.shape[-1] > b.shape[-1]:
        return a[:,:,:b.shape[-1]], b
    elif b.shape[-1] > a.shape[-1]:
        return a, b[:,:,:a.shape[-1]]
    return a, b

def fold_channels_into_batch(x):
    x = rearrange(x, 'b c ... -> (b c) ...')
    return x

def unfold_channels_from_batch(x, channels):
    if channels == 1:
        return x.unsqueeze(1)
    x = rearrange(x, '(b c) ... -> b c ...', c = channels)
    return x
