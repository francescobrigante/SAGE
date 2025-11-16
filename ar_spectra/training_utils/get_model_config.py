import torch


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
