# ===================================================================
# Model Info Utilities
#
#   contains helpers to extract module details and parameter counts
# ===================================================================

import torch.nn as nn

def _nbytes(model: nn.Module) -> int:
    """Compute total bytes occupied by model parameters."""
    return sum(p.nelement() * p.element_size() for p in model.parameters())

def extract_model_config(model: nn.Module) -> dict:
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

    tot_bytes = _nbytes(model)

    return {
        "class": model.__class__.__name__,
        "model_bytes": tot_bytes,
        "num_parameters_total": int(total_params),
        "num_parameters_trainable": int(trainable_params),
        "modules": modules[:512],  # limita la lunghezza per non esagerare nei log
        "repr": repr(model),
    }

import torch
from rich.console import Console
from sage.utils.console import warn

def log_compression_stats(wrapper, train_dl, console: Console) -> None:
    """esegue un dummy pass sul primo batch per loggare dinamicamente la compression rate."""
    was_training = wrapper.training
    try:
        wrapper.eval()
        with torch.no_grad():
            batch = next(iter(train_dl))
            sp_reals, orig_waveforms = batch[0], batch[1]

            # Extract exactly 1 sample from the batch to calculate single-sample dimensions
            sp_single = sp_reals[0:1].to(wrapper.device)
            wav_single = orig_waveforms[0:1]
            
            # Raw audio elements count
            raw_audio_elements = wav_single.shape[1] * wav_single.shape[2]
            
            # Pre-transform
            if wrapper.engine.force_input_mono and sp_single.shape[1] > 1:
                sp_single = sp_single.mean(dim=1, keepdim=True)
            if wrapper.engine.autoencoder.pre_transform_applies_to_target:
                sp_single = wrapper.engine.autoencoder.apply_pre_transform_to_target(sp_single)

            # Encoder pass
            enc_out = wrapper.engine.autoencoder.encode(sp_single, return_info=True)
            latents = enc_out[0] if isinstance(enc_out, tuple) else enc_out
            
            # Calc latent elements
            latent_sh = list(latents.shape)[1:] # remove batch dim
            latent_elements = 1
            for dim in latent_sh:
                latent_elements *= dim
            
            is_cplx = latents.is_complex() if hasattr(latents, 'is_complex') else False
            multiplier = 2 if is_cplx else 1
            
            # Compute real compression rate (independent of actual seconds since it's a ratio)
            compression_rate = raw_audio_elements / (latent_elements * multiplier)
            
            # Visual output
            console.print("")
            console.rule("[bold magenta]Model Compression Stats[/bold magenta]")
            console.print(f"   [cyan]Input Audio Shape:[/cyan] {list(wav_single.shape)[1:]} (Channels x Samples)")
            console.print(f"   [cyan]Latent Tensor Shape:[/cyan] {latent_sh} {'(Complex)' if is_cplx else '(Real)'}")
            console.print(f"   [cyan]True Compression Rate:[/cyan] [bold yellow]{compression_rate:.2f}x[/bold yellow]")
            console.rule()
            console.print("")

    except Exception as e:
        warn(f"Impossibile calcolare latent shape dinamica: {e}", prefix="TRAINER")
    finally:
        if was_training:
            wrapper.train()
