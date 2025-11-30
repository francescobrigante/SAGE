import torch, torchaudio, soundfile as sf
import os
import numpy as np
from ar_spectra.models.eulero_inference import EuleroEncodeDecode
import argparse
from rich.console import Console

console = Console()

def ok(msg: str) -> None:
    console.print(msg, style="bold green")


def warn(msg: str) -> None:
    console.print(msg, style="bold yellow")


def err(msg: str) -> None:
    console.print(msg, style="bold red")
    
   
parser = argparse.ArgumentParser(description="Generate test predictions using a trained Eulero model")
parser.add_argument("--model-checkpoint", type=str, required=True, help="Path to the trained model checkpoint")
parser.add_argument("--target-dir", type=str, required=True, help="Path to the test audio directory")
parser.add_argument("--output-dir", type=str, required=True, help="Path to save the generated predictions")
