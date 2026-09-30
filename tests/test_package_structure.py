# ===============
# Package layering, checked on the source: sage.nn (generic building blocks) never imports
# sage.model / sage.training / sage.inference; sage.model never imports sage.training; the complex
# blocks are reached only lazily. And inference stays light: importing sage.inference and loading
# a model pulls in no training, logging or complex-valued dependency.
# ===============
import ast
import subprocess
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "sage"


def _imports(path: Path):
    """(module, is_top_level) for every absolute import in a file."""
    tree = ast.parse(path.read_text())
    top = set(map(id, tree.body))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            yield node.module, id(node) in top
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, id(node) in top


def _files(subpackage):
    return sorted((SRC / subpackage).rglob("*.py"))


FORBIDDEN = {
    "nn": ("sage.model", "sage.training", "sage.inference"),
    "model": ("sage.training",),
    "utils": ("sage.model", "sage.training", "sage.inference", "sage.nn"),
}


@pytest.mark.parametrize("sub", sorted(FORBIDDEN))
def test_layering(sub):
    bad = [(p.relative_to(SRC), m) for p in _files(sub) for m, _ in _imports(p)
           if m.startswith(FORBIDDEN[sub])]
    assert not bad, f"sage.{sub} must not import {FORBIDDEN[sub]}: {bad}"


def test_complex_blocks_are_only_imported_lazily():
    eager = [(p.relative_to(SRC), m)
             for p in SRC.rglob("*.py") if "complex" not in p.relative_to(SRC).parts
             for m, top in _imports(p) if m.startswith("sage.nn.complex") and top]
    assert not eager, f"sage.nn.complex imported at module level outside the package: {eager}"


def test_inference_is_light():
    code = (
        "import sys, torch\n"
        "from sage.model.autoencoder import SAGEAutoencoder\n"
        "import sage.inference\n"
        "heavy = ('lightning', 'pytorch_lightning', 'wandb', 'librosa', 'laion_clap', 'complextorch',\n"
        "         'complexPyTorch', 'dotenv', 'config', 'sage.training', 'sage.nn.complex')\n"
        "print(','.join(m for m in heavy if m in sys.modules))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd="/", check=True)
    assert out.stdout.strip() == "", f"inference imports training/complex deps: {out.stdout.strip()}"
