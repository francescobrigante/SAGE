# =====================================================================
# FINAL EVAL — paper tables (SELF-CONTAINED). Reads ONLY runs/final_eval/.
# 4 tables: (1) FMA recon, (2) MoisesDB-10s recon, (3) FMA MAEB, (4) MoisesDB MAEB.
# Recon tables also report n_files (full-set correctness check) + pure inference
# timing (infer_ms/file median, RTF = infer_sec/audio_sec). MAEB = task main_score.
# =====================================================================
import csv as _csv, json
from pathlib import Path
import numpy as np, pandas as pd
from IPython.display import display, Markdown

FE = Path("../runs/final_eval")
MODELS = ["sao-vae", "codicodec", "music2latent", "same", "SAGE"]

def _recon_subdir(m):  return "SAGE_e299" if m == "SAGE" else m           # ckpt stem for SAGE
def _maeb_root(m):     return FE / "maeb" / ("maeb_sage" if m == "SAGE" else f"maeb_{m}")
def _maeb_label(m):    return "same*" if m == "same" else m               # 256-d caveat marker

# ---------- reconstruction loaders (mirror load_checkpoint, cells 0/9) ----------
def _col(path, col):
    out = []
    with open(path, newline="") as f:
        for r in _csv.DictReader(f):
            try: out.append(float(r[col]))
            except (KeyError, ValueError, TypeError): pass
    return np.asarray(out, dtype=float)

def _scalar(path, col="score"):
    with open(path, newline="") as f:
        for r in _csv.DictReader(f):
            try: return float(r[col])
            except (KeyError, ValueError, TypeError): return float("nan")
    return float("nan")

def load_recon(metrics: Path) -> dict:
    d = {}
    sp = metrics / "spectral.csv"
    if sp.exists():
        for k in ("si_sdr", "sdr", "stft_loss", "mel_loss"): d[k] = _col(sp, k)
    if (metrics / "cdpam.csv").exists(): d["cdpam"] = _col(metrics / "cdpam.csv", "cdpam")
    for fn, k in (("clap_music.csv", "clap_music"), ("clap_audio.csv", "clap_audio")):
        if (metrics / fn).exists(): d[k] = _col(metrics / fn, "cosine")
    if (metrics / "clap_score.csv").exists():                              # legacy combined
        for k in ("clap_music", "clap_audio"):
            d.setdefault(k, _col(metrics / "clap_score.csv", k))
    for fn, k in (("fad_mert.csv", "fad_mert"), ("fad_gudgud.csv", "fad_gudgud")):
        if (metrics / fn).exists(): d[k] = _scalar(metrics / fn, "score")
    if (metrics / "timing.csv").exists():
        d["_infer"] = _col(metrics / "timing.csv", "infer_sec")
        d["_audio"] = _col(metrics / "timing.csv", "audio_sec")
    return d

PERFILE = ["si_sdr", "sdr", "stft_loss", "mel_loss", "cdpam", "clap_music", "clap_audio"]
SCALAR  = ["fad_mert", "fad_gudgud"]
HIGHER  = {"si_sdr", "sdr", "clap_music", "clap_audio"}                    # else lower-is-better

def recon_table(root: Path, title: str):
    rows = []
    for m in MODELS:
        met = root / _recon_subdir(m) / "metrics"
        row = {"model": m}
        if met.exists():
            d = load_recon(met)
            for k in PERFILE:
                arr = np.array(d.get(k, []))
                arr = arr[np.isfinite(arr)]
                row[k] = float(np.mean(arr)) if len(arr) else float("nan")
            for k in SCALAR:  row[k] = d.get(k, float("nan"))
            row["n_files"] = int(len(d.get("si_sdr", [])))
            it = d.get("_infer", np.array([]))
            if len(it):
                row["infer_ms/file"] = float(np.median(it)) * 1e3
                at = d.get("_audio")
                row["RTF"] = float(np.median(it / np.clip(at, 1e-9, None))) if at is not None and len(at) == len(it) else float("nan")
            else:
                row["infer_ms/file"] = row["RTF"] = float("nan")
        rows.append(row)
    disp = PERFILE + SCALAR + ["n_files", "infer_ms/file", "RTF"]
    df = pd.DataFrame(rows).set_index("model").reindex(columns=disp)
    lower = (set(PERFILE + SCALAR) - HIGHER) | {"infer_ms/file", "RTF"}
    def hl(s):
        if s.name == "n_files": return [""] * len(s)
        fin = s[s.notna()]
        if fin.empty: return [""] * len(s)
        best = fin.min() if s.name in lower else fin.max()
        return ["font-weight:bold;color:#00cc66" if (pd.notna(v) and v == best) else "" for v in s]
    fmt = {**{c: "{:.4f}" for c in PERFILE + SCALAR}, "cdpam": "{:.3f}",
           "n_files": "{:.0f}", "infer_ms/file": "{:.1f}", "RTF": "{:.3f}"}
    display(Markdown(f"### {title}"))
    display(df.style.apply(hl).format(fmt, na_rep="—"))
    return df

# ---------- MAEB loaders (mirror cells 3/6/8) ----------
def _main_score(path: Path):
    d = json.loads(path.read_text())
    for sv in d.get("scores", {}).values():
        if isinstance(sv, list):
            for s in sv:
                if isinstance(s, dict) and "main_score" in s: return float(s["main_score"])
        elif isinstance(sv, dict) and "main_score" in sv:
            return float(sv["main_score"])
    return None

def load_maeb(root: Path) -> dict:
    res = {}
    for local in root.glob("*/local"):
        for jp in local.glob("*.json"):
            if jp.stem in ("summary", "model_meta"): continue
            sc = _main_score(jp)
            if sc is not None: res[jp.stem] = sc
    return res

CORE_FMA = ["FMAGenreClassification", "FMAGenreClustering", "FMAArtistClustering",
            "FMAArtistA2ARetrieval", "FMAGenreAudioReranking", "FMAArtistPairClassification"]
EXTRA    = ["GTZANGenre", "MusicGenreClustering"]
MOIS     = ["MoisesDBGenreClassification", "MoisesDBGenreClustering", "MoisesDBArtistClustering",
            "MoisesDBArtistA2ARetrieval", "MoisesDBGenreAudioReranking",
            "MoisesDBArtistPairClassification", "MoisesDBInstrumentClassification"]

def maeb_table(tasks, core, title, note=None):
    data = {m: load_maeb(_maeb_root(m)) for m in MODELS}
    rows = []
    for m in MODELS:
        r = [data[m].get(t) for t in tasks]
        fin = [v for v in (data[m].get(t) for t in core) if v is not None]
        r.append(sum(fin) / len(core) if core else float("nan"))
        rows.append(r)
    df = pd.DataFrame(rows, index=[_maeb_label(m) for m in MODELS], columns=tasks + ["AVG"])
    def hl(s):
        fin = s[s.notna()]
        if fin.empty: return [""] * len(s)
        best = fin.max()
        return ["font-weight:bold;color:#4C72B0" if (pd.notna(v) and v == best) else "" for v in s]
    display(Markdown(f"### {title}"))
    display(df.style.apply(hl, axis=0).format("{:.4f}", na_rep="—"))
    if note: display(Markdown(note))
    return df

# ---------- render ----------
display(Markdown("# FINAL EVAL — paper tables  \n_Self-contained; reads `runs/final_eval/` only._"))
recon_fma  = recon_table(FE / "recon_fma",            "1) FMA — reconstruction (full test split)")
recon_mois = recon_table(FE / "recon_moisesdb_10s",  "2) MoisesDB-10s — reconstruction (full 1998)")
_NOTE = ("*`same*` = native 256-d (~×32 compression, **not** width-matched vs the 64-d models) "
         "→ MAEB scores are upper-biased, especially classification.*")
maeb_fma  = maeb_table(CORE_FMA + EXTRA, CORE_FMA, "3) FMA — MAEB probing (6 core + 2 extra, AVG over core)", _NOTE)
maeb_mois = maeb_table(MOIS, MOIS,                 "4) MoisesDB — MAEB probing (7 tasks, AVG over all)", _NOTE)
