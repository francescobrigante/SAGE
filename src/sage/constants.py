# =============================================================================
# Library defaults of the data pipeline and the training loop. They are not
# machine-specific (paths live in configs/paths/) and the paper runs never
# change them; the Hydra configs set the values that do vary per run.
# =============================================================================

DEFAULT_SEED = 94                         # dataset crops and the multi-corpus sampler
DEFAULT_AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus")

DEFAULT_AUDIO_LOAD_TIMEOUT = 60           # s, per file: aborts a stuck MP3 decode inside a worker
DEFAULT_DATALOADER_TIMEOUT = 120          # s, per batch: last safety net (e.g. NFS hangs)
DEFAULT_MAX_RETRIES_PER_SAMPLE = 8        # other files tried before a sample is given up
DEFAULT_MAX_PAD_RATIO = 0.05              # at most 5% of a training crop may be padding
DEFAULT_SILENCE_THRESHOLD = -62           # dBFS: crops quieter than this are redrawn

# Partial read: decode only a window around the crop instead of the whole file
# (a large win on multi-minute tracks). The margin, in source-domain samples,
# absorbs resampling edge transients so a full segment can always be cropped.
DEFAULT_PARTIAL_READ = False
DEFAULT_PARTIAL_READ_MARGIN = 4096
