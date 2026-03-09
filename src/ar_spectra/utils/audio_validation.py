# ==============================================================
# Audio Validation Utilities
#
#   contains functions for validating audio file decodability
# ==============================================================
import contextlib
import os
import tempfile
from typing import List, Optional, Tuple
import torchaudio

@contextlib.contextmanager
def stderr_to(path: Optional[str]):
    if path is None:
        yield None
        return
    fd = os.dup(2)
    try:
        with open(path, "w") as handle:
            os.dup2(handle.fileno(), 2)
            yield path
    finally:
        os.dup2(fd, 2)
        os.close(fd)

def read_stderr(path: Optional[str], fail_on_stderr: bool) -> Optional[str]:
    if not fail_on_stderr or not path or path == os.devnull:
        return None
    try:
        with open(path, "r", errors="ignore") as f:
            data = f.read().strip()
    finally:
        try:
            os.remove(path)
        except Exception:
            pass
    if not data:
        return None
    first_line = data.splitlines()[0]
    return f"decoder stderr: {first_line}"

def decode_chunk(
    path_str: str, 
    frame_offset: int, 
    num_frames: int, 
    quiet_decode: bool, 
    fail_on_stderr: bool
) -> Optional[str]:
    if fail_on_stderr:
        tmp = tempfile.NamedTemporaryFile(delete=False)
        tmp.close()
        capture_path = tmp.name
    elif quiet_decode:
        capture_path = os.devnull
    else:
        capture_path = None

    try:
        with stderr_to(capture_path):
            torchaudio.load(path_str, frame_offset=frame_offset, num_frames=num_frames)
        stderr_msg = read_stderr(capture_path, fail_on_stderr)
        if stderr_msg:
            return stderr_msg
        return None
    except Exception as exc:
        if capture_path and capture_path != os.devnull:
            try:
                os.remove(capture_path)
            except Exception:
                pass
        return f"{type(exc).__name__}: {exc}"

def compute_offsets(num_frames: Optional[int], probe_positions: int, probe_frames: int, full_scan: bool) -> List[int]:
    if full_scan or probe_positions <= 1 or not num_frames or num_frames <= 0:
        return [0]
    positions = max(1, min(3, probe_positions))
    offsets = [0]
    if positions >= 2:
        mid = max(0, (num_frames // 2) - (probe_frames // 2))
        offsets.append(mid)
    if positions >= 3:
        end = max(0, num_frames - probe_frames)
        offsets.append(end)
    return sorted(set(offsets))

def check_file(
    path_str: str, 
    probe_positions: int, 
    probe_frames: int, 
    full_scan: bool, 
    quiet_decode: bool, 
    fail_on_stderr: bool
) -> Tuple[str, Optional[str]]:
    """Checks an audio file for corruption by attempting to decode parts of it."""
    try:
        if full_scan:
            err = decode_chunk(path_str, 0, -1, quiet_decode, fail_on_stderr)
            return path_str, err

        num_frames = None
        if probe_positions > 1:
            try:
                info = torchaudio.info(path_str)
                num_frames = int(getattr(info, "num_frames", 0) or 0)
            except Exception:
                num_frames = None

        offsets = compute_offsets(num_frames, probe_positions, probe_frames, full_scan)
        for offset in offsets:
            err = decode_chunk(path_str, offset, probe_frames, quiet_decode, fail_on_stderr)
            if err:
                return path_str, err
        return path_str, None
    except Exception as exc:
        return path_str, f"{type(exc).__name__}: {exc}"
