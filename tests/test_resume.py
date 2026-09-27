# ===============
# Resuming a run by name (sage/training/resume.py): the newest checkpoint under
# runs/<name>/*/ (SLURM requeue save or last.ckpt), the folder a resumed run continues in,
# and the W&B run id stored at its first launch.
# ===============
import os

from sage.training.resume import find_resume_checkpoint, read_run_id, resume_run_dir, run_dir_of


def _touch(path, mtime):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    os.utime(path, (mtime, mtime))
    return path


def test_no_run_folder_means_a_fresh_start(tmp_path):
    assert find_resume_checkpoint(tmp_path / "runs" / "never_ran") is None
    (tmp_path / "runs" / "empty" / "2026-01-01_00-00-00").mkdir(parents=True)
    assert find_resume_checkpoint(tmp_path / "runs" / "empty") is None


def test_the_newest_checkpoint_of_any_launch_is_resumed(tmp_path):
    root = tmp_path / "runs" / "pretrain"
    _touch(root / "d1" / "checkpoints" / "last.ckpt", 100)
    _touch(root / "d1" / "checkpoints" / "epoch_000.ckpt", 300)       # epoch files are not resume points
    requeue = _touch(root / "d1" / "hpc_ckpt_1.ckpt", 200)            # SIGUSR1 save, mid-epoch
    assert find_resume_checkpoint(root) == requeue
    later = _touch(root / "d2" / "checkpoints" / "last.ckpt", 250)
    assert find_resume_checkpoint(root) == later


def test_a_resumed_run_continues_in_its_own_folder(tmp_path):
    root = tmp_path / "runs" / "pretrain"
    last = _touch(root / "d1" / "checkpoints" / "last.ckpt", 1)
    hpc = _touch(root / "d1" / "hpc_ckpt_2.ckpt", 2)
    assert run_dir_of(last) == run_dir_of(hpc) == root / "d1"
    assert resume_run_dir(root, str(hpc)) == (root / "d1").resolve()
    other = _touch(tmp_path / "runs" / "other" / "d9" / "checkpoints" / "last.ckpt", 3)
    assert resume_run_dir(root, str(other)) is None                  # another run's ckpt: new folder
    assert resume_run_dir(root, None) is None


def test_the_w_and_b_run_id_of_the_first_launch(tmp_path):
    assert read_run_id(tmp_path, "pretrain") is None
    (tmp_path / "pretrain").write_text("abc123\n")
    assert read_run_id(tmp_path, "pretrain") == "abc123"
    (tmp_path / "blank").write_text("  \n")
    assert read_run_id(tmp_path, "blank") is None
