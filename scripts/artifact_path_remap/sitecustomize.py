"""Redirect the compute-volume paths recorded inside the released artifacts.

The analysis reports in the artifact bundle record absolute paths of the volume on which the
study ran (`/lambda/nfs/data-imu-us-east3/...`, canonical alias
`/home/ubuntu/data-imu-us-east3/...`). The summary scripts follow those paths. To rerun them on
a restored copy without modifying any artifact, put this directory on PYTHONPATH and point
LMB_REMAP_TARGET at the directory that contains `vlmbehave-artifacts/`:

    export PYTHONPATH=scripts/artifact_path_remap LMB_REMAP_TARGET=/data/lmbehave/
    python3 scripts/summarize_main_track_instrument_validation.py ...

Python imports `sitecustomize` automatically at start-up; the module wraps `open` and the
`os.stat` family so that reads of the old prefixes resolve to the restored copy.
"""
import builtins
import io
import os

OLD_PREFIXES = ("/lambda/nfs/data-imu-us-east3/", "/home/ubuntu/data-imu-us-east3/")
TARGET = os.environ.get("LMB_REMAP_TARGET")


def _remap(path):
    if not TARGET:
        return path
    try:
        text = os.fspath(path)
    except TypeError:
        return path
    if isinstance(text, str):
        for old in OLD_PREFIXES:
            if text.startswith(old):
                return os.path.join(TARGET, text[len(old):])
    return path


if TARGET:
    _io_open = io.open

    def _open(file, *args, **kwargs):
        return _io_open(_remap(file), *args, **kwargs)

    io.open = _open
    builtins.open = _open

    for _name in ("stat", "lstat", "scandir", "listdir", "access"):
        _orig = getattr(os, _name)

        def _wrap(orig):
            def wrapped(path=".", *args, **kwargs):
                return orig(_remap(path), *args, **kwargs)
            return wrapped

        setattr(os, _name, _wrap(_orig))
