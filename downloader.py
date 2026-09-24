"""Download worker, run as a subprocess by server.py so a job can be cancelled.

Reads a JSON spec from stdin; the HF token comes from the HF_TOKEN env var.
Writes one JSON event per line to stdout.
"""
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import hf_hub_download
from tqdm.std import tqdm

_lock = threading.Lock()
_partial = {}  # path -> bytes received so far
_last_report = 0.0


def emit(**event):
    with _lock:
        print(json.dumps(event), flush=True)


def report(path, nbytes, force=False):
    global _last_report
    with _lock:
        _partial[path] = nbytes
        now = time.monotonic()
        if not force and now - _last_report < 0.5:
            return
        _last_report = now
        snapshot = dict(_partial)
    emit(event="progress", files=snapshot)


def progress_class(path):
    """A silent tqdm that forwards byte counts for one file.

    Plain HTTP downloads call update(); Xet downloads also call update_transfer() for
    network bytes, which arrive well before the file is written to disk.
    """

    class FileProgress(tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["disable"] = True
            super().__init__(*args, **kwargs)
            self.written = self.received = 0

        def update(self, n=1):
            self.written += n or 0
            report(path, max(self.written, self.received))

        def update_transfer(self, n=1):
            self.received += n or 0
            report(path, max(self.written, self.received))

        def set_transfer_postfix_str(self, *args, **kwargs):
            pass

    return FileProgress


def main():
    spec = json.loads(sys.stdin.read())
    token = os.environ.get("HF_TOKEN") or None

    def fetch(path):
        hf_hub_download(
            spec["repo_id"],
            path,
            repo_type=spec["repo_type"],
            revision=spec.get("revision") or None,
            local_dir=spec["local_dir"],
            token=token,
            tqdm_class=progress_class(path),
        )

    failed = 0
    with ThreadPoolExecutor(max_workers=spec.get("workers", 4)) as pool:
        futures = {pool.submit(fetch, p): p for p in spec["files"]}
        for fut in as_completed(futures):
            path = futures[fut]
            try:
                fut.result()
                emit(event="file_done", path=path)
            except Exception as exc:  # report per file and keep going
                failed += 1
                emit(event="file_error", path=path, error=f"{type(exc).__name__}: {exc}")

    emit(event="finished", ok=failed == 0)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
