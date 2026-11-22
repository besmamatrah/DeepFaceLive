"""Draven DFL stage diagnostics. Separate from the timestamp CSV."""
import json
import os
import time
from pathlib import Path

_DIAG_PATH = Path(os.environ.get(
    "DRAVEN_DFL_DIAGNOSTICS",
    "/tmp/draven_dfl_stage_diagnostics.jsonl",
))
_LAST_FRAME = {}

def emit(stage, **fields):
    """Append sampled stage evidence; diagnostics must never break DFL."""
    try:
        frame = fields.get("frame_id")
        key = (stage, fields.get("face_index"))
        if frame is not None:
            try:
                frame_num = int(frame)
            except (TypeError, ValueError):
                frame_num = None
            if frame_num is not None:
                previous = _LAST_FRAME.get(key)
                # Keep one event per stage/face every 30 frames.
                if previous is not None and frame_num - previous < 30:
                    return
                _LAST_FRAME[key] = frame_num

        record = {
            "t_monotonic_ns": time.monotonic_ns(),
            "stage": stage,
            **fields,
        }
        line = (json.dumps(record, default=str, separators=(",", ":")) + "\n").encode()
        _DIAG_PATH.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(
            str(_DIAG_PATH),
            os.O_WRONLY | os.O_CREAT | os.O_APPEND,
            0o600,
        )
        try:
            os.write(fd, line)
        finally:
            os.close(fd)
    except Exception:
        # Diagnostics are observational only; never crash a DFL worker.
        pass
