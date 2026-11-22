"""
patch_timestamp_logger.py

Copied next to FaceSwapDFM.py by patch_timestamp_installer.py's `install` command.
Owns per-frame CSV writing and run metadata, off the measured hot path.

Field names and semantics are fixed by the project's predefined dataset
interface. Do not rename fields for style; see DESIGN_REPORT.md.

DRAVEN DATASET INTEGRATION
--------------------------
The Draven orchestrator launches this logger in control mode. Preflight and
real-video warm-up frames are counted by the heartbeat but are not written to
a run CSV. The orchestrator sends begin_run/end_run commands through the
DRAVEN_DFL_CONTROL_DIR control directory. begin_run opens the per-run CSV and
can specify start_after_monotonic_ns so pre-T=0 keepalive frames are excluded.

  * Every written row carries `run_id` as its FIRST column.
  * `t_frame_input_ns` is emitted as an alias of `t_capture_ns` (same instant,
    no second clock read). verify.py requires this column to be monotonic on F1.
  * run_metadata.json carries the run identity supplied by the orchestrator.

All ORIGINAL columns keep their names, meanings and relative order. New
columns are additive: `run_id` is prepended and `t_frame_input_ns` follows
`t_capture_ns`.

Standalone/manual mode remains available when no Draven run/control environment
is supplied; DFL_INSTRUMENT_OUTPUT_DIR and DFL_INSTRUMENT_RUN_ID retain their
legacy behavior.
"""
import atexit
import csv
import json
import os
import platform
import queue
import re
import sys
import threading
import time
import uuid
from dataclasses import dataclass, fields
from typing import Optional

INSTRUMENT_VERSION = "1.3.0"
CONTROL_DIR_DEFAULT = "/tmp/draven_dfl_control"

CSV_FIELDS = [
    "run_id",
    "frame_id",
    "t_capture_ns",
    "t_frame_input_ns",
    "t_inference_start_ns",
    "t_inference_end_ns",
    "t_pass1_start_ns",
    "t_pass1_end_ns",
    "t_pass2_start_ns",
    "t_pass2_end_ns",
    "second_pass_ran",
    "frame_dropped",
]


@dataclass
class InstrumentEvent:
    run_id: Optional[str] = None
    frame_id: Optional[int] = None
    t_capture_ns: Optional[int] = None
    t_frame_input_ns: Optional[int] = None
    t_inference_start_ns: Optional[int] = None
    t_inference_end_ns: Optional[int] = None
    t_pass1_start_ns: Optional[int] = None
    t_pass1_end_ns: Optional[int] = None
    t_pass2_start_ns: Optional[int] = None
    t_pass2_end_ns: Optional[int] = None
    second_pass_ran: int = 0
    frame_dropped: int = 0

    def as_csv_row(self):
        row = {}
        for f in fields(self):
            v = getattr(self, f.name)
            row[f.name] = "" if v is None else v
        return row


# ---- Pass-timing helper methods (predefined names, do not rename) ----------

def mark_pass1_start(event: InstrumentEvent):
    event.t_pass1_start_ns = time.monotonic_ns()
    if event.t_inference_start_ns is None:
        event.t_inference_start_ns = event.t_pass1_start_ns


def mark_pass1_end(event: InstrumentEvent):
    event.t_pass1_end_ns = time.monotonic_ns()
    event.t_inference_end_ns = event.t_pass1_end_ns


def mark_pass2_start(event: InstrumentEvent):
    event.t_pass2_start_ns = time.monotonic_ns()


def mark_pass2_end(event: InstrumentEvent):
    event.t_pass2_end_ns = time.monotonic_ns()
    event.t_inference_end_ns = event.t_pass2_end_ns
    event.second_pass_ran = 1  # set only here — never merely because two_pass is enabled


# Draven run_id = {group}_R{rep:02d}; group = {condition}-{network}-{load}-
# {resolution}-{model}-{quality}  (see runs_index.py GROUP_PATTERN). Parsed only
# to label run_metadata.json; a non-matching id (standalone use) is left
# unlabelled rather than guessed.
_GS = r"[A-Za-z0-9]+"
_DRAVEN_RUN_RE = re.compile(
    rf"^(?P<group>(?P<condition>{_GS})-(?P<network>{_GS})-(?P<load>{_GS})-"
    rf"(?P<resolution>{_GS}p?)-(?P<model>{_GS})-(?P<quality>{_GS}))_R(?P<rep>\d+)$"
)

_CONDITION_TO_PATH = {"R0": "direct", "R1": "local_passthrough", "F1": "local_inference"}


def parse_draven_run_id(run_id: str) -> Optional[dict]:
    """Returns the Draven identity fields encoded in run_id, or None if run_id
    does not follow the Draven naming pattern."""
    m = _DRAVEN_RUN_RE.match(run_id or "")
    if not m:
        return None
    d = m.groupdict()
    return {
        "group_id": d["group"],
        "draven_condition": d["condition"],
        "network_profile": d["network"],
        "load_profile": d["load"],
        "draven_resolution": d["resolution"],
        "dfl_model": d["model"],
        "quality_preset": d["quality"],
        "repetition": int(d["rep"]),
        "path": _CONDITION_TO_PATH.get(d["condition"]),
    }


# ---- Async CSV writer -------------------------------------------------------

CONTROL_QUEUE_MAXSIZE = 2048


class _QueuedEvent:
    def __init__(self, event: InstrumentEvent, run_id: Optional[str]):
        self.event = event
        self.run_id = run_id


class _ControlCommand:
    def __init__(self, payload: dict):
        self.payload = payload
        self.done = threading.Event()
        self.result = {}


class _CsvWriter:
    """
    Background-thread CSV writer.

    In legacy mode this preserves the v1.1.0 file-writing behavior. In control
    mode the same writer thread also owns the active-run state and consumes
    begin_run/end_run commands from the control poller. This keeps frame
    counting and file writes off the measured DFL path.
    """

    def __init__(self, run_dir: str, flush_every: int = 5,
                 csv_path: Optional[str] = None, run_id: Optional[str] = None,
                 control_mode: bool = False, model_metadata: Optional[dict] = None):
        self._run_id = run_id
        self._path = csv_path or os.path.join(run_dir, "frame_events.csv")
        self._flush_every = flush_every
        self._q: "queue.Queue" = queue.Queue(maxsize=CONTROL_QUEUE_MAXSIZE)
        self._stop = threading.Event()
        self._control_mode = control_mode
        self._model_metadata = dict(model_metadata or {})
        self._fh = None
        self._writer = None
        self._count = 0
        self._state_lock = threading.Lock()
        self._submit_lock = threading.Lock()
        self._submission_run_id = None if control_mode else run_id
        self._queue_dropped_total = 0
        self._active_run_id = None if control_mode else run_id
        self._active_start_after_ns = 0
        self._active_path = None
        self._metadata_path = None
        self._ticks_total = 0
        self._frames_processed_total = 0
        self._rows_written_current_run = 0
        self._rows_processed_current_run = 0
        self._rows_dropped_frames_current_run = 0
        self._first_t_capture_ns = None
        self._last_t_capture_ns = None

        if not control_mode:
            self._open_legacy_file(self._path)

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        atexit.register(self.close)

    def _open_legacy_file(self, path: str):
        path = os.path.abspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        is_new = not os.path.exists(path)
        self._fh = open(path, "a", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=CSV_FIELDS)
        if is_new:
            self._writer.writeheader()
            self._fh.flush()

    def _close_file(self):
        if self._fh is None:
            return
        try:
            self._fh.flush()
            self._fh.close()
        finally:
            self._fh = None
            self._writer = None
            self._count = 0

    def emit(self, event: InstrumentEvent):
        # In control mode, bind the event to the logical run at enqueue time.
        # A shared submission lock linearizes frame enqueueing with begin/end
        # commands, so frames submitted before a switch cannot be stamped with
        # the new run by the writer thread later.
        if self._control_mode:
            with self._submit_lock:
                run_id = self._submission_run_id
                try:
                    self._q.put_nowait(_QueuedEvent(event, run_id))
                except queue.Full:
                    self._queue_dropped_total += 1
            return
        try:
            self._q.put_nowait(event)
        except queue.Full:
            self._queue_dropped_total += 1

    def command(self, payload: dict, timeout: float = 10.0):
        cmd = _ControlCommand(payload)
        action = payload.get("action")
        with self._submit_lock:
            # Linearize commands with frame enqueueing. The submission state is
            # advanced before the lock is released, while the command itself is
            # already ahead of any subsequent frame in the queue.
            if action == "begin_run" and self._submission_run_id is None:
                self._submission_run_id = payload.get("run_id")
            elif action == "end_run" and self._submission_run_id is not None:
                self._submission_run_id = None
            try:
                self._q.put_nowait(cmd)
            except queue.Full:
                # Restore the logical state if the command could not be queued.
                if action == "begin_run" and self._submission_run_id == payload.get("run_id"):
                    self._submission_run_id = None
                elif action == "end_run" and self._submission_run_id is None and self._active_run_id is not None:
                    self._submission_run_id = self._active_run_id
                return {
                    "command_id": payload.get("command_id"),
                    "action": action,
                    "run_id": payload.get("run_id"),
                    "status": "error",
                    "detail": "write queue full",
                    "t_ack_wall_ns": time.time_ns(),
                }
        if not cmd.done.wait(timeout):
            return {"status": "error", "detail": "writer command timed out",
                    "command_id": payload.get("command_id"),
                    "action": action, "run_id": payload.get("run_id"),
                    "t_ack_wall_ns": time.time_ns()}
        return cmd.result

    def _handle_event(self, item):
        event = item.event if isinstance(item, _QueuedEvent) else item
        queued_run_id = item.run_id if isinstance(item, _QueuedEvent) else None
        self._ticks_total += 1
        if not event.frame_dropped:
            self._frames_processed_total += 1

        if not self._control_mode:
            if event.run_id is None:
                event.run_id = self._run_id
            if event.t_frame_input_ns is None:
                event.t_frame_input_ns = event.t_capture_ns
            self._writer.writerow(event.as_csv_row())
            self._count += 1
            if self._count % self._flush_every == 0:
                self._fh.flush()
            return

        # CONTROL MODE: idle events are counted but never written. The run id
        # comes from enqueue time, not writer time, to make begin/end switches
        # race-safe.
        if queued_run_id is None or self._active_run_id != queued_run_id or self._writer is None:
            return
        # begin_run may be armed before synchronized T=0. Keep DFL's warm-up
        # and pre-start keepalive frames out of the measured CSV while still
        # allowing the instrumentation heartbeat/counters to run.
        if self._active_start_after_ns:
            capture_ns = event.t_capture_ns
            if capture_ns is None or int(capture_ns) < self._active_start_after_ns:
                return

        event.run_id = queued_run_id
        if event.t_frame_input_ns is None:
            event.t_frame_input_ns = event.t_capture_ns
        self._writer.writerow(event.as_csv_row())
        self._count += 1
        self._rows_written_current_run += 1
        if event.frame_dropped:
            self._rows_dropped_frames_current_run += 1
        else:
            self._rows_processed_current_run += 1
        if self._first_t_capture_ns is None and event.t_capture_ns is not None:
            self._first_t_capture_ns = event.t_capture_ns
        if event.t_capture_ns is not None:
            self._last_t_capture_ns = event.t_capture_ns
        if self._count % self._flush_every == 0:
            self._fh.flush()

    def _write_metadata(self, payload: dict):
        metadata_path = os.path.abspath(payload["metadata_path"])
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        identity = payload.get("identity") or {}
        metadata = {
            "run_id": payload["run_id"],
            "identity": identity,
            "instrument_version": INSTRUMENT_VERSION,
            "start_time_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "os": platform.platform(),
            "python_version": sys.version,
            **self._model_metadata,
        }
        for key in ("condition", "start_after_monotonic_ns"):
            if key in payload:
                metadata[key] = payload[key]
        tmp = metadata_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(metadata, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, metadata_path)
        self._metadata_path = metadata_path

    def _handle_command(self, cmd: _ControlCommand):
        payload = cmd.payload
        action = payload.get("action")
        try:
            if action == "begin_run":
                run_id = payload.get("run_id")
                frame_log_path = payload.get("frame_log_path")
                metadata_path = payload.get("metadata_path")
                if not run_id or not frame_log_path or not metadata_path:
                    raise ValueError("begin_run requires run_id, frame_log_path and metadata_path")
                if self._active_run_id is not None:
                    raise RuntimeError(f"run already active: {self._active_run_id}")

                path = os.path.abspath(frame_log_path)
                metadata = os.path.abspath(metadata_path)
                self._active_path = path
                self._rows_written_current_run = 0
                self._rows_processed_current_run = 0
                self._rows_dropped_frames_current_run = 0
                self._first_t_capture_ns = None
                self._last_t_capture_ns = None
                try:
                    self._active_start_after_ns = max(0, int(payload.get("start_after_monotonic_ns", 0) or 0))
                except (TypeError, ValueError):
                    raise ValueError("start_after_monotonic_ns must be a non-negative integer")

                # Prepare everything before publishing the active run. If any
                # filesystem operation fails, no half-started run is exposed.
                self._open_legacy_file(path)
                try:
                    self._write_metadata({**payload, "metadata_path": metadata})
                except Exception:
                    self._close_file()
                    self._active_path = None
                    raise

                self._active_run_id = run_id
                cmd.result = {
                    "command_id": payload.get("command_id"),
                    "action": action,
                    "run_id": run_id,
                    "status": "ok",
                    "detail": "",
                    "t_ack_wall_ns": time.time_ns(),
                }
            elif action == "end_run":
                if self._active_run_id is None:
                    raise RuntimeError("no active run")
                active_run_id = self._active_run_id
                self._close_file()
                cmd.result = {
                    "command_id": payload.get("command_id"),
                    "action": action,
                    "run_id": active_run_id,
                    "status": "ok",
                    "detail": "",
                    "t_ack_wall_ns": time.time_ns(),
                    "rows_written": self._rows_written_current_run,
                    "rows_processed": self._rows_processed_current_run,
                    "rows_dropped_frames": self._rows_dropped_frames_current_run,
                    "first_t_capture_ns": self._first_t_capture_ns,
                    "last_t_capture_ns": self._last_t_capture_ns,
                }
                self._active_run_id = None
                self._active_start_after_ns = 0
                self._active_path = None
                self._metadata_path = None
                self._rows_written_current_run = 0
                self._rows_processed_current_run = 0
                self._rows_dropped_frames_current_run = 0
                self._first_t_capture_ns = None
                self._last_t_capture_ns = None
            else:
                raise ValueError(f"unknown action: {action}")
        except Exception as exc:
            if action == "begin_run" and self._submission_run_id == payload.get("run_id"):
                with self._submit_lock:
                    if self._submission_run_id == payload.get("run_id"):
                        self._submission_run_id = None
            cmd.result = {
                "command_id": payload.get("command_id"),
                "action": action,
                "run_id": payload.get("run_id"),
                "status": "error",
                "detail": f"{type(exc).__name__}: {exc}",
                "t_ack_wall_ns": time.time_ns(),
            }
        finally:
            cmd.done.set()

    def snapshot_status(self):
        return {
            "pid": os.getpid(),
            "instrument_version": INSTRUMENT_VERSION,
            "mode": "control" if self._control_mode else "legacy",
            "active_run_id": self._active_run_id,
            "updated_wall_ns": time.time_ns(),
            "ticks_total": self._ticks_total,
            "frames_processed_total": self._frames_processed_total,
            "rows_written_current_run": self._rows_written_current_run,
            "queue_maxsize": CONTROL_QUEUE_MAXSIZE,
            "queue_size": self._q.qsize(),
            "queue_dropped_total": self._queue_dropped_total,
        }

    def _run(self):
        while not self._stop.is_set() or not self._q.empty():
            try:
                item = self._q.get(timeout=0.25)
            except queue.Empty:
                continue
            if isinstance(item, _ControlCommand):
                self._handle_command(item)
            else:
                self._handle_event(item)

    def close(self):
        if self._stop.is_set():
            return
        self._stop.set()
        self._thread.join(timeout=2.0)
        self._close_file()


class _ControlPoller:
    def __init__(self, writer: _CsvWriter, control_dir: str):
        self._writer = writer
        self._dir = os.path.abspath(control_dir)
        self._stop = threading.Event()
        os.makedirs(self._dir, exist_ok=True)
        self._last_command_id = self._read_initial_command_id()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _read_initial_command_id(self):
        """Treat a valid pre-existing command as stale, never as new work."""
        path = os.path.join(self._dir, "control.json")
        try:
            with open(path) as f:
                payload = json.load(f)
            command_id = payload.get("command_id")
            return command_id or None
        except (OSError, json.JSONDecodeError, AttributeError):
            # A malformed/half-written file is ignored until a later valid
            # command replaces it. It must not kill the heartbeat thread.
            return None

    @staticmethod
    def _atomic_json(path: str, payload: dict):
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    def _write_status(self):
        self._atomic_json(os.path.join(self._dir, "status.json"), self._writer.snapshot_status())

    def _run(self):
        while not self._stop.is_set():
            try:
                self._write_status()
                control_path = os.path.join(self._dir, "control.json")
                if os.path.exists(control_path):
                    with open(control_path) as f:
                        payload = json.load(f)
                    command_id = payload.get("command_id")
                    if command_id and command_id != self._last_command_id:
                        result = self._writer.command(payload)
                        self._last_command_id = command_id
                        self._atomic_json(os.path.join(self._dir, "ack.json"), result)
            except Exception:
                # Phase 3b hardens malformed/stale control files. The heartbeat
                # itself must remain alive in this core phase.
                pass
            self._stop.wait(1.0)

    def close(self):
        self._stop.set()
        self._thread.join(timeout=2.0)


_writer_singleton: Optional[_CsvWriter] = None
_control_poller: Optional[_ControlPoller] = None
_lock = threading.Lock()


def _is_control_mode() -> bool:
    return not any(os.environ.get(name) for name in (
        "DRAVEN_RUN_ID", "DRAVEN_FRAME_LOG", "DFL_INSTRUMENT_RUN_ID",
        "DFL_INSTRUMENT_OUTPUT_DIR",
    ))


def init_logger(output_dir: str, run_id: str, condition: str, model_name: str,
                resolution: str, target_fps: float, two_pass_enabled: bool,
                onnxruntime_device: str = "unknown",
                csv_path: Optional[str] = None):
    """Initialize the legacy writer or the long-lived control-mode logger."""
    global _writer_singleton, _control_poller
    with _lock:
        if _writer_singleton is not None:
            return _writer_singleton

        control_mode = _is_control_mode()
        if control_mode:
            control_dir = os.environ.get("DRAVEN_DFL_CONTROL_DIR", CONTROL_DIR_DEFAULT)
            _writer_singleton = _CsvWriter(
                os.path.abspath(control_dir), run_id=None, control_mode=True,
                model_metadata={
                    "model_name": model_name,
                    "resolution": resolution,
                    "target_fps": target_fps,
                    "two_pass_enabled": bool(two_pass_enabled),
                    "onnxruntime_device": onnxruntime_device,
                },
            )
            _control_poller = _ControlPoller(_writer_singleton, control_dir)
            return _writer_singleton

        # v1.1.0 legacy path preserved.
        if csv_path:
            run_dir = os.path.dirname(os.path.abspath(csv_path))
        else:
            run_dir = os.path.join(output_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)
        meta_path = os.path.join(run_dir, "run_metadata.json")
        if not os.path.exists(meta_path):
            meta = {
                "run_id": run_id,
                "source_revision": "fc7b787",
                "condition": condition,
                "model_name": model_name,
                "resolution": resolution,
                "target_fps": target_fps,
                "two_pass_enabled": two_pass_enabled,
                "os": platform.platform(),
                "python_version": sys.version,
                "onnxruntime_device": onnxruntime_device,
                "start_time_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "instrument_version": INSTRUMENT_VERSION,
            }
            draven = parse_draven_run_id(run_id)
            if draven is not None:
                meta["draven"] = draven
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
        _writer_singleton = _CsvWriter(run_dir, csv_path=csv_path, run_id=run_id)
        return _writer_singleton


def emit_event(event: InstrumentEvent):
    if _writer_singleton is None:
        return  # logger not initialized; drop rather than raise in the hot path
    _writer_singleton.emit(event)


def shutdown_logger():
    global _control_poller, _writer_singleton
    if _control_poller is not None:
        _control_poller.close()
        _control_poller = None
    if _writer_singleton is not None:
        _writer_singleton.close()
        _writer_singleton = None


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


def ensure_logger_from_env(model_name: str, resolution: str, two_pass_enabled: bool,
                            onnxruntime_device: str = "unknown"):
    """
    Self-configuring, idempotent logger bootstrap. Called automatically by the
    patched on_tick() once the DFM model is ready — no manual init_logger()
    call required at app startup.

    In Draven control mode, run identity and CSV paths arrive later through
    begin_run/end_run commands in DRAVEN_DFL_CONTROL_DIR. This keeps startup
    and warm-up frames out of the measured run. Legacy environment-based
    output remains available for standalone/manual use:
        DFL_INSTRUMENT_OUTPUT_DIR   (default: "./dfl_instrument_output")
        DFL_INSTRUMENT_RUN_ID       (default: a fresh uuid4 hex, so ad-hoc
                                     manual runs still get a distinct folder)
        DFL_INSTRUMENT_CONDITION    (default: "unknown" — set this explicitly
                                     for real runs, e.g. "dfl_single_pass")
        DFL_INSTRUMENT_TARGET_FPS   (default: 0, meaning "not specified")

    model_name / resolution / two_pass_enabled are read from the live DFM
    model/state instead of the environment, since those are DFL-internal
    facts already available at the call site — sourcing them from env vars
    would risk silently drifting from what's actually running.

    Cheap no-op after the first successful call (checked without acquiring
    the lock), so calling this on every tick — which the patch does, because
    on_tick() has no other reliable "model just became ready" hook without
    duplicating that state — adds no measurable overhead to the hot path.
    """
    if _writer_singleton is not None:
        return
    # Draven contract first, then the original standalone variables.
    draven_run_id = os.environ.get("DRAVEN_RUN_ID") or None
    draven_frame_log = os.environ.get("DRAVEN_FRAME_LOG") or None

    output_dir = os.environ.get("DFL_INSTRUMENT_OUTPUT_DIR", "./dfl_instrument_output")
    run_id = draven_run_id or os.environ.get("DFL_INSTRUMENT_RUN_ID") or new_run_id()

    # Condition: an explicit DFL_INSTRUMENT_CONDITION always wins. Otherwise, if
    # the run_id is a Draven id, take its condition label (R0/R1/F1) from it
    # rather than recording "unknown", which would make the run unidentifiable.
    condition = os.environ.get("DFL_INSTRUMENT_CONDITION")
    if not condition:
        parsed = parse_draven_run_id(run_id)
        condition = parsed["draven_condition"] if parsed else "unknown"

    try:
        target_fps = float(os.environ.get("DFL_INSTRUMENT_TARGET_FPS", "0") or 0)
    except ValueError:
        target_fps = 0.0
    init_logger(
        output_dir=output_dir,
        run_id=run_id,
        condition=condition,
        model_name=model_name,
        resolution=resolution,
        target_fps=target_fps,
        two_pass_enabled=two_pass_enabled,
        onnxruntime_device=onnxruntime_device,
        csv_path=draven_frame_log,
    )
