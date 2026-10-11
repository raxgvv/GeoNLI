#!/usr/bin/env python3
"""Safely stop the backend and frontend recorded by start_geonli.py."""

from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import sys
import time


RUNTIME = Path("/tmp/geonli")
STATE = RUNTIME / "services.json"
LOCK = RUNTIME / "launcher.lock"


def proc_info(pid: int) -> tuple[str | None, int | None, int | None, str, bytes]:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat[stat.rfind(")") + 2 :].split()
        start_ticks = fields[19]
        pgid, sid = int(fields[2]), int(fields[3])
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
        environ = Path(f"/proc/{pid}/environ").read_bytes()
        return start_ticks, pgid, sid, cmdline, environ
    except (OSError, IndexError, ValueError):
        return None, None, None, "", b""


def matching_group_member(record: dict, pgid: int) -> bool:
    expected = "backend.app" if record.get("name") == "backend" else "vite"
    launch_id = record.get("launch_id")
    if not launch_id:
        return False
    marker = f"GEONLI_LAUNCH_ID={launch_id}".encode()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        _ticks, member_pgid, member_sid, command, environ = proc_info(int(entry.name))
        if (member_pgid == pgid and member_sid == pgid and expected in command.lower()
                and marker in environ.split(b"\0")):
            return True
    return False


def group_has_running_process(pgid: int) -> bool:
    """Ignore zombies, which cannot handle signals and may await their parent."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = (entry / "stat").read_text()
            fields = stat[stat.rfind(")") + 2 :].split()
            state, member_pgid = fields[0], int(fields[2])
        except (OSError, IndexError, ValueError):
            continue
        if member_pgid == pgid and state not in {"Z", "X"}:
            return True
    return False


def signal_record(record: dict, grace_seconds: float = 8.0) -> bool:
    """Return true only when the recorded launch process group is gone."""
    try:
        pid, pgid = int(record["pid"]), int(record["pgid"])
    except (KeyError, TypeError, ValueError):
        print(f"{record.get('name', 'Service')}: invalid process record; left untouched.")
        return False
    ticks, actual_pgid, actual_sid, command, environ = proc_info(pid)
    launch_id = record.get("launch_id")
    if launch_id:
        identity_matches = f"GEONLI_LAUNCH_ID={launch_id}".encode() in environ.split(b"\0")
    else:
        # Compatibility with state files from older launcher revisions.
        identity_matches = bool(record.get("identity", "") and record["identity"] in command)
    is_recorded_leader = (ticks == str(record.get("start_ticks")) and actual_pgid == pgid
                          and actual_sid == int(record.get("sid", pgid)) and identity_matches)
    if ticks is None or ticks != str(record.get("start_ticks")):
        # A missing or reused PID means the recorded launcher process is gone.
        # Only act if an unmistakable launched child remains in the same group.
        if not matching_group_member(record, pgid):
            print(f"{record.get('name', 'Service').capitalize()}: already stopped.")
            return True
    elif not is_recorded_leader and not matching_group_member(record, pgid):
        print(f"{record.get('name', 'Service').capitalize()}: process identity did not match; left untouched.")
        return False

    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        print(f"{record.get('name', 'Service').capitalize()}: already stopped.")
        return True
    print(f"{record.get('name', 'Service').capitalize()}: sent SIGTERM to its verified process group.")
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if not group_has_running_process(pgid):
            return True
        time.sleep(0.2)
    if not group_has_running_process(pgid):
        return True
    try:
        os.killpg(pgid, signal.SIGKILL)
        print(f"{record.get('name', 'Service').capitalize()}: required SIGKILL after timeout.")
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if not group_has_running_process(pgid):
            return True
        time.sleep(0.2)
    return not group_has_running_process(pgid)


def main() -> int:
    try:
        state = json.loads(STATE.read_text())
    except FileNotFoundError:
        print("GeoNLI launcher state not found; no launcher-managed services to stop.")
        return 0
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Cannot read runtime state at {STATE}: {exc}", file=sys.stderr)
        return 1

    services = state.get("services", {})
    stopped = True
    for name in ("frontend", "backend"):
        record = services.get(name)
        if record:
            stopped = signal_record(record) and stopped
        else:
            print(f"{name.capitalize()}: no process record.")
    if stopped:
        STATE.unlink(missing_ok=True)
        LOCK.unlink(missing_ok=True)
        try:
            RUNTIME.rmdir()
        except OSError:
            pass
        print("GeoNLI launcher-managed services are stopped. GPU memory used by the backend is released as its process exits; other applications may still use GPU memory.")
        return 0
    print("At least one process could not be confirmed stopped. Runtime state was retained for safe follow-up.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
