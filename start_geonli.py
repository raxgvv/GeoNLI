#!/usr/bin/env python3
"""Start GeoNLI's Flask backend and Vite frontend together."""

from __future__ import annotations

import argparse
import errno
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid


ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "ui" / "frontend"
RUNTIME = Path("/tmp/geonli")
STATE = RUNTIME / "services.json"
LOCK = RUNTIME / "launcher.lock"
LOG_DIR = ROOT / "logs"
BACKEND_PORT = 8000
DEFAULT_FRONTEND_PORT = 5173
# Model initialization can take several minutes on first startup.
STARTUP_TIMEOUT = 600


def proc_start_ticks(pid: int) -> str | None:
    """Read Linux /proc start time, which protects against stale/reused PIDs."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        return stat[stat.rfind(")") + 2 :].split()[19]
    except (OSError, IndexError):
        return None


def group_has_running_process(pgid: int) -> bool:
    """A zombie cannot serve requests or respond to signals; ignore it here."""
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


def proc_cmdline(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return ""


def proc_has_launch_id(pid: int, launch_id: str | None) -> bool:
    if not launch_id:
        return False
    try:
        environment = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
        return f"GEONLI_LAUNCH_ID={launch_id}".encode() in environment
    except OSError:
        return False


def record_alive(record: dict) -> bool:
    pid = record.get("pid")
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return False
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat[stat.rfind(")") + 2 :].split()
        if fields[0] in {"Z", "X"} or fields[19] != str(record.get("start_ticks")):
            return False
    except (OSError, IndexError):
        return False
    # `conda run` may exec Python in place, changing argv while preserving the
    # unique environment marker assigned to this launcher.
    return proc_has_launch_id(pid, record.get("launch_id"))


def load_state() -> dict | None:
    try:
        return json.loads(STATE.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def check_status() -> int:
    state = load_state()
    if not state:
        print("GeoNLI launcher has no runtime state. Services may be stopped or were started another way.")
        return 1
    active = False
    for name in ("backend", "frontend"):
        record = state.get("services", {}).get(name, {})
        alive = record_alive(record)
        active |= alive
        url = record.get("url", "unknown URL")
        print(f"{name.capitalize()}: {'running' if alive else 'stopped'} ({url})")
    return 0 if active else 1


def occupied_port(port: int) -> tuple[bool, str]:
    try:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
            return False, ""
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise RuntimeError(f"Cannot check port {port}: {exc}") from exc
    socket_inodes = set()
    for table in (Path("/proc/net/tcp"), Path("/proc/net/tcp6")):
        try:
            rows = table.read_text().splitlines()[1:]
        except OSError:
            continue
        for row in rows:
            fields = row.split()
            if len(fields) > 9 and fields[1].rsplit(":", 1)[-1].upper() == f"{port:04X}" and fields[3] == "0A":
                socket_inodes.add(fields[9])
    owners = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or not socket_inodes:
            continue
        try:
            owns_socket = False
            for fd in (entry / "fd").iterdir():
                try:
                    target = os.readlink(fd)
                except OSError:
                    continue
                if target.startswith("socket:[") and target[8:-1] in socket_inodes:
                    owns_socket = True
                    break
        except OSError:
            continue
        if owns_socket:
            cmd = proc_cmdline(int(entry.name))
            owners.append(f"PID {entry.name}: {cmd or 'command unavailable'}")
    return True, "; ".join(owners) or "owner unavailable (port bind failed)"


def frontend_port() -> int:
    config = FRONTEND / "vite.config.js"
    if config.exists():
        match = re.search(r"\bport\s*:\s*(\d+)", config.read_text())
        if match:
            return int(match.group(1))
    return DEFAULT_FRONTEND_PORT


def terminate_group(record: dict, timeout: float = 8.0) -> None:
    """Terminate only a process group created in a new session by this launcher."""
    try:
        pid = int(record["pid"])
        pgid = int(record["pgid"])
    except (KeyError, TypeError, ValueError):
        return
    # The launcher is the session/group leader, and its saved start time and
    # command fingerprint must still match before signaling its process tree.
    try:
        group_matches = os.getpgid(pid) == pgid and os.getsid(pid) == int(record.get("sid", pgid))
    except (ProcessLookupError, PermissionError, ValueError):
        group_matches = False
    if proc_start_ticks(pid) != str(record.get("start_ticks")) or not group_matches:
        return
    if not proc_has_launch_id(pid, record.get("launch_id")):
        return
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and group_has_running_process(pgid):
        time.sleep(0.2)
    if not group_has_running_process(pgid):
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def cleanup(state: dict | None, processes: dict[str, subprocess.Popen] | None = None) -> None:
    if not state:
        return
    services = state.get("services", {})
    # Stop frontend first so it cannot continue issuing requests during backend shutdown.
    for name in ("frontend", "backend"):
        terminate_group(services.get(name, {}))
    for process in (processes or {}).values():
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            print("A launcher child is still running; runtime state retained for safe follow-up.")
            return
    for record in services.values():
        if record_alive(record):
            print(f"Could not verify that {record.get('name', 'a service')} stopped; runtime state retained.")
            return
    try:
        STATE.unlink(missing_ok=True)
        LOCK.unlink(missing_ok=True)
    except OSError as exc:
        print(f"Could not remove runtime state: {exc}")


def stream_output(name: str, pipe, log_path: Path) -> None:
    with log_path.open("a", encoding="utf-8", buffering=1) as log:
        for line in iter(pipe.readline, ""):
            log.write(line)
            print(f"[{name}] {line}", end="" if line.endswith("\n") else "\n", flush=True)
    pipe.close()


def wait_reachable(url: str, process: subprocess.Popen, name: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            print(f"{name} exited during startup (exit code {process.returncode}). See logs/ for details.")
            return False
        try:
            with urllib.request.urlopen(url, timeout=2) as response:
                if response.status < 500:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(1)
    print(f"Timed out waiting for {name} at {url} after {timeout}s. See logs/ for details.")
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--open", action="store_true", help="Open the frontend once it is ready")
    parser.add_argument("--status", action="store_true", help="Show launcher-managed service status")
    args = parser.parse_args()
    if args.status:
        return check_status()

    # Turn `kill <launcher-pid>` into the same verified child cleanup as Ctrl+C.
    def handle_termination(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, handle_termination)

    required_dirs = [FRONTEND, ROOT / "backend", ROOT / "models" / "captioning-model", ROOT / "models" / "vqa-model", ROOT / "models" / "sam3"]
    missing = [str(path) for path in required_dirs if not path.is_dir()]
    if missing:
        print("Required project directories are missing:\n  " + "\n  ".join(missing), file=sys.stderr)
        return 1
    conda = shutil.which("conda") or str(Path.home() / "miniconda" / "bin" / "conda")
    node, npm = shutil.which("node"), shutil.which("npm")
    for label, executable in (("Conda", conda), ("Node.js", node), ("npm", npm)):
        if not executable or not Path(executable).exists():
            print(f"{label} is unavailable. Install it or add it to PATH, then retry.", file=sys.stderr)
            return 1
    check = subprocess.run([conda, "run", "-n", "py310", "python", "--version"], capture_output=True, text=True)
    if check.returncode:
        detail = (check.stderr or check.stdout).strip()
        print("Conda environment 'py310' is unavailable. Create/fix it, then retry."
              + (f"\nConda reported: {detail}" if detail else ""), file=sys.stderr)
        return 1
    for port in (BACKEND_PORT, frontend_port()):
        try:
            occupied, owner = occupied_port(port)
        except RuntimeError as exc:
            print(f"{exc}. Run the launcher in a normal local terminal with permission to inspect/bind localhost ports.", file=sys.stderr)
            return 1
        if occupied:
            print(f"Port {port} is already occupied: {owner}\nStop that service yourself or configure a free port; no process was terminated.", file=sys.stderr)
            return 1

    RUNTIME.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    try:
        lock_fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        if load_state() and check_status() == 0:
            print("GeoNLI is already running. Use --status to inspect it or stop_geonli.py to stop it.")
            return 1
        print(f"Runtime lock {LOCK} exists. Check for another launcher before removing stale runtime files.", file=sys.stderr)
        return 1
    os.write(lock_fd, str(os.getpid()).encode())
    os.close(lock_fd)

    backend_env = os.environ.copy()
    backend_env.update({
        "CAPTIONING_MODEL_DIR": str(ROOT / "models" / "captioning-model"),
        "VQA_MODEL_DIR": str(ROOT / "models" / "vqa-model"),
        "SAM3_ROOT": str(ROOT / "models" / "sam3"),
        "PYTHONPATH": os.pathsep.join(filter(None, [str(ROOT), str(ROOT / "models" / "sam3"), str(ROOT / "models" / "SAR2Optical"), os.environ.get("PYTHONPATH", "")])),
        "BACKEND_HOST": "127.0.0.1",
        "BACKEND_PORT": str(BACKEND_PORT),
        "PYTHONUNBUFFERED": "1",
        "GEONLI_LOG": str(LOG_DIR / "backend-app.log"),
    })
    identity = f"geonli-launch-{uuid.uuid4().hex}"
    backend_cmd = [conda, "run", "--no-capture-output", "-n", "py310", "python", "-u", "-m", "backend.app"]
    frontend_cmd = [npm, "run", "dev", "--", "--host", "127.0.0.1", "--port", str(frontend_port()), "--strictPort"]
    # Unique environment marker makes the launched command fingerprint inspectable.
    backend_env["GEONLI_LAUNCH_ID"] = identity
    frontend_env = os.environ.copy()
    frontend_env["GEONLI_LAUNCH_ID"] = identity
    processes: dict[str, subprocess.Popen] = {}
    services: dict[str, dict] = {}
    state = {"launcher_pid": os.getpid(), "launch_id": identity, "services": services}
    try:
        for name, command, cwd, env, url, fingerprint, logname in (
            ("backend", backend_cmd, ROOT, backend_env, f"http://127.0.0.1:{BACKEND_PORT}/geoNLI/health", "conda run", "backend.log"),
            ("frontend", frontend_cmd, FRONTEND, frontend_env, f"http://127.0.0.1:{frontend_port()}/", "npm run dev", "frontend.log"),
        ):
            proc = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
            processes[name] = proc
            record = {"name": name, "pid": proc.pid, "pgid": os.getpgid(proc.pid), "sid": os.getsid(proc.pid), "start_ticks": proc_start_ticks(proc.pid), "identity": fingerprint, "launch_id": identity, "url": url, "log": str(LOG_DIR / logname)}
            services[name] = record
            STATE.write_text(json.dumps(state, indent=2))
            threading.Thread(target=stream_output, args=(name.upper(), proc.stdout, LOG_DIR / logname), daemon=True).start()
        print("Waiting for backend health and frontend HTTP readiness (model initialization may take several minutes)...")
        for name, url in (("backend", services["backend"]["url"]), ("frontend", services["frontend"]["url"])):
            if not wait_reachable(url, processes[name], name, STARTUP_TIMEOUT):
                raise RuntimeError(f"{name} did not become ready")
        print(f"Backend ready: {services['backend']['url']}")
        print(f"Frontend ready: {services['frontend']['url']}")
        print(f"Logs: {LOG_DIR}")
        if args.open:
            import webbrowser
            webbrowser.open(services["frontend"]["url"])
        print("Press Ctrl+C to stop both services.")
        while all(proc.poll() is None for proc in processes.values()):
            time.sleep(1)
        ended = [name for name, proc in processes.items() if proc.poll() is not None]
        if ended:
            print("Service exited: " + ", ".join(ended))
        return 1
    except KeyboardInterrupt:
        print("\nStopping GeoNLI services...")
        return 0
    except Exception as exc:
        print(f"Startup failed: {exc}", file=sys.stderr)
        return 1
    finally:
        cleanup(state, processes)


if __name__ == "__main__":
    raise SystemExit(main())
