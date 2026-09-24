#!/usr/bin/env python3
"""Large Model Coordination (LMC): one large-model test run on the machine at a time.

Several agent sessions share one Mac. Two multi-gigabyte test runs at once thrash memory and turn
both into upper bounds, so a session that wants a large-model test asks the LMC for the lock, runs
only once it holds it, and releases the lock as the first thing it does when the run ends. The LMC
is this script over a small state directory; there is no daemon. Every command takes the state
lock, reaps dead holders, schedules, and returns.

    Tools/lmc/lmc.py request --test full-check           # enqueue; prints the request id and state
    Tools/lmc/lmc.py wait <id> [--timeout 540]            # block until granted, satisfied, or told to re-request
    Tools/lmc/lmc.py acquire --test <name>                # request + wait in one call
    Tools/lmc/lmc.py run --test <name> -- <command...>    # acquire, run the command, release with its outcome
    Tools/lmc/lmc.py release <id> [--outcome passed|failed|stopped]
    Tools/lmc/lmc.py cancel <id>                          # withdraw a queued request
    Tools/lmc/lmc.py status [--json]
    Tools/lmc/lmc.py history [-n 30]

Tests. A test is a name the session chooses (`full-check`, `deepseek-v41-released`, `sam3-parity`).
A request names one or more; the request holds the lock for all of them, and only a request with
the same set from the same working tree combines with it.

Coalescing. Requests for the same test set from the same working tree form one run, whether that
run is still queued or already in progress. The first request is the runner; later ones ride. When
the runner releases with `passed` or `failed`, every rider is satisfied and its `wait` returns that
outcome with the runner's log path. When the run is `stopped` (a signal, a dead holder, an explicit
`--outcome stopped`), every rider is told to re-request. A run covers the tree as it stood at its
start, so a session whose edits landed after that start passes `--fresh` to queue its own run
instead of riding.

Priority. `--priority N` (1 most urgent, 99 least, default 50). `full-check` is always 100, the
lowest, whatever is passed. The scheduler grants the pending run with the lowest number, then the
earliest request.

Liveness. A request records its session's process id (`CLAUDE_PID`, or `--pid`). A holder whose
process is gone is reaped as `stopped`; a queued request whose session is gone is dropped. `run`
also records the child it spawned. Another process reaps the run as `stopped` when that child is
gone; `run` itself waits on the child and releases with the command's exit status. The legacy
advisory file `~/.claude/inferkit-test-slot-mlx` is written while the lock is held and honored while
its holder is alive, so scripts that still read it stay coordinated.

Exit codes for `wait`, `acquire`, and a `run` that did not run its command:
    0  granted: this session holds the lock; run, then release
   10  satisfied: another session ran the same test and it passed
   11  satisfied: another session ran the same test and it failed (read its log)
   12  re-request: the run this request rode on was stopped mid-run
   13  timeout: the request is still queued; call `wait` again
   14  cancelled: the request was withdrawn or its session died
`run` exits with its command's own status once the command has run.

State lives in `~/.claude/inferkit-lmc/` (`INFERKIT_LMC_DIR` overrides it). `state.json` is the
queue; `history.log` is one line per event; `logs/` holds the output of every `run`.
"""

import argparse
import contextlib
import datetime as _dt
import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

STATE_VERSION = 1
FULL_CHECK = "full-check"
FULL_CHECK_PRIORITY = 100
DEFAULT_PRIORITY = 50
FINISHED_RETENTION = 24 * 3600

EXIT_GRANTED = 0
EXIT_SATISFIED_PASSED = 10
EXIT_SATISFIED_FAILED = 11
EXIT_REREQUEST = 12
EXIT_TIMEOUT = 13
EXIT_CANCELLED = 14
EXIT_USAGE = 2

TERMINAL_OUTCOME_EXIT = {"passed": EXIT_SATISFIED_PASSED, "failed": EXIT_SATISFIED_FAILED}


# ----------------------------------------------------------------------------------------------
# Environment


def state_dir():
    return Path(os.environ.get("INFERKIT_LMC_DIR") or Path.home() / ".claude" / "inferkit-lmc")


def legacy_slot_path():
    override = os.environ.get("INFERKIT_LMC_LEGACY_SLOT")
    if override is not None:
        return Path(override) if override else None
    return Path.home() / ".claude" / "inferkit-test-slot-mlx"


def default_session():
    for key in ("INFERKIT_LMC_SESSION", "CLAUDE_CODE_SESSION_ID"):
        value = os.environ.get(key)
        if value:
            return value[:8]
    return f"{os.environ.get('USER', 'user')}@{os.uname().nodename.split('.')[0]}"


def default_pid():
    value = os.environ.get("CLAUDE_PID")
    if value and value.isdigit():
        return int(value)
    return os.getppid()


def working_tree(path=None):
    start = Path(path or os.getcwd()).resolve()
    try:
        out = subprocess.run(["git", "-C", str(start), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, check=False)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except OSError:
        pass
    return str(start)


def pid_alive(pid):
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except (ValueError, OverflowError):
        return False
    return True


def now():
    return time.time()


def stamp(seconds=None):
    return _dt.datetime.fromtimestamp(seconds if seconds is not None else now()).strftime("%Y-%m-%d %H:%M:%S")


def elapsed(seconds):
    seconds = int(max(0, seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def normalize_test(name):
    name = re.sub(r"[\s_]+", "-", name.strip().lower())
    if name in ("fullcheck", "full-check", "full-checks"):
        return FULL_CHECK
    return name


def coalescing_key(tests, root):
    return "|".join(sorted(tests)) + "@" + root


# ----------------------------------------------------------------------------------------------
# State


class State:
    """The queue on disk, edited under one advisory lock.

    `with State() as st:` holds the lock for the block; every read-modify-write of the queue runs
    inside one, including the reap and the scheduling pass that follow every command.
    """

    def __init__(self, directory=None):
        self.dir = Path(directory or state_dir())
        self.path = self.dir / "state.json"
        self.lock_path = self.dir / "lock"
        self.history_path = self.dir / "history.log"
        self.data = None
        self._lock_file = None

    def __enter__(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "logs").mkdir(exist_ok=True)
        self._lock_file = open(self.lock_path, "a+")
        fcntl.flock(self._lock_file, fcntl.LOCK_EX)
        self.data = self._load()
        self.reap()
        self.schedule()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if exc_type is None:
                self.reap()
                self.schedule()
                self._save()
        finally:
            fcntl.flock(self._lock_file, fcntl.LOCK_UN)
            self._lock_file.close()
            self._lock_file = None
        return False

    def _load(self):
        if not self.path.exists():
            return {"version": STATE_VERSION, "requests": {}, "runs": {}, "legacy_written": None}
        with open(self.path) as handle:
            data = json.load(handle)
        data.setdefault("requests", {})
        data.setdefault("runs", {})
        data.setdefault("legacy_written", None)
        return data

    def _save(self):
        tmp = self.path.with_suffix(".json.tmp")
        with open(tmp, "w") as handle:
            json.dump(self.data, handle, indent=2, sort_keys=True)
        os.replace(tmp, self.path)

    def log(self, event, **fields):
        line = f"{stamp()} {event}"
        for key, value in fields.items():
            if value is not None:
                line += f" {key}={value}"
        with open(self.history_path, "a") as handle:
            handle.write(line + "\n")

    # -- queries

    @property
    def requests(self):
        return self.data["requests"]

    @property
    def runs(self):
        return self.data["runs"]

    def running(self):
        for run in self.runs.values():
            if run["state"] == "running":
                return run
        return None

    def run_requests(self, run):
        return [self.requests[rid] for rid in run["requests"] if rid in self.requests]

    # -- legacy slot

    def legacy_holder(self):
        """The live holder of the legacy slot file when it is not ours, else None."""
        path = legacy_slot_path()
        if path is None or not path.exists():
            return None
        try:
            text = path.read_text().strip()
        except OSError:
            return None
        if text == self.data.get("legacy_written"):
            return None
        parts = text.split()
        pid = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
        if pid_alive(pid):
            return {"text": text, "pid": pid, "session": parts[0] if parts else "?"}
        return None

    def legacy_write(self, run):
        path = legacy_slot_path()
        if path is None:
            return
        text = f"session-{run['runner_session']} {run['holder_pid']} {stamp(run['started_at'])[11:]}"
        try:
            path.write_text(text + "\n")
            self.data["legacy_written"] = text
        except OSError:
            pass

    def legacy_clear(self):
        path = legacy_slot_path()
        written = self.data.get("legacy_written")
        if path is None or not written:
            return
        try:
            if path.exists() and path.read_text().strip() == written:
                path.unlink()
        except OSError:
            pass
        self.data["legacy_written"] = None

    # -- mutations

    def request(self, tests, root, session, pid, priority, note, fresh=False):
        tests = sorted({normalize_test(t) for t in tests if t.strip()})
        if not tests:
            raise ValueError("a request names at least one test")
        if FULL_CHECK in tests:
            priority = FULL_CHECK_PRIORITY
        else:
            priority = max(1, min(99, priority))
        key = coalescing_key(tests, root)
        rid = uuid.uuid4().hex[:8]
        req = {
            "id": rid, "session": session, "pid": pid, "tests": tests, "root": root, "key": key,
            "priority": priority, "note": note, "requested_at": now(), "state": "queued",
            "run": None, "role": None, "outcome": None, "acknowledged": False,
        }
        run = None if fresh else self._joinable_run(key)
        if run is None:
            run = {
                "id": uuid.uuid4().hex[:8], "key": key, "tests": tests, "root": root,
                "priority": priority, "state": "pending", "requests": [], "runner": None,
                "runner_session": None, "holder_pid": None, "child_pid": None,
                "supervisor_pid": None, "log": None,
                "created_at": now(), "started_at": None, "ended_at": None, "outcome": None,
            }
            self.runs[run["id"]] = run
        run["priority"] = min(run["priority"], priority) if FULL_CHECK not in tests else FULL_CHECK_PRIORITY
        run["requests"].append(rid)
        req["run"] = run["id"]
        req["role"] = "rider" if run["requests"][:-1] else "runner"
        if run["state"] == "running":
            req["state"] = "riding"
        self.requests[rid] = req
        self.log("request", id=rid, session=session, tests=",".join(tests), run=run["id"],
                 role=req["role"], priority=priority, root=root)
        return req

    def _joinable_run(self, key):
        for run in self.runs.values():
            if run["key"] == key and run["state"] in ("pending", "running"):
                return run
        return None

    def cancel(self, rid, reason="cancelled"):
        req = self.requests.get(rid)
        if req is None or req["state"] not in ("queued", "riding"):
            return False
        run = self.runs.get(req["run"])
        req["state"] = "cancelled"
        req["outcome"] = reason
        if run is not None:
            run["requests"] = [r for r in run["requests"] if r != rid]
            if run["state"] == "pending" and not run["requests"]:
                del self.runs[run["id"]]
            elif run["state"] == "running" and run["runner"] == rid:
                self.end_run(run, "stopped", reason=f"runner {reason}")
        self.log("cancel", id=rid, session=req["session"], reason=reason)
        return True

    def release(self, rid, outcome, log=None, reason=None):
        req = self.requests.get(rid)
        if req is None:
            return None
        run = self.runs.get(req["run"])
        if run is None or run["state"] != "running" or run["runner"] != rid:
            return None
        if log:
            run["log"] = log
        self.end_run(run, outcome, reason=reason)
        return run

    def release_session(self, session, outcome, log=None):
        for run in self.runs.values():
            if run["state"] == "running" and run["runner_session"] == session:
                return self.release(run["runner"], outcome, log=log)
        return None

    def end_run(self, run, outcome, reason=None):
        run["state"] = outcome
        run["outcome"] = outcome
        run["ended_at"] = now()
        for req in self.run_requests(run):
            if req["state"] not in ("queued", "riding", "granted"):
                continue
            if outcome == "stopped":
                req["state"] = "rerequest"
            else:
                req["state"] = "satisfied"
            req["outcome"] = outcome
        self.legacy_clear()
        self.log("end", run=run["id"], outcome=outcome, runner=run["runner"],
                 session=run["runner_session"], riders=len(run["requests"]) - 1, reason=reason,
                 log=run["log"], took=elapsed(run["ended_at"] - (run["started_at"] or run["ended_at"])))

    def reap(self):
        """Drops requests whose sessions are gone and ends a run whose holder is gone."""
        for req in list(self.requests.values()):
            if req["state"] in ("queued", "riding") and not pid_alive(req["pid"]):
                self.cancel(req["id"], reason="session-died")
        run = self.running()
        if run is not None:
            holder_alive = pid_alive(run["holder_pid"])
            # The supervisor reaps its own child and reports the exit through `release`.
            supervising = run.get("supervisor_pid") == os.getpid()
            child_alive = run["child_pid"] is None or supervising or pid_alive(run["child_pid"])
            if not holder_alive or not child_alive:
                which = "holder" if not holder_alive else "child"
                self.end_run(run, "stopped", reason=f"{which}-process-gone")
        cutoff = now() - FINISHED_RETENTION
        for rid, req in list(self.requests.items()):
            terminal = req["state"] in ("satisfied", "rerequest", "cancelled", "released")
            if terminal and (req["acknowledged"] or req["requested_at"] < cutoff):
                del self.requests[rid]
        for run_id, run in list(self.runs.items()):
            finished = run["state"] not in ("pending", "running")
            if finished and (run["ended_at"] or 0) < cutoff:
                del self.runs[run_id]

    def schedule(self):
        if self.running() is not None or self.legacy_holder() is not None:
            return None
        pending = [r for r in self.runs.values() if r["state"] == "pending" and r["requests"]]
        if not pending:
            return None
        pending.sort(key=lambda r: (r["priority"], min([self.requests[q]["requested_at"]
                                                        for q in r["requests"] if q in self.requests]
                                                       or [r["created_at"]])))
        run = pending[0]
        runner = self.requests[run["requests"][0]]
        run["state"] = "running"
        run["runner"] = runner["id"]
        run["runner_session"] = runner["session"]
        run["holder_pid"] = runner["pid"]
        run["started_at"] = now()
        runner["state"] = "granted"
        runner["role"] = "runner"
        for rid in run["requests"][1:]:
            self.requests[rid]["state"] = "riding"
            self.requests[rid]["role"] = "rider"
        self.legacy_write(run)
        self.log("grant", run=run["id"], id=runner["id"], session=runner["session"],
                 tests=",".join(run["tests"]), riders=len(run["requests"]) - 1)
        return run

    def set_child(self, rid, child_pid, log, supervisor_pid=None):
        req = self.requests.get(rid)
        run = self.runs.get(req["run"]) if req else None
        if run is not None and run["state"] == "running":
            run["child_pid"] = child_pid
            run["supervisor_pid"] = supervisor_pid
            run["log"] = log

    def acknowledge(self, rid):
        req = self.requests.get(rid)
        if req is not None:
            req["acknowledged"] = True


# ----------------------------------------------------------------------------------------------
# Presentation


def describe_request(req, st):
    run = st.runs.get(req["run"]) if req else None
    parts = [f"id={req['id']}", f"state={req['state']}", f"role={req['role']}",
             f"tests={','.join(req['tests'])}", f"session={req['session']}",
             f"priority={req['priority']}"]
    if run is not None:
        parts.append(f"run={run['id']}")
        if run.get("log"):
            parts.append(f"log={run['log']}")
    if req.get("outcome"):
        parts.append(f"outcome={req['outcome']}")
    return " ".join(parts)


def status_lines(st):
    lines = []
    legacy = st.legacy_holder()
    run = st.running()
    if run is not None:
        lines.append(f"HOLDING  run={run['id']} tests={','.join(run['tests'])} "
                     f"session={run['runner_session']} pid={run['holder_pid']} "
                     f"for {elapsed(now() - run['started_at'])} riders={len(run['requests']) - 1}"
                     + (f" log={run['log']}" if run.get("log") else "")
                     + f" root={run['root']}")
    elif legacy is not None:
        lines.append(f"HOLDING  legacy slot {legacy['text']!r} (pid {legacy['pid']} alive)")
    else:
        lines.append("FREE")
    pending = [r for r in st.runs.values() if r["state"] == "pending"]
    pending.sort(key=lambda r: (r["priority"], r["created_at"]))
    for position, prun in enumerate(pending, 1):
        sessions = ",".join(st.requests[q]["session"] for q in prun["requests"] if q in st.requests)
        lines.append(f"QUEUE {position:>2}  run={prun['id']} tests={','.join(prun['tests'])} "
                     f"priority={prun['priority']} sessions={sessions} "
                     f"waiting {elapsed(now() - prun['created_at'])} root={prun['root']}")
    for req in st.requests.values():
        if req["state"] in ("satisfied", "rerequest") and not req["acknowledged"]:
            lines.append(f"DONE     {describe_request(req, st)}")
    return lines


def status_json(st):
    return {"holding": st.running(), "legacy": st.legacy_holder(), "runs": st.runs,
            "requests": st.requests}


# ----------------------------------------------------------------------------------------------
# Commands


def _session_args(args):
    session = args.session or default_session()
    pid = args.pid or default_pid()
    root = working_tree(args.root)
    return session, pid, root


def cmd_request(args):
    session, pid, root = _session_args(args)
    with State(args.state_dir) as st:
        req = st.request(args.test, root, session, pid, args.priority, args.note, args.fresh)
    print(describe_request(req, st))
    return 0


def wait_for(rid, timeout, poll, state_dir_path, quiet=True):
    """Polls the request until it leaves the queue. Returns (exit_code, request)."""
    deadline = now() + timeout if timeout else None
    last = None
    while True:
        with State(state_dir_path) as st:
            req = st.requests.get(rid)
            if req is None:
                return EXIT_CANCELLED, None
            code = _exit_for(req)
            if code is not None and code != EXIT_TIMEOUT:
                if code != EXIT_GRANTED:
                    st.acknowledge(rid)
                return code, dict(req, run_log=(st.runs.get(req["run"]) or {}).get("log"))
            summary = status_lines(st)[0]
        if not quiet and summary != last:
            print(f"waiting  {describe_request(req, st)} | {summary}", flush=True)
            last = summary
        if deadline is not None and now() >= deadline:
            return EXIT_TIMEOUT, req
        time.sleep(poll)


def _exit_for(req):
    state = req["state"]
    if state == "granted":
        return EXIT_GRANTED
    if state == "satisfied":
        return TERMINAL_OUTCOME_EXIT.get(req["outcome"], EXIT_SATISFIED_FAILED)
    if state == "rerequest":
        return EXIT_REREQUEST
    if state == "cancelled":
        return EXIT_CANCELLED
    if state == "released":
        return EXIT_SATISFIED_PASSED
    return EXIT_TIMEOUT


def report_wait(code, req):
    messages = {
        EXIT_GRANTED: "granted: you hold the lock; release it first thing when the run ends",
        EXIT_SATISFIED_PASSED: "satisfied: another session's run of the same test passed",
        EXIT_SATISFIED_FAILED: "satisfied: another session's run of the same test failed; read its log",
        EXIT_REREQUEST: "re-request: the run this request rode on was stopped mid-run",
        EXIT_TIMEOUT: "timeout: still queued; call wait again with the same id",
        EXIT_CANCELLED: "cancelled: the request was withdrawn or its session died",
    }
    line = f"LMC {messages[code]}"
    if req is not None:
        line += f" | id={req['id']} tests={','.join(req['tests'])}"
        if req.get("run_log"):
            line += f" log={req['run_log']}"
    print(line, flush=True)


def cmd_wait(args):
    code, req = wait_for(args.id, args.timeout, args.poll, args.state_dir, quiet=False)
    report_wait(code, req)
    return code


def cmd_acquire(args):
    session, pid, root = _session_args(args)
    with State(args.state_dir) as st:
        req = st.request(args.test, root, session, pid, args.priority, args.note, args.fresh)
    print(describe_request(req, st), flush=True)
    code, req = wait_for(req["id"], args.timeout, args.poll, args.state_dir, quiet=False)
    report_wait(code, req)
    return code


def cmd_release(args):
    session = args.session or default_session()
    with State(args.state_dir) as st:
        if args.id:
            run = st.release(args.id, args.outcome, log=args.log, reason="release")
        else:
            run = st.release_session(session, args.outcome, log=args.log)
        if run is not None:
            req = st.requests.get(run["runner"])
            if req is not None:
                req["state"] = "released"
                st.acknowledge(req["id"])
    if run is None:
        print("LMC nothing to release: no running run under that id or session", file=sys.stderr)
        return 1
    print(f"LMC released run={run['id']} outcome={args.outcome} riders_notified={len(run['requests']) - 1}")
    return 0


def cmd_cancel(args):
    with State(args.state_dir) as st:
        ok = st.cancel(args.id)
    print("LMC cancelled" if ok else "LMC nothing to cancel: not a queued request")
    return 0 if ok else 1


def cmd_status(args):
    with State(args.state_dir) as st:
        if args.json:
            print(json.dumps(status_json(st), indent=2, sort_keys=True))
        else:
            for line in status_lines(st):
                print(line)
    return 0


def cmd_history(args):
    path = State(args.state_dir).history_path
    if not path.exists():
        return 0
    lines = path.read_text().splitlines()
    for line in lines[-args.lines:]:
        print(line)
    return 0


def memory_free_percent():
    try:
        out = subprocess.run(["memory_pressure"], capture_output=True, text=True, check=False).stdout
    except OSError:
        return None
    match = re.search(r"free percentage:\s*(\d+)%", out)
    return int(match.group(1)) if match else None


def wait_for_quiet(free_percent, seconds, names, poll=5):
    """Waits until memory is free enough and no named process runs for `seconds` in a row."""
    quiet_since = None
    while True:
        free = memory_free_percent()
        busy = subprocess.run(["pgrep", "-x", *names], capture_output=True, text=True,
                              check=False).stdout.split()
        # `pgrep -x` matches the process NAME only; matching command text would match this script.
        quiet = (free is None or free >= free_percent) and not busy
        if quiet:
            quiet_since = quiet_since or now()
            if now() - quiet_since >= seconds:
                return
        else:
            quiet_since = None
        time.sleep(poll)


def cmd_run(args):
    if not args.command:
        print("run: give the command after `--`", file=sys.stderr)
        return EXIT_USAGE
    session, pid, root = _session_args(args)
    with State(args.state_dir) as st:
        req = st.request(args.test, root, session, pid, args.priority, args.note, args.fresh)
        log = args.log or str(st.dir / "logs" / f"{stamp().replace(' ', 'T').replace(':', '')}-{req['run']}.log")
    print(describe_request(req, st), flush=True)
    code, got = wait_for(req["id"], args.timeout, args.poll, args.state_dir, quiet=False)
    if code != EXIT_GRANTED:
        report_wait(code, got)
        return code
    if args.quiet_seconds:
        print(f"LMC granted; waiting for a quiet machine (free >= {args.quiet_free}%, "
              f"none of {','.join(args.quiet_names)} for {args.quiet_seconds}s)", flush=True)
        wait_for_quiet(args.quiet_free, args.quiet_seconds, args.quiet_names)
    print(f"LMC granted; running: {' '.join(args.command)} | log={log}", flush=True)
    outcome = "stopped"
    status = 1
    spawned = []

    def forward(signum, _frame):
        for child in spawned:
            with contextlib.suppress(ProcessLookupError):
                child.send_signal(signum)

    previous = {s: signal.signal(s, forward) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    with open(log, "ab") as handle, contextlib.suppress(KeyboardInterrupt):
        try:
            child = subprocess.Popen(args.command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except OSError as error:
            with State(args.state_dir) as st:
                st.release(req["id"], "stopped", log=log, reason=f"spawn-failed: {error}")
                st.acknowledge(req["id"])
            print(f"LMC released outcome=stopped: could not start the command: {error}", file=sys.stderr)
            return 127
        spawned.append(child)
        with State(args.state_dir) as st:
            st.set_child(req["id"], child.pid, log, supervisor_pid=os.getpid())
        try:
            for chunk in iter(lambda: child.stdout.read(4096), b""):
                handle.write(chunk)
                handle.flush()
                if not args.silent:
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
            status = child.wait()
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
        if status >= 0:
            outcome = "passed" if status == 0 else "failed"
    with State(args.state_dir) as st:
        run = st.release(req["id"], outcome, log=log, reason=f"exit={status}")
        st.acknowledge(req["id"])
    riders = len(run["requests"]) - 1 if run else 0
    print(f"LMC released outcome={outcome} exit={status} riders_notified={riders} log={log}", flush=True)
    return status if status >= 0 else 128 - status


# ----------------------------------------------------------------------------------------------
# CLI


def build_parser():
    parser = argparse.ArgumentParser(prog="lmc", description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--state-dir", default=None, help="override INFERKIT_LMC_DIR")
    sub = parser.add_subparsers(dest="command_name", required=True)

    def identity(p):
        p.add_argument("--session", help="session name (default: CLAUDE_CODE_SESSION_ID, 8 chars)")
        p.add_argument("--pid", type=int, help="the session's long-lived process (default: CLAUDE_PID)")
        p.add_argument("--root", help="working tree the run covers (default: git toplevel of cwd)")

    def requesting(p):
        identity(p)
        p.add_argument("--test", action="append", required=True, help="a test name; repeatable")
        p.add_argument("--priority", type=int, default=DEFAULT_PRIORITY,
                       help=f"1 most urgent, 99 least (default {DEFAULT_PRIORITY}; full-check is always 100)")
        p.add_argument("--note", help="free text shown in status")
        p.add_argument("--fresh", action="store_true",
                       help="queue a run of your own instead of riding one already in progress")

    def waiting(p):
        p.add_argument("--timeout", type=float, default=0, help="seconds; 0 waits forever")
        p.add_argument("--poll", type=float, default=5, help="seconds between checks")

    p = sub.add_parser("request", help="enqueue a lock request")
    requesting(p)
    p.set_defaults(func=cmd_request)

    p = sub.add_parser("wait", help="block until a request is granted, satisfied, or must be re-requested")
    p.add_argument("id")
    waiting(p)
    p.set_defaults(func=cmd_wait)

    p = sub.add_parser("acquire", help="request and wait")
    requesting(p)
    waiting(p)
    p.set_defaults(func=cmd_acquire)

    p = sub.add_parser("run", help="acquire, run a command, release with its outcome")
    requesting(p)
    waiting(p)
    p.add_argument("--log", help="where to write the command's output (default: the LMC logs directory)")
    p.add_argument("--silent", action="store_true", help="write output to the log only")
    p.add_argument("--quiet-seconds", type=int, default=0,
                   help="after the grant, wait for this many consecutive quiet seconds before running")
    p.add_argument("--quiet-free", type=int, default=70, help="memory_pressure free%% that counts as quiet")
    p.add_argument("--quiet-names", nargs="+", default=["xctest", "swift-test", "swift-build"],
                   help="process names (pgrep -x) that keep the machine busy")
    p.add_argument("command", nargs=argparse.REMAINDER, help="-- command to run")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("release", help="release the lock a request holds")
    p.add_argument("id", nargs="?", help="the runner's request id (default: this session's running run)")
    p.add_argument("--session")
    p.add_argument("--outcome", choices=["passed", "failed", "stopped"], default="passed")
    p.add_argument("--log", help="record where the run's output went, for the riders")
    p.set_defaults(func=cmd_release)

    p = sub.add_parser("cancel", help="withdraw a queued request")
    p.add_argument("id")
    p.set_defaults(func=cmd_cancel)

    p = sub.add_parser("status", help="show the holder, the queue, and unacknowledged outcomes")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("history", help="tail the event log")
    p.add_argument("-n", "--lines", type=int, default=30)
    p.set_defaults(func=cmd_history)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "command", None) and args.command and args.command[0] == "--":
        args.command = args.command[1:]
    try:
        return args.func(args)
    except ValueError as error:
        print(f"lmc: {error}", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
