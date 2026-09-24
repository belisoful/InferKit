#!/usr/bin/env python3
"""Exercises the LMC queue against an isolated state directory.

    python3 Tools/lmc/test_lmc.py
"""

import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lmc  # noqa: E402

ROOT_A = "/tree/a"
ROOT_B = "/tree/b"


class SleepingProcess:
    """A child whose pid stands in for a session; `kill()` makes the session 'die'."""

    def __init__(self):
        self.proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])

    @property
    def pid(self):
        return self.proc.pid

    def kill(self):
        self.proc.kill()
        self.proc.wait()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self.proc.poll() is None:
            self.kill()


class LMCTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name) / "lmc"
        self.legacy = Path(self.tmp.name) / "legacy-slot"
        os.environ["INFERKIT_LMC_LEGACY_SLOT"] = str(self.legacy)
        self.me = os.getpid()

    def tearDown(self):
        os.environ.pop("INFERKIT_LMC_LEGACY_SLOT", None)
        self.tmp.cleanup()

    def state(self):
        return lmc.State(self.dir)

    def request(self, tests, session, pid=None, root=ROOT_A, priority=50, fresh=False):
        with self.state() as st:
            req = st.request(tests, root, session, pid or self.me, priority, None, fresh)
        return req["id"]

    def req(self, rid):
        with self.state() as st:
            return dict(st.requests.get(rid) or {})

    def run_of(self, rid):
        with self.state() as st:
            return dict(st.runs[st.requests[rid]["run"]])

    def test_first_request_is_granted_and_second_waits(self):
        a = self.request(["sam3"], "s1")
        b = self.request(["deepseek"], "s2")
        self.assertEqual(self.req(a)["state"], "granted")
        self.assertEqual(self.req(b)["state"], "queued")
        with self.state() as st:
            st.release(a, "passed")
        self.assertEqual(self.req(b)["state"], "granted")

    def test_same_tests_same_tree_coalesce_into_one_run(self):
        a = self.request(["full-check"], "s1")
        b = self.request(["full_check"], "s2")
        self.assertEqual(self.req(a)["run"], self.req(b)["run"])
        self.assertEqual(self.req(a)["role"], "runner")
        self.assertEqual(self.req(b)["role"], "rider")
        self.assertEqual(self.req(b)["state"], "riding")

    def test_a_different_tree_does_not_coalesce(self):
        a = self.request(["full-check"], "s1", root=ROOT_A)
        b = self.request(["full-check"], "s2", root=ROOT_B)
        self.assertNotEqual(self.req(a)["run"], self.req(b)["run"])

    def test_a_passed_run_satisfies_its_riders(self):
        a = self.request(["full-check"], "s1")
        b = self.request(["full-check"], "s2")
        with self.state() as st:
            st.release(a, "passed", log="/logs/x.log")
        rider = self.req(b)
        self.assertEqual((rider["state"], rider["outcome"]), ("satisfied", "passed"))
        self.assertEqual(lmc._exit_for(rider), lmc.EXIT_SATISFIED_PASSED)

    def test_a_failed_run_satisfies_its_riders_with_the_failure(self):
        a = self.request(["full-check"], "s1")
        b = self.request(["full-check"], "s2")
        with self.state() as st:
            st.release(a, "failed")
        self.assertEqual(lmc._exit_for(self.req(b)), lmc.EXIT_SATISFIED_FAILED)

    def test_a_stopped_run_tells_its_riders_to_rerequest(self):
        a = self.request(["full-check"], "s1")
        b = self.request(["full-check"], "s2")
        c = self.request(["full-check"], "s3")
        with self.state() as st:
            st.release(a, "stopped")
        self.assertEqual(self.req(b)["state"], "rerequest")
        self.assertEqual(lmc._exit_for(self.req(c)), lmc.EXIT_REREQUEST)

    def test_full_check_is_always_lowest_priority(self):
        holder = self.request(["sam3"], "s0")
        full = self.request(["full-check"], "s1", priority=1)
        later = self.request(["deepseek"], "s2", priority=99)
        self.assertEqual(self.req(full)["priority"], lmc.FULL_CHECK_PRIORITY)
        with self.state() as st:
            st.release(holder, "passed")
        self.assertEqual(self.req(later)["state"], "granted")
        self.assertEqual(self.req(full)["state"], "queued")

    def test_lower_priority_number_runs_first_then_earliest(self):
        holder = self.request(["sam3"], "s0")
        late_urgent = self.request(["a"], "s1", priority=50)
        urgent = self.request(["b"], "s2", priority=10)
        also_urgent = self.request(["c"], "s3", priority=10)
        with self.state() as st:
            st.release(holder, "passed")
        self.assertEqual(self.req(urgent)["state"], "granted")
        self.assertEqual(self.req(also_urgent)["state"], "queued")
        self.assertEqual(self.req(late_urgent)["state"], "queued")

    def test_a_request_rides_a_run_already_in_progress_unless_fresh(self):
        a = self.request(["full-check"], "s1")
        self.assertEqual(self.req(a)["state"], "granted")
        b = self.request(["full-check"], "s2")
        self.assertEqual(self.req(a)["run"], self.req(b)["run"])
        self.assertEqual(self.req(b)["state"], "riding")
        c = self.request(["full-check"], "s3", fresh=True)
        self.assertNotEqual(self.req(a)["run"], self.req(c)["run"])
        self.assertEqual(self.req(c)["state"], "queued")

    def test_a_dead_holder_is_reaped_as_stopped(self):
        with SleepingProcess() as session:
            a = self.request(["full-check"], "s1", pid=session.pid)
            b = self.request(["full-check"], "s2")
            self.assertEqual(self.req(a)["state"], "granted")
            session.kill()
            self.assertEqual(self.req(b)["state"], "rerequest")
            self.assertEqual(self.run_of(b)["outcome"], "stopped")

    def test_a_dead_child_is_reaped_as_stopped(self):
        a = self.request(["sam3"], "s1")
        with SleepingProcess() as child:
            with self.state() as st:
                st.set_child(a, child.pid, "/logs/x.log")
            child.kill()
        self.assertEqual(self.req(a)["state"], "rerequest")

    def test_a_queued_request_whose_session_died_is_dropped(self):
        a = self.request(["sam3"], "s1")
        with SleepingProcess() as session:
            b = self.request(["deepseek"], "s2", pid=session.pid)
            session.kill()
            self.assertEqual(self.req(b)["state"], "cancelled")
        with self.state() as st:
            st.release(a, "passed")
        with self.state() as st:
            self.assertIsNone(st.running())

    def test_cancelling_the_runner_promotes_nobody_and_ends_the_run(self):
        a = self.request(["sam3"], "s1")
        b = self.request(["deepseek"], "s2")
        with self.state() as st:
            self.assertFalse(st.cancel(a))  # a granted runner releases; it does not cancel
            self.assertTrue(st.cancel(b))
        self.assertEqual(self.req(b)["state"], "cancelled")

    def test_the_legacy_slot_is_written_while_held_and_honored_while_alive(self):
        a = self.request(["sam3"], "s1")
        text = self.legacy.read_text().strip()
        self.assertTrue(text.startswith("session-s1 "))
        with self.state() as st:
            st.release(a, "passed")
        self.assertFalse(self.legacy.exists())
        with SleepingProcess() as other:
            self.legacy.write_text(f"session-old {other.pid} 12:00:00\n")
            b = self.request(["sam3"], "s2")
            self.assertEqual(self.req(b)["state"], "queued")
            other.kill()
        self.assertEqual(self.req(b)["state"], "granted")

    def test_acknowledged_outcomes_are_pruned(self):
        a = self.request(["full-check"], "s1")
        b = self.request(["full-check"], "s2")
        with self.state() as st:
            st.release(a, "passed")
        code, req = lmc.wait_for(b, timeout=1, poll=0.1, state_dir_path=self.dir)
        self.assertEqual(code, lmc.EXIT_SATISFIED_PASSED)
        self.assertEqual(self.req(b), {})

    def test_wait_times_out_and_keeps_the_request(self):
        self.request(["sam3"], "s1")
        b = self.request(["deepseek"], "s2")
        code, _ = lmc.wait_for(b, timeout=0.3, poll=0.1, state_dir_path=self.dir)
        self.assertEqual(code, lmc.EXIT_TIMEOUT)
        self.assertEqual(self.req(b)["state"], "queued")

    def test_run_command_end_to_end(self):
        script = str(Path(__file__).resolve().parent / "lmc.py")
        env = dict(os.environ, INFERKIT_LMC_DIR=str(self.dir), CLAUDE_PID=str(self.me))
        out = subprocess.run([sys.executable, script, "run", "--test", "smoke", "--session", "s1",
                              "--silent", "--", "sh", "-c", "echo hello; exit 3"],
                             capture_output=True, text=True, env=env)
        self.assertEqual(out.returncode, 3, out.stdout + out.stderr)
        self.assertIn("outcome=failed", out.stdout)
        log = [line for line in out.stdout.splitlines() if "log=" in line][-1].split("log=")[-1]
        self.assertEqual(Path(log).read_text().strip(), "hello")
        with self.state() as st:
            self.assertIsNone(st.running())
        status = subprocess.run([sys.executable, script, "status"], capture_output=True, text=True, env=env)
        self.assertEqual(status.stdout.splitlines()[0], "FREE")

    def test_run_stopped_by_signal_releases_as_stopped(self):
        script = str(Path(__file__).resolve().parent / "lmc.py")
        env = dict(os.environ, INFERKIT_LMC_DIR=str(self.dir), CLAUDE_PID=str(self.me))
        proc = subprocess.Popen([sys.executable, script, "run", "--test", "full-check", "--session", "s1",
                                 "--root", ROOT_A, "--silent", "--", "sleep", "60"], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        deadline = time.time() + 10
        while time.time() < deadline:
            with self.state() as st:
                run = st.running()
                if run is not None and run["child_pid"]:
                    break
            time.sleep(0.1)
        rider = self.request(["full-check"], "s2")
        proc.terminate()
        out, _ = proc.communicate(timeout=10)
        self.assertIn("outcome=stopped", out)
        self.assertEqual(self.req(rider)["state"], "rerequest")


if __name__ == "__main__":
    unittest.main()
