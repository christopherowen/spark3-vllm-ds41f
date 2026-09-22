"""Host-independent behavioral tests for the fail-closed memory guard."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MEMGUARD = ROOT / "scripts" / "memguard.sh"
MEMGUARD_START = ROOT / "scripts" / "memguard-start.sh"


class MemguardTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.state = self.root / "state"
        self.actions = self.root / "actions"
        self.ready = self.root / "ready"
        self._write_command(
            "docker",
            """#!/usr/bin/env bash
set -eu
case "$1" in
  inspect)
    state=$(cat "$FAKE_STATE")
    if [[ "$state" == absent ]]; then
      exit 1
    fi
    printf '%s\n' "$state"
    ;;
  kill)
    printf 'kill %s\n' "$*" >>"$FAKE_ACTIONS"
    printf 'false\n' >"$FAKE_STATE"
    ;;
  *)
    exit 2
    ;;
esac
""",
        )
        self._write_command(
            "awk",
            """#!/usr/bin/env bash
set -eu
printf '%s\n' "$FAKE_MEM_AVAILABLE_KIB"
""",
        )
        self._write_command(
            "logger",
            """#!/usr/bin/env bash
set -eu
printf 'logger %s\n' "$*" >>"$FAKE_ACTIONS"
""",
        )
        self._write_command(
            "sleep",
            """#!/usr/bin/env bash
set -eu
state=$(cat "$FAKE_STATE")
case "$state" in
  absent) printf 'true\n' >"$FAKE_STATE" ;;
  true) printf 'false\n' >"$FAKE_STATE" ;;
esac
""",
        )
        self._write_command(
            "sudo",
            """#!/usr/bin/env bash
set -eu
printf 'sudo %s\n' "$*" >>"$FAKE_ACTIONS"
if [[ "$*" == *" systemd-run "* ]]; then
  for argument in "$@"; do
    case "$argument" in
      --setenv=READY_FILE=*)
        ready=${argument#--setenv=READY_FILE=}
        : >"$ready"
        ;;
    esac
  done
fi
""",
        )

    def _write_command(self, name: str, body: str) -> None:
        command = self.bin / name
        command.write_text(body, encoding="utf-8")
        command.chmod(0o755)

    def _run(
        self, *, state: str, available_gib: int, wait_seconds: int
    ) -> subprocess.CompletedProcess[str]:
        self.state.write_text(f"{state}\n", encoding="utf-8")
        environment = os.environ.copy()
        environment.update(
            {
                "PATH": f"{self.bin}:{environment['PATH']}",
                "CONTAINER_NAME": "candidate",
                "THRESHOLD_GIB": "5",
                "INTERVAL": "0.25",
                "MEMGUARD_PHASE": "startup",
                "WAIT_FOR_CONTAINER_SECONDS": str(wait_seconds),
                "READY_FILE": str(self.ready),
                "FAKE_STATE": str(self.state),
                "FAKE_ACTIONS": str(self.actions),
                "FAKE_MEM_AVAILABLE_KIB": str(available_gib * 1024 * 1024),
            }
        )
        return subprocess.run(
            [str(MEMGUARD)],
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )

    def test_prelaunch_guard_is_ready_before_container_and_tracks_its_lifetime(
        self,
    ) -> None:
        result = self._run(state="absent", available_gib=16, wait_seconds=120)

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(self.ready.exists())
        self.assertFalse(self.actions.exists())

    def test_running_container_is_killed_immediately_below_threshold(self) -> None:
        result = self._run(state="true", available_gib=4, wait_seconds=0)

        self.assertEqual(result.returncode, 1)
        self.assertFalse(self.ready.exists())
        self.assertIn("kill kill --signal KILL candidate", self.actions.read_text())

    def test_prelaunch_guard_rejects_low_memory_without_starting_container(
        self,
    ) -> None:
        result = self._run(state="absent", available_gib=4, wait_seconds=120)

        self.assertEqual(result.returncode, 1)
        self.assertFalse(self.ready.exists())
        self.assertNotIn("kill --signal", self.actions.read_text())

    def test_start_wrapper_confirms_protected_prelaunch_unit(self) -> None:
        environment = os.environ.copy()
        environment.update(
            {
                "PATH": f"{self.bin}:{environment['PATH']}",
                "XDG_RUNTIME_DIR": str(self.root),
                "FAKE_ACTIONS": str(self.actions),
            }
        )

        result = subprocess.run(
            [
                str(MEMGUARD_START),
                "config/cluster.json",
                "startup",
                "prelaunch",
            ],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("mode prelaunch", result.stdout)
        actions = self.actions.read_text(encoding="utf-8")
        self.assertIn("--property=OOMScoreAdjust=-1000", actions)
        self.assertIn("--property=MemoryMin=64M", actions)
        self.assertIn("--property=CPUWeight=1000", actions)
        self.assertIn("--setenv=INTERVAL=0.25", actions)
        self.assertIn("--setenv=WAIT_FOR_CONTAINER_SECONDS=120", actions)


if __name__ == "__main__":
    unittest.main()
