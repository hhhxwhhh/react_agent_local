"""调度器预算透传的单元测试（不启动真实子进程）。"""

from __future__ import annotations

import io
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import scheduler  # noqa: E402


class TestSchedulerBudget(unittest.TestCase):
    def _build_cmd(self, **kwargs) -> list[str]:
        proc = MagicMock()
        proc.stdout = []
        proc.wait.return_value = 0
        with patch.object(scheduler.subprocess, "Popen", return_value=proc) as popen:
            scheduler._run_generation(io.StringIO(), **kwargs)
        self.assertTrue(popen.called)
        return list(popen.call_args.args[0])

    def test_budget_flags_are_forwarded(self):
        cmd = self._build_cmd(max_steps=60, max_diff_lines=1200)
        self.assertEqual(cmd[cmd.index("--max-steps") + 1], "60")
        self.assertEqual(cmd[cmd.index("--max-diff-lines") + 1], "1200")

    def test_budget_flags_are_omitted_by_default(self):
        cmd = self._build_cmd()
        self.assertNotIn("--max-steps", cmd)
        self.assertNotIn("--max-diff-lines", cmd)

    def test_once_mode_forwards_budget(self):
        with patch.object(scheduler, "_preflight", return_value=True), patch.object(
            scheduler, "_run_generation", return_value=0
        ) as run, patch.object(scheduler, "_ledger_entries", return_value=[]):
            code = scheduler._run_once(io.StringIO(), 60, 1200)
        self.assertEqual(code, 0)
        run.assert_called_once()
        self.assertEqual(run.call_args.args[1:], (60, 1200))

    def test_parser_accepts_budget_flags(self):
        args = scheduler._build_parser().parse_args(["--max-steps", "60", "--max-diff-lines", "1200"])
        self.assertEqual(args.max_steps, 60)
        self.assertEqual(args.max_diff_lines, 1200)
        defaults = scheduler._build_parser().parse_args([])
        self.assertIsNone(defaults.max_steps)
        self.assertIsNone(defaults.max_diff_lines)


if __name__ == "__main__":
    unittest.main()
