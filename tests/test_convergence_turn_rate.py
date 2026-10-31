import unittest

import harness


MARKER = "下一步唯一合法动作：run_tests"


def _git_step(observation: str, duplicate: bool = True) -> dict:
    return {
        "tool": "git",
        "input": {"operation": "diff"},
        "observation": observation,
        "duplicate": duplicate,
    }


def _run_tests_step() -> dict:
    return {"tool": "run_tests", "input": {}, "observation": "OK"}


def _read_step() -> dict:
    return {"tool": "read_file", "input": {"path": "a.py"}, "observation": "x"}


class TestMeasureConvergenceTurnRate(unittest.TestCase):
    def test_empty_trace(self):
        result = harness.measure_convergence_turn_rate([])
        self.assertEqual(result["blocked"], 0)
        self.assertEqual(result["marked"], 0)
        self.assertEqual(result["converged"], 0)
        self.assertEqual(result["turn_rate"], 0.0)
        self.assertEqual(result["marked_turn_rate"], 0.0)

    def test_none_trace(self):
        result = harness.measure_convergence_turn_rate(None)
        self.assertEqual(result["blocked"], 0)

    def test_marked_and_converged(self):
        trace = [
            _git_step("diff 输出\n" + MARKER),
            _run_tests_step(),
        ]
        result = harness.measure_convergence_turn_rate(trace)
        self.assertEqual(result["blocked"], 1)
        self.assertEqual(result["marked"], 1)
        self.assertEqual(result["converged"], 1)
        self.assertEqual(result["turn_rate"], 1.0)
        self.assertEqual(result["marked_turn_rate"], 1.0)

    def test_marked_but_not_converged(self):
        trace = [
            _git_step("diff 输出\n" + MARKER),
            _read_step(),
        ]
        result = harness.measure_convergence_turn_rate(trace)
        self.assertEqual(result["blocked"], 1)
        self.assertEqual(result["marked"], 1)
        self.assertEqual(result["converged"], 0)
        self.assertEqual(result["turn_rate"], 0.0)
        self.assertEqual(result["marked_turn_rate"], 0.0)

    def test_unmarked_but_converged(self):
        trace = [
            _git_step("diff 输出，无内联标记"),
            _run_tests_step(),
        ]
        result = harness.measure_convergence_turn_rate(trace)
        self.assertEqual(result["blocked"], 1)
        self.assertEqual(result["marked"], 0)
        self.assertEqual(result["converged"], 1)
        self.assertEqual(result["turn_rate"], 1.0)
        self.assertEqual(result["marked_turn_rate"], 0.0)

    def test_non_duplicate_git_ignored(self):
        trace = [
            _git_step("diff 输出\n" + MARKER, duplicate=False),
            _run_tests_step(),
        ]
        result = harness.measure_convergence_turn_rate(trace)
        self.assertEqual(result["blocked"], 0)

    def test_non_git_duplicate_ignored(self):
        trace = [
            {
                "tool": "read_file",
                "input": {"path": "a.py"},
                "observation": MARKER,
                "duplicate": True,
            },
            _run_tests_step(),
        ]
        result = harness.measure_convergence_turn_rate(trace)
        self.assertEqual(result["blocked"], 0)

    def test_marker_constant_matches_agent(self):
        self.assertEqual(harness.NEXT_LEGAL_ACTION_MARKER, MARKER)


if __name__ == "__main__":
    unittest.main()
