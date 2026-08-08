"""`deploy/run_gpu_eval.sh`'s case selector, exercised without a GPU.

The script is the only thing that runs on a paid box, unattended, with nobody
watching — and `CASES` is a knob whose failure mode is *silent*: a typo that
fell through to "run everything" would spend the night and the money on the
matrix the operator explicitly asked to skip. So it is tested three ways:

* ``bash -n`` — the script still parses at all;
* sourced with ``TIDAL_EVAL_SOURCE_ONLY=1``, which stops it after the function
  definitions, so ``parse_cases`` / ``compute_total_cases`` are called for real
  rather than re-implemented in the test;
* end to end with a stub ``PYTHON_BIN`` and a stub ``vllm``, which is the only
  way to see that a deselected case is not merely absent from an array but is
  never handed to the harness, and that ``PROGRESS total`` matches.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "deploy" / "run_gpu_eval.sh"

ALL_CASES = (
    "online_only",
    "offline_only",
    "naive",
    "technique_a",
    "technique_b",
    "diurnal_technique_a",
    "diurnal_technique_b",
    "deadline_control",
    "deadline_laxity",
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def source_and_run(snippet: str, **env: str) -> subprocess.CompletedProcess:
    """Source the script (functions only) and run `snippet` against it."""
    environ = dict(os.environ)
    environ.update(
        {
            "TIDAL_EVAL_SOURCE_ONLY": "1",
            # Sourcing still lays out the run directory; keep it off /results.
            "RESULTS_DIR": tempfile.mkdtemp(prefix="tidal-cases-"),
            **env,
        }
    )
    return subprocess.run(
        ["bash", "-c", f'. "{SCRIPT}"\n{snippet}'],
        cwd=REPO,
        env=environ,
        capture_output=True,
        text=True,
    )


def stub_python(tmp_path: Path) -> Path:
    """A `PYTHON_BIN` that fakes the harness, pytest and the plot renderer.

    Everything else — the JSON parsing the script does through heredocs, the
    pool arithmetic — is forwarded to the real interpreter, because those are
    the parts the script actually depends on being correct.
    """
    path = tmp_path / "stub-python"
    path.write_text(
        f"""#!/usr/bin/env bash
if [ "$1" = "-m" ]; then
  case "$2" in
    pytest)
      if [ "${{FAIL_CANARY:-0}}" = "1" ]; then echo "1 failed"; exit 1; fi
      echo "1 passed"; exit 0 ;;
    tidal.eval.plots)
      echo "rendered"; exit 0 ;;
    tidal.eval.harness)
      out=""; prev=""
      for arg in "$@"; do
        if [ "$prev" = "--out" ]; then out="$arg"; fi
        prev="$arg"
      done
      printf '%s\\n' "$*" >>"${{HARNESS_ARGV_LOG}}"
      mkdir -p "$(dirname "$out")"
      printf '%s' '{{"condition":"stub","batch":{{"completed":100,"makespan_s":10.0}}}}' >"$out"
      exit 0 ;;
  esac
fi
exec {sys.executable} "$@"
"""
    )
    path.chmod(0o755)
    return path


def run_script(tmp_path: Path, **env: str) -> subprocess.CompletedProcess:
    tmp_path.mkdir(parents=True, exist_ok=True)
    vllm = tmp_path / "vllm"
    vllm.write_text("#!/usr/bin/env bash\nexit 0\n")
    vllm.chmod(0o755)
    results = tmp_path / "results"
    results.mkdir(exist_ok=True)
    environ = dict(os.environ)
    environ.update(
        {
            "REPO_DIR": str(REPO),
            "RESULTS_DIR": str(results),
            "PYTHON_BIN": str(stub_python(tmp_path)),
            "VLLM_BIN": str(vllm),
            "HARNESS_ARGV_LOG": str(tmp_path / "argv.log"),
            "PREWARM": "0",
            "MAKE_TARBALL": "0",
            "RESUME": "0",
            **env,
        }
    )
    return subprocess.run(
        ["bash", str(SCRIPT)], cwd=REPO, env=environ, capture_output=True, text=True
    )


def progress(stderr: str, verb: str) -> list[str]:
    return re.findall(rf"^PROGRESS {verb} (.+)$", stderr, flags=re.MULTILINE)


def cases_run(result: subprocess.CompletedProcess) -> list[str]:
    return progress(result.stderr, "case_done")


# ---------------------------------------------------------------------------
# the script still parses
# ---------------------------------------------------------------------------


def test_the_script_is_valid_bash():
    assert subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True).returncode == 0


# ---------------------------------------------------------------------------
# parse_cases
# ---------------------------------------------------------------------------


def test_the_default_selection_is_every_case():
    done = source_and_run('parse_cases "$CASES"; printf "%s\\n" "${SELECTED_CASES[*]}"')
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == list(ALL_CASES)


def test_only_the_listed_cases_are_selected():
    done = source_and_run('parse_cases "$CASES"; printf "%s\\n" "${SELECTED_CASES[*]}"')
    full = done.stdout.split()
    trimmed = source_and_run(
        'parse_cases "$CASES"; printf "%s\\n" "${SELECTED_CASES[*]}"',
        CASES="technique_a,deadline_laxity",
    )
    assert trimmed.stdout.split() == ["technique_a", "deadline_laxity"]
    assert len(full) == 9


def test_whitespace_and_duplicates_are_tolerated():
    """`CASES` gets pasted into a CaaS `env_variables` field by hand; a stray
    space after a comma must not cost a run."""
    done = source_and_run(
        'parse_cases "$CASES"; printf "%s\\n" "${SELECTED_CASES[*]}"',
        CASES=" naive , technique_a ,naive",
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["naive", "technique_a"]


@pytest.mark.parametrize("bad", ["typo", "probe_offline", "diurnal", "online-only"])
def test_an_unknown_case_is_fatal_and_prints_the_valid_list(bad):
    """Including names that *look* plausible: the probe is not selectable (it
    always runs), and neither `diurnal` nor a dashed spelling is a case."""
    done = source_and_run('parse_cases "$CASES"', CASES=f"technique_a,{bad}")
    assert done.returncode != 0
    assert f"unknown case(s) in CASES: {bad}" in done.stderr
    for name in ALL_CASES:
        assert name in done.stderr


def test_every_unknown_name_is_reported_at_once():
    done = source_and_run('parse_cases "$CASES"', CASES="nope,online_only,nah")
    assert "unknown case(s) in CASES: nope nah" in done.stderr


def test_a_selection_that_resolves_to_nothing_is_fatal():
    done = source_and_run('parse_cases "$CASES"', CASES="  ,  ,")
    assert done.returncode != 0
    assert "CASES selected nothing" in done.stderr


def test_an_unknown_case_fails_before_any_other_preflight_check():
    """The point of failing in preflight is that it costs seconds. It has to
    beat the `vllm`-on-PATH check to be the error the operator actually sees."""
    done = subprocess.run(
        ["bash", str(SCRIPT)],
        cwd=REPO,
        env={**os.environ, "CASES": "bogus", "RESULTS_DIR": "/tmp", "VLLM_BIN": ""},
        capture_output=True,
        text=True,
    )
    assert done.returncode == 1
    assert "unknown case(s) in CASES: bogus" in done.stderr
    assert "no 'vllm' on PATH" not in done.stderr


# ---------------------------------------------------------------------------
# the /status denominator
# ---------------------------------------------------------------------------


def test_total_counts_the_selection_plus_the_always_run_probe():
    for cases, expected in (
        ("", "10"),  # 1 probe + 9
        ("technique_a", "2"),
        ("technique_a,technique_b,diurnal_technique_a", "4"),
    ):
        env = {"CASES": cases} if cases else {}
        done = source_and_run('parse_cases "$CASES"; CANARY_OK=1; compute_total_cases', **env)
        assert done.stdout.strip() == expected, (cases, done.stdout, done.stderr)


def test_a_failed_canary_only_discounts_selected_technique_b_cases():
    both = source_and_run('parse_cases "$CASES"; CANARY_OK=0; compute_total_cases')
    assert both.stdout.strip() == "8"  # 10 - technique_b - diurnal_technique_b
    one = source_and_run(
        'parse_cases "$CASES"; CANARY_OK=0; compute_total_cases',
        CASES="technique_a,technique_b",
    )
    assert one.stdout.strip() == "2"  # probe + technique_a
    none = source_and_run(
        'parse_cases "$CASES"; CANARY_OK=0; compute_total_cases',
        CASES="deadline_control,deadline_laxity",
    )
    assert none.stdout.strip() == "3"


# ---------------------------------------------------------------------------
# end to end, with a stub harness
# ---------------------------------------------------------------------------


def test_a_full_run_reports_and_runs_every_case(tmp_path):
    done = run_script(tmp_path)
    assert done.returncode == 0, done.stderr[-3000:]
    assert progress(done.stderr, "total") == ["10"]
    assert cases_run(done) == ["probe_offline", *ALL_CASES]


def test_a_trimmed_run_never_hands_the_deselected_cases_to_the_harness(tmp_path):
    done = run_script(tmp_path, CASES="technique_a,diurnal_technique_b")
    assert done.returncode == 0, done.stderr[-3000:]
    assert progress(done.stderr, "total") == ["3"]
    assert cases_run(done) == ["probe_offline", "technique_a", "diurnal_technique_b"]

    # Not just absent from the progress stream — never invoked at all. The
    # probe is there because it always runs, whatever CASES says.
    invocations = (tmp_path / "argv.log").read_text().splitlines()
    assert len(invocations) == 3
    assert sum("--condition offline_only" in line for line in invocations) == 1
    assert not any("deadline" in line for line in invocations)
    for name in ("online_only.json", "naive.json", "technique_b.json", "deadline_laxity.json"):
        assert list((tmp_path / "results").rglob(name)) == []


def test_the_sizing_probe_runs_even_when_offline_only_is_deselected(tmp_path):
    """The probe is the divisor every pool size is computed from, so it is not
    the same thing as the measured `offline_only` condition and does not go
    away with it."""
    done = run_script(tmp_path, CASES="technique_a")
    assert done.returncode == 0, done.stderr[-3000:]
    assert cases_run(done) == ["probe_offline", "technique_a"]
    assert list((tmp_path / "results").rglob("offline_probe.json"))
    assert "offline ceiling" in done.stderr
    # And the pool it sized is a real number, not the fallback.
    assert re.search(r"matrix pool size: \d+ items", done.stderr)


def test_the_manifest_records_the_selection(tmp_path):
    done = run_script(tmp_path, CASES=" deadline_control , technique_a ")
    assert done.returncode == 0, done.stderr[-3000:]
    manifest = next((tmp_path / "results").rglob("MANIFEST.txt")).read_text()
    # The resolved selection, in the order the script runs it...
    assert "cases:             technique_a deadline_control" in manifest
    # ...and what the operator actually typed, verbatim.
    assert "cases_env:" in manifest
    assert " deadline_control , technique_a " in manifest


def test_figures_are_skipped_when_no_matrix_case_was_selected(tmp_path):
    """Rendering an empty matrix directory would fail, and a failed render is a
    failed run — for an invocation that did exactly what it was asked to."""
    done = run_script(tmp_path, CASES="deadline_control")
    assert done.returncode == 0, done.stderr[-3000:]
    assert "skipping figures" in done.stderr
    assert "plots" not in done.stderr

    with_matrix = run_script(tmp_path / "second", CASES="naive")
    assert "rendering figures" in with_matrix.stderr


def test_a_failed_canary_trims_the_total_and_the_technique_b_cases(tmp_path):
    done = run_script(tmp_path, FAIL_CANARY="1")
    assert done.returncode == 1  # the canary failure itself is a failed case
    assert progress(done.stderr, "total") == ["8"]
    assert "technique_b" not in cases_run(done)
    assert "diurnal_technique_b" not in cases_run(done)
    assert len(cases_run(done)) == 8


def test_warmup_seconds_reach_every_case_including_the_probe(tmp_path):
    """WARMUP_S is a pass-through to `--warmup-s`, and warm-up is engine
    warm-up: the probe measures the ceiling every other pool is sized against,
    so it has to be warmed the same way the conditions it sizes are."""
    done = run_script(tmp_path, CASES="technique_a", WARMUP_S="30")
    assert done.returncode == 0, done.stderr[-3000:]
    invocations = (tmp_path / "argv.log").read_text().splitlines()
    assert len(invocations) == 2
    assert all("--warmup-s 30" in line for line in invocations)

    # Unset means "whatever --gpu-preset says": the flag is not passed at all.
    run_script(tmp_path / "bare", CASES="technique_a")
    bare = (tmp_path / "bare" / "argv.log").read_text()
    assert "--warmup-s" not in bare
