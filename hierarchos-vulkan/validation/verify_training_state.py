"""Check complete training trajectories against pre-optimization GPU results.

Goldens were obtained from the pre-optimization shaders and runtime, retaining
the earlier CE/objective work. They cover every parameter, both Adam moments,
per-slot step counters, losses, recurrent states and initial-state adjoints after
each update. Unlike a loss-only check, this catches drift before it affects loss.
No command in this verifier records or updates goldens.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

CRATE = Path(__file__).resolve().parents[1]
GOLDEN = CRATE / "validation" / "kernel_golden" / "training_state.json"


def compare_snapshots(expected, actual):
    errors = []
    if len(expected) != len(actual):
        errors.append(f"step count {len(actual)} != {len(expected)}")
    for index, (reference, observed) in enumerate(zip(expected, actual)):
        for field in ("step", "optimizer_step", "optimizer_slot_steps"):
            if reference.get(field) != observed.get(field):
                errors.append(f"step {index}: {field} differs")
        before = reference.get("tensors", {})
        after = observed.get("tensors", {})
        for name in sorted(before.keys() | after.keys()):
            if before.get(name) != after.get(name):
                errors.append(f"step {index}: {name} differs")
    return errors


def verify(probe, golden=GOLDEN):
    recorded = json.loads(Path(golden).read_text(encoding="utf-8"))
    if recorded.get("schema") != 1 or not recorded.get("cases"):
        raise RuntimeError("unsupported or empty training-state golden file")
    scratch = CRATE / "target" / "verify_training_stack"
    scratch.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix="state-parity-", dir=scratch) as temp:
        temp = Path(temp)
        for index, case in enumerate(recorded["cases"]):
            snapshot = temp / f"state_{index}.json"
            command = [str(probe), "--model-dir", str(temp / f"fixture_{index}"),
                       *case["args"], "--state-output", str(snapshot)]
            result = subprocess.run(command, cwd=CRATE, text=True, capture_output=True)
            if result.returncode:
                raise RuntimeError(f"{case['name']} probe exit {result.returncode}: {result.stderr[-3000:]}")
            output = json.loads(result.stdout[result.stdout.index("{"):])
            device = output["device"]
            if device not in case["devices"]:
                raise RuntimeError(f"{case['name']}: no pre-optimization trajectory recorded for {device}")
            expected = case["devices"][device]
            actual = json.loads(snapshot.read_text(encoding="utf-8"))
            errors = compare_snapshots(expected, actual)
            rows.append({"case": case["name"], "device": device, "steps": len(actual),
                         "tensor_comparisons": sum(len(step["tensors"]) for step in expected),
                         "pass": not errors, "errors": errors})
    return {"pass": all(row["pass"] for row in rows), "cases": rows,
            "baseline_commit": recorded["baseline"]["base_commit"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, default=CRATE / "target" / "debug" /
                        ("training_step_probe.exe" if os.name == "nt" else "training_step_probe"))
    parser.add_argument("--golden", type=Path, default=GOLDEN)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify(args.probe.resolve(), args.golden)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for row in report["cases"]:
        if not row["pass"]:
            print(f"FAIL {row['case']}: " + "; ".join(row["errors"][:8]))
    print(f"{'PASS' if report['pass'] else 'FAIL'}: "
          f"{sum(row['steps'] for row in report['cases'])} training steps, "
          f"{sum(row['tensor_comparisons'] for row in report['cases'])} exact tensor digests")
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
