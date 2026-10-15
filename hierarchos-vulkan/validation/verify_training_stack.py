#!/usr/bin/env python3
"""Run the whole Hierarchos Vulkan training verification ladder in one pass.

The checks below were previously run by hand, in this order, because each one
constrains a different layer and a change can satisfy one while breaking another:

  1. build the four ladder binaries          - every required harness builds
  2. binding-access audit (`--check`)        - no shader still declares access that
                                              is provably narrower than its
                                              declaration
  3. shader provenance                       - the checked-in `.spv` files are what
                                              their sources and documented flags
                                              produce
  4. golden digests, three parity harnesses  - kernel outputs are still
                                              bit-identical, checked against
                                              recorded digests instead of
                                              pre-change artifacts
  5. training-step probe                     - the end-to-end step still has the
                                              expected dispatch and barrier counts,
                                              the expected loss, and a finite graph
  6. full training trajectories              - parameters, Adam moments/clocks,
                                              losses and recurrent adjoints match
                                              the pre-optimization baseline

Each step reports PASS, FAIL, or BLOCKED, and the script exits non-zero unless
every step passed. A step that cannot run — missing toolchain, missing binary,
unrecorded device — is a failure, not a pass: the value of the ladder is that green
means verified.

The probe is compared against `validation/training_step_probe/step_expectation.json`.
It runs on a fresh model package under `target/`, because an existing package keeps
advancing through AdamW and every run would then measure a different point in
training. The dispatch count is structural and must match, the barrier count is an
upper bound so that an elision improvement does not fail the ladder, and the loss is
compared bit-for-bit per device name, since bit-identical arithmetic is only
guaranteed within one driver. A device with no recorded loss is reported as
unverified, which fails, and `--record-expectation` writes the entry for the device
in this machine.

Usage:
    python validation/verify_training_stack.py
    python validation/verify_training_stack.py --release
    python validation/verify_training_stack.py --only golden
    python validation/verify_training_stack.py --record-expectation
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

CRATE = Path(__file__).resolve().parents[1]
GOLDEN_DIR = CRATE / "validation" / "kernel_golden"
EXPECTATION = CRATE / "validation" / "training_step_probe" / "step_expectation.json"
SCRATCH = CRATE / "target" / "verify_training_stack"

# The probe is a fixed fixture: these arguments and the expectation go together,
# and the comparison fails if they drift apart.
PROBE_ARGS = ["--batch", "2", "--tokens", "8", "--steps", "2", "--warmup", "0"]

HARNESSES = [
    ("golden dense (matmul)", "matmul_kernel_validation", GOLDEN_DIR / "dense.json"),
    ("golden reduction", "reduction_kernel_validation", GOLDEN_DIR / "reduction.json"),
    ("golden attention", "attention_kernel_cache_check", GOLDEN_DIR / "attention.json"),
]

# Exactly the binaries this ladder runs. `cargo build --bins` would also compile
# every unrelated harness in the crate, so an unrelated bin that does not build
# would make this ladder red for a reason that has nothing to do with the stack it
# verifies.
LADDER_BINS = [harness for _, harness, _ in HARNESSES] + ["training_step_probe"]

EXPECTATION_NOTE = (
    "Expectation for validation/verify_training_stack.py. probe_args, dispatch_count "
    "and shader_barrier_count_max describe the fixture structurally and are compared "
    "for every device; the barrier count is an upper bound so an elision improvement "
    "does not fail the ladder. mean_loss is recorded per device name and compared "
    "bit-for-bit, because bit-identical arithmetic is only guaranteed within one "
    "driver. Re-record with --record-expectation after an intended numerical change."
)


class Step:
    """One verification step and its outcome."""

    def __init__(self, name, detail="", ok=False, blocked=False, output=""):
        self.name = name
        self.detail = detail
        self.ok = ok
        self.blocked = blocked
        self.output = output

    @property
    def status(self):
        if self.ok:
            return "PASS"
        return "BLOCKED" if self.blocked else "FAIL"


def run(command):
    """Run a command in the crate directory, returning (exit code, combined output).

    A missing executable is code 127 with the error text rather than an exception,
    so every step still produces a status line.
    """
    try:
        completed = subprocess.run(
            [str(part) for part in command],
            cwd=str(CRATE),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except OSError as err:
        return 127, f"cannot execute {command[0]}: {err}"
    return completed.returncode, completed.stdout


def last_nonempty(output):
    for line in reversed(output.strip().splitlines()):
        if line.strip():
            return line.strip()
    return ""


def binary(name, profile):
    return CRATE / "target" / profile / (name + (".exe" if os.name == "nt" else ""))


def build(profile, skip):
    if skip:
        return None
    flag = ["--release"] if profile == "release" else []
    binaries = [part for name in LADDER_BINS for part in ("--bin", name)]
    code, output = run(["cargo", "build", *flag, *binaries])
    name = "cargo build (ladder binaries)"
    if code != 0:
        return Step(name, f"exit {code}: {last_nonempty(output)}", output=output)
    return Step(name, f"finished ({profile})", ok=True, output=output)


def binding_audit():
    code, output = run(
        [
            sys.executable,
            str(CRATE / "validation" / "audit_shader_binding_access.py"),
            "--check",
            "--json",
            str(SCRATCH / "binding_access.json"),
        ]
    )
    detail = last_nonempty(output)
    return Step("binding-access audit", detail, ok=code == 0, output=output)


def shader_provenance():
    report = SCRATCH / "shader_provenance.json"
    code, output = run(
        [
            sys.executable,
            str(CRATE / "validation" / "verify_training_shaders.py"),
            "--output",
            str(report),
        ]
    )
    if code == 0:
        try:
            variants = json.loads(report.read_text(encoding="utf-8"))["variants"]
            detail = f"{len(variants)} variant(s) match their sources and flags"
        except (OSError, KeyError, ValueError):
            detail = "rebuilt artifacts match their sources and flags"
        return Step("shader provenance", detail, ok=True, output=output)
    blocked = "must be on PATH" in output
    return Step(
        "shader provenance",
        f"exit {code}: {last_nonempty(output)}",
        blocked=blocked,
        output=output,
    )


def golden_harness(name, harness, golden, profile):
    executable = binary(harness, profile)
    if not executable.exists():
        return Step(name, f"{executable} is missing; do not pass --skip-build", blocked=True)
    if not golden.exists():
        return Step(
            name,
            f"{golden} is missing; record it with --write-golden before enforcing this step",
            blocked=True,
        )
    # Exercise both reduction comparison modes on every run. Comparing two
    # independently dispatched copies also detects a comparison accidentally
    # moved before the candidate dispatch, even without old artifacts in CI.
    baseline = [CRATE / "shaders"] if harness == "reduction_kernel_validation" else []
    code, output = run([executable, *baseline, "--golden", golden, "--golden-strict"])
    return Step(name, last_nonempty(output), ok=code == 0, output=output)


def training_step(record, profile):
    executable = binary("training_step_probe", profile)
    if not executable.exists():
        return Step("training-step probe", f"{executable} is missing; do not pass --skip-build", blocked=True)
    fixture = SCRATCH / "fresh_fixture"
    if fixture.exists():
        shutil.rmtree(fixture)
    code, output = run([executable, "--model-dir", fixture, *PROBE_ARGS])
    if code != 0:
        return Step("training-step probe", f"exit {code}: {last_nonempty(output)}", output=output)
    try:
        observed = json.loads(output[output.index("{"):])
    except (ValueError, KeyError) as err:
        return Step("training-step probe", f"unparseable output: {err}", output=output)

    device = observed["device"]
    expectation = json.loads(EXPECTATION.read_text(encoding="utf-8")) if EXPECTATION.exists() else {}
    devices = expectation.get("devices", {})

    if record:
        devices[device] = {"mean_loss": observed["mean_loss"]}
        EXPECTATION.parent.mkdir(parents=True, exist_ok=True)
        EXPECTATION.write_text(
            json.dumps(
                {
                    "note": EXPECTATION_NOTE,
                    "probe_args": PROBE_ARGS,
                    "dispatch_count": observed["dispatch_count"],
                    "shader_barrier_count_max": observed["shader_barrier_count"],
                    "devices": devices,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return Step(
            "training-step probe",
            f"recorded for {device}: {observed['dispatch_count']} dispatches, "
            f"{observed['shader_barrier_count']} barriers, loss {observed['mean_loss']}",
            ok=True,
            output=output,
        )

    problems = []
    if expectation.get("probe_args") != PROBE_ARGS:
        problems.append(f"recorded probe arguments {expectation.get('probe_args')} do not match {PROBE_ARGS}")
    expected_dispatches = expectation.get("dispatch_count")
    if expected_dispatches is None:
        problems.append("no expectation recorded; run with --record-expectation first")
    elif observed["dispatch_count"] != expected_dispatches:
        problems.append(f"dispatch count {observed['dispatch_count']} != recorded {expected_dispatches}")
    barrier_max = expectation.get("shader_barrier_count_max")
    if barrier_max is not None and observed["shader_barrier_count"] > barrier_max:
        problems.append(
            f"barrier count {observed['shader_barrier_count']} exceeds the recorded upper bound {barrier_max}"
        )
    recorded_loss = devices.get(device, {}).get("mean_loss")
    if recorded_loss is None:
        problems.append(f"no expectation recorded for {device}; run with --record-expectation on this device")
    elif observed["mean_loss"] != recorded_loss:
        problems.append(f"loss {observed['mean_loss']} != recorded {recorded_loss}")

    detail = (
        f"{observed['dispatch_count']} dispatches, {observed['shader_barrier_count']} barriers "
        f"(upper bound {barrier_max}), loss {observed['mean_loss']} on {device}"
    )
    if problems:
        return Step("training-step probe", "; ".join(problems), output=output)
    return Step("training-step probe", detail, ok=True, output=output)


def training_state(profile):
    code, output = run([
        sys.executable, CRATE / "validation" / "verify_training_state.py",
        "--probe", binary("training_step_probe", profile),
        "--output", SCRATCH / "training_state.json",
    ])
    return Step("training-state parity", last_nonempty(output), ok=code == 0, output=output)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--release", action="store_true", help="use release binaries")
    parser.add_argument("--skip-build", action="store_true", help="do not run cargo build")
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="run only the steps whose name contains this text (repeatable)",
    )
    parser.add_argument(
        "--record-expectation",
        action="store_true",
        help="record the probe's instance counts and loss for the device in this machine",
    )
    parser.add_argument("--receipt", type=Path, help="write a JSON summary of this run here")
    args = parser.parse_args(argv)

    profile = "release" if args.release else "debug"
    SCRATCH.mkdir(parents=True, exist_ok=True)

    ladder = [
        ("cargo build", lambda: build(profile, args.skip_build)),
        ("binding-access audit", binding_audit),
        ("shader provenance", shader_provenance),
        *[
            (name, lambda name=name, harness=harness, golden=golden: golden_harness(name, harness, golden, profile))
            for name, harness, golden in HARNESSES
        ],
        ("training-step probe", lambda: training_step(args.record_expectation, profile)),
        ("training-state parity", lambda: training_state(profile)),
    ]

    selected = ladder
    if args.only:
        selected = [entry for entry in ladder if any(needle in entry[0] for needle in args.only)]
        if not selected:
            available = ", ".join(name for name, _ in ladder)
            print(f"--only {args.only} matched no step; available: {available}", file=sys.stderr)
            return 2

    steps = []
    for name, check in selected:
        step = check()
        if step is None:
            continue
        steps.append(step)

    width = max(len(step.name) for step in steps)
    print()
    for step in steps:
        print(f"{step.status:8} {step.name:<{width}}  {step.detail}")
    failures = [step for step in steps if not step.ok]
    print()
    print(f"{len(steps) - len(failures)}/{len(steps)} step(s) passed")

    for step in failures:
        body = step.output.strip()
        print(f"--- output of {step.name} ({step.status}) ---")
        print(body[-4000:] if body else "(no output)")

    if args.receipt:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(
            json.dumps(
                [
                    {"step": step.name, "status": step.status, "detail": step.detail}
                    for step in steps
                ],
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
