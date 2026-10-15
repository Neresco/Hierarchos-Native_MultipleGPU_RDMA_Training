"""Rebuild and validate every shader covered by the training optimization release.

The manifest records the actual compiler and flags for each embedded variant;
attention, reductions and mixed-precision variants do not share one build recipe.
Compilation goes to temporary files and never overwrites the shipping artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import struct
import tempfile

CRATE = Path(__file__).resolve().parents[1]
MANIFEST = Path(__file__).with_name("training_shader_manifest.json")


def shader_storage_usage(binary):
    """Read the fixed workgroup ABI and explicit Workgroup variable count."""
    words = struct.unpack(f"<{len(binary) // 4}I", binary)
    offset, variables, local_size = 5, 0, None
    while offset < len(words):
        count, opcode = words[offset] >> 16, words[offset] & 0xffff
        if count == 0 or offset + count > len(words):
            raise RuntimeError("malformed SPIR-V instruction")
        if opcode == 59 and count >= 4 and words[offset + 3] == 4:
            variables += 1
        if opcode == 16 and count == 6 and words[offset + 2] == 17:
            local_size = list(words[offset + 3:offset + 6])
        offset += count
    return {"workgroup_variables": variables, "local_size": local_size}


def verify(manifest=MANIFEST):
    recipes = json.loads(Path(manifest).read_text(encoding="utf-8"))["shaders"]
    validator = shutil.which("spirv-val")
    if validator is None:
        raise RuntimeError("spirv-val must be on PATH")
    rows = []
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    with tempfile.TemporaryDirectory(prefix="training-shader-release-") as scratch:
        for recipe in recipes:
            compiler = shutil.which(recipe["compiler"])
            if compiler is None:
                raise RuntimeError(f"{recipe['compiler']} must be on PATH")
            source = CRATE / "shaders" / recipe["source"]
            embedded = CRATE / "shaders" / recipe["artifact"]
            rebuilt = Path(scratch) / embedded.name
            command = [compiler, *recipe["flags"], str(source), "-o", str(rebuilt)]
            result = subprocess.run(command, cwd=CRATE, text=True, capture_output=True)
            row = {**recipe, "source_sha256": digest(source), "sha256": digest(embedded),
                   "compile_exit": result.returncode, "matches": False, "valid": False,
                   "policy_matches": True}
            if result.returncode == 0:
                validation = subprocess.run(
                    [validator, "--target-env", recipe["target_env"], str(rebuilt)],
                    text=True, capture_output=True,
                )
                row.update(rebuilt_sha256=digest(rebuilt),
                           matches=rebuilt.read_bytes() == embedded.read_bytes(),
                           valid=validation.returncode == 0)
                policy = recipe.get("storage_policy")
                if validation.returncode == 0 and policy:
                    usage = shader_storage_usage(rebuilt.read_bytes())
                    row["storage_usage"] = usage
                    row["policy_matches"] = (
                        usage["local_size"] == policy["local_size"]
                        and usage["workgroup_variables"] <= policy["max_workgroup_variables"]
                    )
                    if not row["policy_matches"]:
                        row["error"] = f"storage policy {policy} does not match {usage}"
                if validation.returncode:
                    row["error"] = validation.stdout + validation.stderr
            else:
                row["error"] = result.stdout + result.stderr
            rows.append(row)
    return {"pass": all(row["matches"] and row["valid"] and row["policy_matches"] for row in rows), "variants": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    for row in report["variants"]:
        if not row["matches"] or not row["valid"] or not row["policy_matches"]:
            print(f"FAIL {row['artifact']}: matches={row['matches']} valid={row['valid']} "
                  + row.get("error", ""))
    print(f"{'PASS' if report['pass'] else 'FAIL'}: {len(report['variants'])} training shader variants")
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
