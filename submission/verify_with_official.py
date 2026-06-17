#!/usr/bin/env python3
"""Submission verifier bridge for Poseidon bounty candidates.

This script always checks a candidate with the independent verifier in this
repository. It runs the vendored official verifier only when explicitly passed
`--official`, so normal smoke checks do not execute reference code.

Input JSON examples:

  {"challenge": "cico", "rp": 10, "free_inputs": [14 integers]}
  {"challenge": "zerotest", "rp": 6, "p_hat": [16 integers]}

`rf` is optional and defaults to 6 (the funded bounty instance). Reduced-round
research candidates may set it explicitly, e.g.

  {"challenge": "zerotest", "rf": 4, "rp": 5, "p_hat": [16 integers]}

The process exits 0 only if every requested verifier returns true.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = REPO_ROOT / "reference" / "poseidon-tools"
VENDORED_COMMIT = REFERENCE_DIR / "VENDORED_COMMIT.txt"
FIELD_MODULUS = 2**31 - 2**24 + 1

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class InputError(ValueError):
    """Raised for malformed candidate JSON."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError as exc:
        raise InputError(f"input file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise InputError(f"input file is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise InputError("input JSON must be an object")
    return data


def _as_field_list(value: Any, *, name: str, expected_len: int | None = None) -> list[int]:
    if not isinstance(value, list):
        raise InputError(f"`{name}` must be a list")
    out: list[int] = []
    for i, item in enumerate(value):
        if not isinstance(item, int):
            raise InputError(f"`{name}[{i}]` must be an integer")
        if item < 0 or item >= FIELD_MODULUS:
            raise InputError(f"`{name}[{i}]` must be in [0, {FIELD_MODULUS})")
        out.append(item)
    if expected_len is not None and len(out) != expected_len:
        raise InputError(f"`{name}` must have length {expected_len}, got {len(out)}")
    return out


def _challenge(data: dict[str, Any], override: str | None) -> str:
    value = override or data.get("challenge") or data.get("type")
    if not isinstance(value, str):
        raise InputError("challenge must be provided as `cico` or `zerotest`")
    value = value.lower().replace("_", "-")
    aliases = {
        "cico": "cico",
        "cico-2": "cico",
        "zerotest": "zerotest",
        "zero-test": "zerotest",
        "zero_test": "zerotest",
    }
    try:
        return aliases[value]
    except KeyError as exc:
        raise InputError(f"unsupported challenge `{value}`; expected cico or zerotest") from exc


def _rp(data: dict[str, Any], override: int | None) -> int:
    value = override if override is not None else data.get("rp", data.get("r_p"))
    if not isinstance(value, int):
        raise InputError("RP must be provided as integer `rp` or via `--rp`")
    if value < 0:
        raise InputError("RP must be non-negative")
    return value


def _rf(data: dict[str, Any], override: int | None) -> int:
    """RF for the instance. Defaults to 6 (the funded bounty); reduced-round
    research candidates may override via the `rf`/`r_f` field or `--rf`."""
    value = override if override is not None else data.get("rf", data.get("r_f", 6))
    if not isinstance(value, int):
        raise InputError("RF must be an integer")
    if value <= 0 or value % 2:
        raise InputError("RF must be a positive even integer")
    return value


def _extract_vector(data: dict[str, Any], challenge: str) -> tuple[str, list[int]]:
    if challenge == "cico":
        for key in ("free_inputs", "x", "X", "inputs", "solution"):
            if key in data:
                return key, _as_field_list(data[key], name=key, expected_len=14)
        raise InputError("CICO input needs `free_inputs` with 14 integers")

    for key in ("p_hat", "P_hat", "coefficients", "solution"):
        if key in data:
            return key, _as_field_list(data[key], name=key, expected_len=16)
    raise InputError("zero-test input needs `p_hat` with 16 integers")


def _vendored_commit_text() -> str:
    if not VENDORED_COMMIT.exists():
        return "unknown"
    return VENDORED_COMMIT.read_text(encoding="utf-8").strip()


def _verify_independent(challenge: str, vector: list[int], rf: int, rp: int) -> bool:
    from poseidon_attack.verifiers import verify_cico, verify_zerotest

    if challenge == "cico":
        return bool(verify_cico(vector, r_f=rf, r_p=rp))
    return bool(verify_zerotest(vector, r_f=rf, r_p=rp))


def _verify_official(challenge: str, vector: list[int], rf: int, rp: int) -> bool:
    sys.path.insert(0, str(REFERENCE_DIR))
    if challenge == "cico":
        module = importlib.import_module("bounties.cico_verifier")
        return bool(module.verify_cico_solution(vector, r_f=rf, r_p=rp))
    module = importlib.import_module("bounties.zerotest_verifier")
    return bool(module.verify_zerotest_solution(vector, r_f=rf, r_p=rp))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate", type=Path, help="candidate JSON file")
    parser.add_argument("--challenge", choices=("cico", "zerotest"), help="override JSON challenge")
    parser.add_argument("--rp", type=int, help="override JSON RP")
    parser.add_argument("--rf", type=int, help="override JSON RF (default 6, the funded bounty)")
    parser.add_argument(
        "--official",
        action="store_true",
        help="also execute the vendored official verifier at reference/poseidon-tools",
    )
    args = parser.parse_args(argv)

    try:
        data = _load_json(args.candidate)
        challenge = _challenge(data, args.challenge)
        rp = _rp(data, args.rp)
        rf = _rf(data, args.rf)
        vector_name, vector = _extract_vector(data, challenge)
        independent = _verify_independent(challenge, vector, rf, rp)
        official = None
        if args.official:
            official = _verify_official(challenge, vector, rf, rp)
    except InputError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2), file=sys.stderr)
        return 2

    report: dict[str, Any] = {
        "ok": independent and (official is not False),
        "challenge": challenge,
        "rf": rf,
        "rp": rp,
        "vector_field": vector_name,
        "vector_len": len(vector),
        "independent": independent,
        "official": official,
        "official_executed": args.official,
        "official_reference": str(REFERENCE_DIR.relative_to(REPO_ROOT)),
        "vendored_commit": _vendored_commit_text(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
