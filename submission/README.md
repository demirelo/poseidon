# submission/

Verification helpers for checking candidate witnesses against the independent
and the vendored official reference verifiers.

## Official verifier bridge

`verify_with_official.py` checks CICO and zero-test candidates against the
independent verifier first. It executes the vendored official verifier only when
`--official` is present.

Examples:

```sh
python3 submission/verify_with_official.py candidate.json
python3 submission/verify_with_official.py candidate.json --official
```

Candidate JSON shapes:

```json
{"challenge": "cico", "rp": 10, "free_inputs": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
```

```json
{"challenge": "zerotest", "rp": 6, "p_hat": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}
```

`rf` is optional and defaults to `6` (the full-round challenge instance).
Reduced-round research candidates set it explicitly (CLI overrides `--rf` /
`--rp` also work):

```json
{"challenge": "zerotest", "rf": 4, "rp": 5, "p_hat": [16 integers]}
```

The script exits `0` only when every requested verifier returns true. It exits
`1` for a well-formed candidate that fails verification and `2` for malformed
input.

The vendored reference is pinned in
`reference/poseidon-tools/VENDORED_COMMIT.txt`. Running with `--official`
executes code from that vendored tree and should be treated as the final
submission gate, not as an ordinary smoke test.

## Smoke test

`submission/examples/` holds two intentionally-failing candidates (all zeros)
and one valid reduced-round witness. The verifier must reject the bad ones
cleanly (exit `1`, not an import crash) and accept the good one (exit `0`):

```sh
python3 submission/verify_with_official.py submission/examples/bad_zerotest.json
# expect: JSON report on stdout, exit code 1 (candidate correctly rejected -- not an import crash)
python3 submission/verify_with_official.py submission/examples/bad_cico.json
# expect: exit code 1 (correctly rejected)
python3 submission/verify_with_official.py submission/examples/good_zerotest_rf4_rp5.json
# expect: JSON report with "independent": true, exit code 0 (RF=4/RP=5 witness accepted)
```

Or run all three at once from the repo root with `make smoke`.
