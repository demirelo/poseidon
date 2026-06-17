"""
Extract published golden vectors from the vendored reference WITHOUT executing
it.  We parse tests/test_poseidon.py with `ast` and literal-eval only the three
data assignments we need (a circulant MDS first row, its round constants, and
the expected permutation output).  No reference code is imported or run.

Output: tests/golden_vectors.json
"""
import ast
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REF = os.path.join(ROOT, "reference", "poseidon-tools", "tests", "test_poseidon.py")

WANT = {"_KB_MDS_FIRST_ROW_16", "_KB_ROUND_CONSTANTS_16", "_KB_EXPECTED_16"}

with open(REF) as f:
    tree = ast.parse(f.read())

found = {}
for node in tree.body:
    # plain `x = [...]` is ast.Assign; annotated `x: list[int] = [...]` is ast.AnnAssign
    if isinstance(node, ast.Assign):
        targets = [t for t in node.targets if isinstance(t, ast.Name)]
        value = node.value
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        targets = [node.target]
        value = node.value
    else:
        continue
    for tgt in targets:
        if tgt.id in WANT and value is not None:
            found[tgt.id] = ast.literal_eval(value)

missing = WANT - set(found)
if missing:
    raise SystemExit(f"could not extract: {missing}")

out = {
    "plonky3_w16": {
        "mds_first_row": found["_KB_MDS_FIRST_ROW_16"],
        "round_constants": found["_KB_ROUND_CONSTANTS_16"],
        "expected_perm_of_range16": found["_KB_EXPECTED_16"],
        "rf": 8, "rp": 20, "t": 16, "alpha": 3,
        "source": "Plonky3 koala-bear/src/poseidon1.rs via leanSpec (vendored test_poseidon.py)",
    }
}
with open(os.path.join(HERE, "golden_vectors.json"), "w") as f:
    json.dump(out, f, indent=2)
print(f"wrote golden_vectors.json: "
      f"{len(out['plonky3_w16']['round_constants'])} round constants, "
      f"expected[0]={out['plonky3_w16']['expected_perm_of_range16'][0]}")
