#!/usr/bin/env bash
set -euo pipefail

# Black-box interoperability check against the upstream Python reader.  This
# script never downloads dependencies: an absent package is an explicit
# environment skip (exit 77), not a passing test.
moon_bin="${MOON_BIN:-moon}"
python_bin="${PYTHON_BIN:-python3}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if ! "$python_bin" -c 'import safetensors, numpy' >/dev/null 2>&1; then
  printf 'SKIP: install safetensors==0.6.2 and numpy==2.2.6 to run interop\n' >&2
  exit 77
fi

version="$($python_bin -c 'import importlib.metadata as m; print(m.version("safetensors"))')"
numpy_version="$($python_bin -c 'import importlib.metadata as m; print(m.version("numpy"))')"
test "$version" = "0.6.2"
test "$numpy_version" = "2.2.6"
printf 'Python safetensors version: %s\n' "$version"
printf 'NumPy version: %s\n' "$numpy_version"
workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT
fixture="$workdir/fixed_fixture.safetensors"
parts="$workdir/parts"

"$python_bin" - "$ROOT/test/fixtures/tiny_u8.safetensors.hex" "$fixture" <<'PY'
import pathlib, sys
hex_path, output_path = sys.argv[1:]
data = bytes.fromhex(pathlib.Path(hex_path).read_text())
pathlib.Path(output_path).write_bytes(data)
PY

test "$(wc -c < "$fixture")" -eq 106
test "$(sha256sum "$fixture" | awk '{print $1}')" = \
  e4e4fcacec662f078e53191ea0346ecf25b63f5944c296205f09223e29fe8a5b

"$moon_bin" run src/cmd/sftk -- validate "$fixture" --format json | grep -F '"valid":true' >/dev/null
"$moon_bin" run src/cmd/sftk -- split "$fixture" --max-bytes 3B --output-dir "$parts" --write >/dev/null

"$python_bin" - "$fixture" "$parts" <<'PY'
from safetensors import safe_open
import sys, pathlib
fixture, parts = sys.argv[1:]
with safe_open(fixture, framework="np") as f:
    assert f.keys() == ["a"], f.keys()
    assert f.get_tensor("a").tolist() == [97, 98, 99]
index = pathlib.Path(parts) / "model.safetensors.index.json"
import json
mapping = json.loads(index.read_text())["weight_map"]
assert mapping["a"] == "model-00001-of-00001.safetensors", mapping
with safe_open(str(pathlib.Path(parts) / mapping["a"]), framework="np") as f:
    assert f.get_tensor("a").tolist() == [97, 98, 99]
PY

printf 'PASS: toolkit output is readable by Python safetensors (%s)\n' "$version"
