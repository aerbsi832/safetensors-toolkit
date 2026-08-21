#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
python3 - "$ROOT/test/fixtures/tiny_u8.safetensors.hex" <<'PY'
import hashlib, pathlib, sys
path = pathlib.Path(sys.argv[1])
data = bytes.fromhex(path.read_text())
assert len(data) == 106, len(data)
assert hashlib.sha256(data).hexdigest() == "e4e4fcacec662f078e53191ea0346ecf25b63f5944c296205f09223e29fe8a5b"
assert data[-3:] == b"abc"
print("fixed fixture verified: bytes=106 sha256=e4e4fc...e29fe8a5b")
PY
