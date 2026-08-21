# Interoperability tests

Run `scripts/python_interop.sh` with Python 3, `safetensors==0.6.2`, and
`numpy==2.2.6` installed. The script materializes a deterministic U8 fixture,
validates it with `sftk`, splits it, and reads both the input and shard with
`safe_open`. It prints the installed package version. If the package is absent,
the script exits 77 with `SKIP`; that is an environment gap, not a pass. It
rejects other package versions so the recorded oracle stays reproducible.

The test intentionally uses the package as a black-box reader and does not
vendor Python, download models, or claim cross-platform evidence.
