# Generated fixtures

Only tiny, deterministic, reviewable fixtures are committed. The hex fixture
`tiny_u8.safetensors.hex` materializes to 106 bytes with the SHA-256 recorded in
`manifest.json`; `scripts/check_fixtures.sh` verifies both values, and the
interop script verifies them before reading it.
Large-file tests
are generated at runtime and must record generator version, expected length or
digest, license, and cleanup. `scripts/python_interop.sh` currently generates
its fixture at runtime; it is not a vendored upstream artifact.
