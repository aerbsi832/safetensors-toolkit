# SafeTensors Toolkit for MoonBit

[简体中文](README.zh-CN.md)

SafeTensors Toolkit is a native MoonBit CLI and library for validating SafeTensors headers, comparing container structure, and performing deterministic whole-tensor split/merge operations without interpreting tensor values.

It implements the SafeTensors v1 container shape: an 8-byte little-endian header length, strict UTF-8 JSON header, `__metadata__`, tensor dtype/shape/ranges, and a contiguous data buffer. The supported dtype matrix includes current official packed F4/F6/F8 forms as well as BOOL, integer, F16/BF16/F32/F64, C64, and unsigned integer forms.

## Quick start

```bash
moon run src/cmd/sftk -- validate model.safetensors
moon run src/cmd/sftk -- info model.safetensors --format json
moon run src/cmd/sftk -- query model.safetensors 'prefix=encoder.,dtype=F16|F32,rank=2'
moon run src/cmd/sftk -- diff before.safetensors after.safetensors

# Planning is the default: no output is written.
moon run src/cmd/sftk -- split model.safetensors \
  --max-bytes 2GiB --output-dir shards

# Write only after reviewing the deterministic plan.
moon run src/cmd/sftk -- split model.safetensors \
  --max-bytes 2GiB --output-dir shards --write
moon run src/cmd/sftk -- merge shards/model.safetensors.index.json \
  --output merged.safetensors --write
moon run src/cmd/sftk -- rename-prefix model.safetensors \
  --from-prefix encoder. --to-prefix model.encoder. \
  --output renamed.safetensors --write
```

`split` emits a standard Hugging Face-style JSON index with `metadata.total_size` and `weight_map`. `merge` consumes that strict supported subset. Existing output requires `--force` in addition to `--write`.

For a minimal native consumer example (header-only validation):

```bash
moon run src/examples/validate --target native -- model.safetensors
```

## Commands and exit codes

| Command | Behavior | Exit code |
|---|---|---|
| `validate PATH` | Bounded header read and full structural validation | `0` valid, `1` invalid container |
| `info PATH` | Header-only counts and dtype distribution | `0` on success |
| `query PATH SELECTOR` | Deterministic header-only tensor query | `0` on success |
| `diff LEFT RIGHT` | Tensor/metadata structure diff; never reads tensor data | `0` equal, `1` differs |
| `split …` | Whole-tensor greedy plan; `--write` performs bounded range copies | `0` on success |
| `merge …` | Validates index/shard coverage; `--write` emits a canonical merged layout | `0` on success |
| `rename-prefix …` | Rewrites only tensor names; payload ranges remain byte-for-byte opaque | `0` on success |

Argument errors use exit `2`; file I/O and publication conflicts use exit `3`. Machine-readable results are available through `--format json`; diagnostics are written to stderr.

## Safety and compatibility boundary

- Header length is decoded as unsigned little-endian bytes; the default header bound is 100 MiB.
- The JSON parser rejects duplicate object keys, malformed strings/unicode escapes, invalid number syntax, excessive depth, and unsupported tensor fields.
- Shape multiplication, dtype bit widths, ranges, holes, overlaps, and final data-buffer coverage are checked before data is copied.
- Native input inspection reads only the 8-byte prefix plus declared bounded header. Split/merge copies data with a fixed configurable buffer (default 1 MiB) and does not decode tensor values.
- Split, rename, and merge compute a bounded payload fingerprint and re-check the source before publication. This detects same-size payload replacement, but is not cryptographic authenticity or an OS-level compare-and-swap.
- Split output is staged into temporary files; shard files are published before the index, and the index is published last. A successful index therefore does not refer to unpublished shards. Cross-file all-or-nothing publication and recovery from process interruption are not yet guaranteed.
- `merge` deliberately emits canonical tensor-name order. It preserves bytes for each tensor name, but does not promise that the merged file is byte-identical to the pre-split source.

The strict index reader intentionally accepts only `metadata.total_size` and `weight_map`, with safe shard basenames. Third-party index extensions are rejected.

## Verification

```bash
moon fmt --check
moon check --target all --deny-warn
moon build --target all --deny-warn
moon test --target all --deny-warn
bash scripts/cli_e2e.sh
PYTHON_BIN=python3 bash scripts/python_interop.sh
moon info
```

The local suite includes strict prefix/JSON/layout cases, structural diff, deterministic plan/index/merge tests, and a native split→merge integration test that checks each named tensor's copied bytes. `scripts/cli_e2e.sh` builds a tiny original fixture at runtime and exercises all seven CLI commands, including `query` and `rename-prefix`.

The fixed external interoperability check uses `safetensors==0.6.2` and NumPy `2.2.6`:

```bash
bash scripts/python_interop.sh
```

It covers tiny U8 fixtures; the full dtype/large-file matrix is a documented follow-up.

## Design material

- [整体设计](docs/整体设计.md)
- [项目申请书](docs/项目申请书.md)

## License

Apache-2.0.
