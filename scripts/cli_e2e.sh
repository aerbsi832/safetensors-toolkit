#!/usr/bin/env bash
set -euo pipefail

moon_bin="${MOON_BIN:-moon}"
workdir="$(mktemp -d)"
trap 'rm -rf "$workdir"' EXIT

source_file="$workdir/source.safetensors"
parts_dir="$workdir/parts"
merged_file="$workdir/merged.safetensors"
header='{"z":{"dtype":"U8","shape":[3],"data_offsets":[0,3]},"a":{"dtype":"U8","shape":[4],"data_offsets":[3,7]},"m":{"dtype":"U8","shape":[2],"data_offsets":[7,9]}}'
header_length="$(printf '%s' "$header" | wc -c)"
prefix="$(printf '\\x%02x\\x%02x\\x%02x\\x%02x\\x00\\x00\\x00\\x00' \
  "$((header_length & 255))" \
  "$(((header_length >> 8) & 255))" \
  "$(((header_length >> 16) & 255))" \
  "$(((header_length >> 24) & 255))")"
printf '%b%s%s' "$prefix" "$header" 'abcdefghi' > "$source_file"

"$moon_bin" run src/cmd/sftk -- validate "$source_file" --format json | grep -F '"valid":true' >/dev/null
"$moon_bin" run src/cmd/sftk -- info "$source_file" | grep -F 'tensors=3' >/dev/null
"$moon_bin" run src/cmd/sftk -- diff "$source_file" "$source_file" --format json | grep -F '"equal":true' >/dev/null
"$moon_bin" run src/cmd/sftk -- split "$source_file" --max-bytes 5B --output-dir "$parts_dir" --dry-run | grep -F 'shards=3' >/dev/null
"$moon_bin" run src/cmd/sftk -- split "$source_file" --max-bytes 5B --output-dir "$parts_dir" --write >/dev/null
test -f "$parts_dir/model.safetensors.index.json"
"$moon_bin" run src/cmd/sftk -- merge "$parts_dir/model.safetensors.index.json" --output "$merged_file" --dry-run | grep -F 'tensors=3' >/dev/null
"$moon_bin" run src/cmd/sftk -- merge "$parts_dir/model.safetensors.index.json" --output "$merged_file" --write >/dev/null
"$moon_bin" run src/cmd/sftk -- validate "$merged_file" --format json | grep -F '"valid":true' >/dev/null

set +e
"$moon_bin" run src/cmd/sftk -- diff "$source_file" "$merged_file" >/dev/null
diff_status="$?"
set -e
test "$diff_status" -eq 1

printf 'not-a-safetensors-file' > "$workdir/invalid.safetensors"
set +e
"$moon_bin" run src/cmd/sftk -- validate "$workdir/invalid.safetensors" >/dev/null 2>&1
invalid_status="$?"
"$moon_bin" run src/cmd/sftk -- unknown-command >/dev/null 2>&1
argument_status="$?"
set -e
test "$invalid_status" -eq 1
test "$argument_status" -eq 2
