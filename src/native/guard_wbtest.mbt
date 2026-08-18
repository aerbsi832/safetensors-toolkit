///|
fn guard_limits() -> @model.ParseLimits {
  @model.ParseLimits::default()
}

///|
fn guard_prefix(header : Bytes) -> Bytes {
  let bytes : Array[Byte] = []
  let length = header.length().to_uint64()
  for index in 0..<8 {
    bytes.push(((length >> (index * 8)) & 0xffUL).to_int().to_byte())
  }
  Bytes::from_array(bytes)
}

///|
fn guard_snapshot() -> @model.ValidatedFile raise @model.SftkError {
  @codec.validate_layout(
    @codec.parse_header(
      b"{\"x\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]}}",
      limits=guard_limits(),
    ),
    data_len=4UL,
  )
}

///|
async fn guard_write(
  path : String,
  snapshot : @model.ValidatedFile,
  data : Bytes,
) -> Unit {
  let header = @codec.encode_header(snapshot)
  @fs.write_file(
    path,
    guard_prefix(header) + header + data,
    create_mode=CreateOrTruncate,
    sync=Full,
  )
}

///|
/// The guard rejects a same-size payload replacement while preserving the
/// exact header and ranges.
async test "payload identity guard rejects same-size replacement" {
  @async.with_task_group() <| group => {
    let root = @fs.tmpdir(prefix="sftk-payload-guard")
    group.add_defer(() => @fs.rmdir(root, recursive=true))
    let path = root + "/source.safetensors"
    let snapshot = guard_snapshot()
    guard_write(path, snapshot, b"abcd")
    let file = open_regular(path, false)
    let before = payload_digest(
      file,
      path,
      container_data_offset(file, path),
      snapshot.data_len,
      8,
    )
    file.close()
    guard_write(path, snapshot, b"abXd")
    let mut rejected = false
    verify_source_unchanged(
      path,
      snapshot,
      before,
      @model.ExecutionOptions::default(),
    ) catch {
      @model.SftkError::Conflict(path=_, message~) =>
        rejected = message.contains("payload")
      _ => ()
    }
    assert_true(rejected)
  }
}

///|
/// Merge must not choose arbitrary metadata from the first shard when the
/// input set describes different model provenance.
test "merge rejects inconsistent shard metadata" {
  let left = @codec.validate_layout(
    @codec.parse_header(
      b"{\"__metadata__\":{\"format\":\"pt\"},\"a\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}",
      limits=guard_limits(),
    ),
    data_len=1UL,
  )
  let right = @codec.validate_layout(
    @codec.parse_header(
      b"{\"__metadata__\":{\"format\":\"mlx\"},\"b\":{\"dtype\":\"U8\",\"shape\":[1],\"data_offsets\":[0,1]}}",
      limits=guard_limits(),
    ),
    data_len=1UL,
  )
  let plan : @model.MergePlan = {
    entries: [
      {
        name: "a",
        shard_name: "one.safetensors",
        source_start: 0UL,
        source_end: 1UL,
        output_start: 0UL,
        output_end: 1UL,
      },
      {
        name: "b",
        shard_name: "two.safetensors",
        source_start: 0UL,
        source_end: 1UL,
        output_start: 1UL,
        output_end: 2UL,
      },
    ],
    output_data_bytes: 2UL,
  }
  let result : @model.ValidatedFile? = try
    Some(
      checked_merge_snapshot(plan, [
        ("one.safetensors", "one.safetensors", left),
        ("two.safetensors", "two.safetensors", right),
      ]),
    )
  catch {
    @model.SftkError::Conflict(path=_, message~) if message.contains("metadata") =>
      None
    _ => None
  } noraise {
    _ => None
  }
  let rejected = result is None
  assert_true(rejected)
}
