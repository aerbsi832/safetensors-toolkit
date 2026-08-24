///|
fn test_path(root : String, name : String) -> String {
  root + "/" + name
}

///|
fn test_limits() -> @model.ParseLimits {
  @model.ParseLimits::default()
}

///|
fn test_header_prefix(header : Bytes) -> Bytes {
  let bytes : Array[Byte] = []
  let length = header.length().to_uint64()
  for index in 0..<8 {
    bytes.push(((length >> (index * 8)) & 0xffUL).to_int().to_byte())
  }
  Bytes::from_array(bytes)
}

///|
fn source_snapshot() -> @model.ValidatedFile raise @model.SftkError {
  @codec.validate_layout(
    @codec.parse_header(
      b"{\"z\":{\"dtype\":\"U8\",\"shape\":[3],\"data_offsets\":[0,3]},\"a\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[3,7]},\"m\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[7,9]}}",
      limits=test_limits(),
    ),
    data_len=9UL,
  )
}

///|
async fn write_container(
  path : String,
  snapshot : @model.ValidatedFile,
  data : Bytes,
) -> Unit {
  let header = @codec.encode_header(snapshot)
  assert_true(header.length() < 1024)
  assert_eq(test_header_prefix(header)[0].to_int(), header.length() & 0xff)
  assert_eq(
    @codec.decode_header_length(
      test_header_prefix(header),
      limits=test_limits(),
    ),
    header.length().to_uint64(),
  )
  @fs.write_file(
    path,
    test_header_prefix(header) + header + data,
    create_mode=CreateOrTruncate,
    sync=Full,
  )
}

///|
async test "native inspect reads only header and split then merge preserves named data" {
  @async.with_task_group() <| group => {
    let root = @fs.tmpdir(prefix="sftk-native")
    group.add_defer(() => @fs.rmdir(root, recursive=true))
    let source = test_path(root, "source.safetensors")
    let snapshot = source_snapshot()
    write_container(source, snapshot, b"abcdefghi")
    let inspected = inspect_file(source, test_limits())
    assert_eq(inspected.data_len, 9UL)
    let split = @shard.plan_split(inspected, @model.ShardPolicy::new(5UL, 2))
    let output_dir = test_path(root, "parts")
    execute_split(source, output_dir, split, @model.ExecutionOptions::default())
    let index_path = test_path(output_dir, "model.safetensors.index.json")
    let index_bytes = @fs.read_file(index_path).binary()
    let index = @shard.parse_index(index_bytes, limits=test_limits())
    assert_eq(index.weight_map, split.index.weight_map)
    let shard_paths = split.shards.map(shard => {
      test_path(output_dir, shard.name)
    })
    let shard_files : Array[@model.ValidatedFile] = []
    for path in shard_paths {
      shard_files.push(inspect_file(path, test_limits()))
    }
    let merge_plan = @shard.plan_merge(index, shard_files)
    let merged = test_path(root, "merged.safetensors")
    execute_merge(
      shard_paths,
      merged,
      merge_plan,
      @model.ExecutionOptions::default(),
    )
    let merged_file = inspect_file(merged, test_limits())
    assert_eq(merged_file.header.tensors.map(tensor => tensor.name), [
      "a", "m", "z",
    ])
    let input = @fs.open(merged, mode=ReadOnly)
    defer input.close()
    let header = @codec.encode_header(merged_file)
    let data_offset = 8L + header.length().to_int64()
    assert_eq(input.read_exactly_at(4, position=data_offset), b"defg")
    assert_eq(input.read_exactly_at(2, position=data_offset + 4L), b"hi")
    assert_eq(input.read_exactly_at(3, position=data_offset + 6L), b"abc")
  }
}

///|
async test "native prefix rename publishes a canonical multi-tensor header" {
  @async.with_task_group() <| group => {
    let root = @fs.tmpdir(prefix="sftk-rename")
    group.add_defer(() => @fs.rmdir(root, recursive=true))
    let source = test_path(root, "source.safetensors")
    write_container(source, source_snapshot(), b"abcdefghi")
    let output = test_path(root, "renamed.safetensors")
    let options = @model.ExecutionOptions::default()
    let plan = plan_prefix_rename_file(source, "a", "encoder.a", options)
    execute_rename(source, output, plan, options)
    let observed = inspect_file(output, test_limits())
    assert_eq(
      observed.header.tensors.map(tensor => {
        (tensor.name, tensor.data_start, tensor.data_end)
      }),
      [("encoder.a", 3UL, 7UL), ("m", 7UL, 9UL), ("z", 0UL, 3UL)],
    )
  }
}

///|
async test "native inspect reports a truncated prefix" {
  @async.with_task_group() <| group => {
    let root = @fs.tmpdir(prefix="sftk-truncated")
    group.add_defer(() => @fs.rmdir(root, recursive=true))
    let path = test_path(root, "short.safetensors")
    @fs.write_file(path, b"tiny", create_mode=CreateOrTruncate)
    let rejected = try inspect_file(path, test_limits()) catch {
      @model.SftkError::Truncated(phase=_, expected=_, actual=_) => true
      _ => false
    } noraise {
      _ => false
    }
    assert_true(rejected)
  }
}

///|
async test "native inspect handles a sparse large payload without loading it" {
  @async.with_task_group() <| group => {
    let root = @fs.tmpdir(prefix="sftk-sparse")
    group.add_defer(() => @fs.rmdir(root, recursive=true))
    let path = test_path(root, "large.safetensors")
    let data_len = 16UL * 1024UL * 1024UL
    let header = b"{\"x\":{\"dtype\":\"U8\",\"shape\":[16777216],\"data_offsets\":[0,16777216]}}"
    let prefix = test_header_prefix(header)
    @fs.write_file(
      path,
      prefix + header,
      create_mode=CreateOrTruncate,
      sync=Full,
    )
    let file = @fs.open(path, mode=ReadWrite)
    file.write_at(
      b"\x00",
      position=(prefix.length() + header.length()).to_int64() +
        data_len.reinterpret_as_int64() -
        1L,
    )
    file.close()
    let inspected = inspect_file(path, test_limits())
    assert_eq(inspected.data_len, data_len)
    assert_eq(inspected.header.tensors.length(), 1)
  }
}

///|
async test "split cleans staged shards when a later temporary path cannot be allocated" {
  @async.with_task_group() <| group => {
    let root = @fs.tmpdir(prefix="sftk-cleanup")
    group.add_defer(() => @fs.rmdir(root, recursive=true))
    let source = test_path(root, "source.safetensors")
    let snapshot = source_snapshot()
    write_container(source, snapshot, b"abcdefghi")
    let split = @shard.plan_split(
      inspect_file(source, test_limits()),
      @model.ShardPolicy::new(5UL, 2),
    )
    assert_true(split.shards.length() >= 2)
    let output_dir = test_path(root, "parts")
    @fs.mkdir(output_dir, recursive=true)
    let blocked_name = split.shards[1].name
    for counter = 0; counter < 64; counter = counter + 1 {
      @fs.write_file(
        test_path(output_dir, blocked_name + ".sftk-tmp-" + counter.to_string()),
        b"block",
        create_mode=CreateNew,
      )
    }
    let rejected = try
      execute_split(
        source,
        output_dir,
        split,
        @model.ExecutionOptions::default(),
      )
    catch {
      @model.SftkError::Conflict(path=_, message=_) => true
      _ => false
    } noraise {
      _ => false
    }
    assert_true(rejected)
    assert_true(
      !@fs.exists(test_path(output_dir, split.shards[0].name + ".sftk-tmp-0")),
    )
  }
}
