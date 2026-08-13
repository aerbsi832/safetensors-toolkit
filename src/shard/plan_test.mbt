///|
fn shard_limits() -> @model.ParseLimits {
  @model.ParseLimits::default()
}

///|
fn validated(
  header : Bytes,
  data_len : UInt64,
) -> @model.ValidatedFile raise @model.SftkError {
  @codec.validate_layout(
    @codec.parse_header(header, limits=shard_limits()),
    data_len~,
  )
}

///|
test "split plans whole tensors in source-range order with stable shard names" {
  let source = validated(
    b"{\"z\":{\"dtype\":\"U8\",\"shape\":[3],\"data_offsets\":[0,3]},\"a\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[3,7]},\"m\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[7,9]}}",
    9UL,
  )
  let plan = plan_split(source, @model.ShardPolicy::new(5UL, 2))
  assert_eq(plan.shards.length(), 3)
  assert_eq(plan.shards[0].name, "model-00001-of-00003.safetensors")
  assert_eq(plan.shards[0].tensors[0].name, "z")
  assert_eq(plan.shards[1].tensors[0].name, "a")
  assert_eq(plan.shards[2].tensors[0].name, "m")
  assert_eq(plan.index.weight_map.map(pair => pair.0), ["a", "m", "z"])
  let decoded = parse_index(
    @codec.encode_shard_index(plan.index),
    limits=shard_limits(),
  )
  assert_eq(decoded.weight_map, plan.index.weight_map)
}

///|
test "merge verifies complete index coverage and canonicalizes output offsets" {
  let index : @model.ShardIndex = {
    format: "huggingface-safetensors-index-v1",
    total_size: 9UL,
    weight_map: [
      ("a", "model-00002-of-00003.safetensors"),
      ("m", "model-00003-of-00003.safetensors"),
      ("z", "model-00001-of-00003.safetensors"),
    ],
  }
  let first = validated(
    b"{\"z\":{\"dtype\":\"U8\",\"shape\":[3],\"data_offsets\":[0,3]}}", 3UL,
  )
  let second = validated(
    b"{\"a\":{\"dtype\":\"U8\",\"shape\":[4],\"data_offsets\":[0,4]}}", 4UL,
  )
  let third = validated(
    b"{\"m\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[0,2]}}", 2UL,
  )
  let plan = plan_merge(index, [first, second, third])
  assert_eq(plan.entries.map(entry => entry.name), ["a", "m", "z"])
  assert_eq(plan.entries.map(entry => entry.output_start), [0UL, 4UL, 6UL])
  assert_eq(plan.output_data_bytes, 9UL)
}

///|
test "merge rejects duplicate tensor names in manually constructed indexes" {
  let invalid : @model.ShardIndex = {
    format: "huggingface-safetensors-index-v1",
    total_size: 0UL,
    weight_map: [("a", "one.safetensors"), ("a", "two.safetensors")],
  }
  let rejected = try plan_merge(invalid, []) catch {
    @model.SftkError::InvalidIndex(message=_) => true
    _ => false
  } noraise {
    _ => false
  }
  assert_true(rejected)
}
