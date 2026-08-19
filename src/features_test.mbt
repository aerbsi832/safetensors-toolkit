///|
fn feature_file() -> @model.ValidatedFile raise @model.SftkError {
  parse_and_validate(
    b"{\"__metadata__\":{\"format\":\"pt\",\"empty\":\"\"},\"encoder.weight\":{\"dtype\":\"F32\",\"shape\":[2,2],\"data_offsets\":[0,16]},\"encoder.bias\":{\"dtype\":\"F16\",\"shape\":[2],\"data_offsets\":[16,20]}}",
    data_len=20UL,
    limits=@model.ParseLimits::default(),
  )
}

///|
test "selector parser and header query return deterministic matches" {
  let file = feature_file()
  let query = @query.parse_selector("prefix=encoder.,dtype=F16|F32,rank=2")
  let result = query_tensors(file, query)
  assert_eq(result.matches.length(), 1)
  assert_eq(result.matches[0].tensor.name, "encoder.weight")
  assert_eq(
    @query.render_selector(query),
    "prefix=encoder.,dtype=F16|F32,rank=2",
  )
}

///|
test "layout and metadata policies surface header-only evidence" {
  let file = feature_file()
  let layout = summarize_layout(file)
  assert_eq(layout.tensor_count, 2)
  assert_eq(layout.total_bytes, 20UL)
  assert_eq(layout.largest_tensor.unwrap().tensor.name, "encoder.weight")
  let policy : @model.MetadataPolicy = {
    required_keys: ["format", "missing"],
    forbidden_keys: [],
    max_value_bytes: 10,
    allow_empty_values: false,
  }
  let validation = validate_metadata(file, policy)
  assert_false(validation.valid)
  assert_eq(validation.issues.length(), 2)
}

///|
test "header rename plan preserves all payload ranges" {
  let file = feature_file()
  let plan = plan_renames(file, [{ from: "encoder.weight", to: "model.weight" }])
  let renamed = apply_rename_plan(file, plan)
  assert_eq(renamed.data_len, file.data_len)
  assert_eq(renamed.header.tensors[0].name, "model.weight")
  assert_eq(renamed.header.tensors[0].data_end, 16UL)
  let collision = try
    plan_renames(file, [{ from: "encoder.weight", to: "encoder.bias" }])
  catch {
    @model.SftkError::InvalidTensor(..) => true
    _ => false
  } noraise {
    _ => false
  }
  assert_true(collision)
}

///|
test "split summary and named shard validation identify complete coverage" {
  let file = feature_file()
  let plan = plan_split(file, @model.ShardPolicy::new(16UL, 1))
  let summary = summarize_split(plan)
  assert_eq(summary.shard_count, 2)
  let first_header = @codec.validate_layout(
    { tensors: [file.header.tensors[0]], metadata: file.header.metadata },
    data_len=16UL,
  )
  let second_header = @codec.validate_layout(
    {
      tensors: [{ ..file.header.tensors[1], data_start: 0UL, data_end: 4UL }],
      metadata: file.header.metadata,
    },
    data_len=4UL,
  )
  let shards = [
    (plan.shards[0].name, first_header),
    (plan.shards[1].name, second_header),
  ]
  let validation = validate_shard_set(plan.index, shards)
  assert_true(validation.valid)
}
