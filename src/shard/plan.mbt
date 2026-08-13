///|
fn checked_add(
  left : UInt64,
  right : UInt64,
  context : String,
) -> UInt64 raise @model.SftkError {
  let maximum = UInt64::lnot(0UL)
  if left > maximum - right {
    raise @model.SftkError::ArithmeticOverflow(context~)
  }
  left + right
}

///|
fn tensor_bytes(tensor : @model.TensorInfo) -> UInt64 {
  tensor.data_end - tensor.data_start
}

///|
fn compare_tensor_ranges(
  left : @model.TensorInfo,
  right : @model.TensorInfo,
) -> Int {
  if left.data_start < right.data_start {
    -1
  } else if left.data_start > right.data_start {
    1
  } else {
    left.name.lexical_compare(right.name)
  }
}

///|
fn pad_number(value : Int, width : Int) -> String {
  let text = value.to_string()
  if text.length() >= width {
    text
  } else {
    "0".repeat(width - text.length()) + text
  }
}

///|
fn shard_name(number : Int, total : Int) -> String {
  "model-" +
  pad_number(number, 5) +
  "-of-" +
  pad_number(total, 5) +
  ".safetensors"
}

///|
fn build_shard(
  source : Array[@model.TensorInfo],
  name : String,
) -> @model.PlannedShard raise @model.SftkError {
  let tensors : Array[@model.PlannedTensor] = []
  let mut output_start = 0UL
  for tensor in source {
    let output_end = checked_add(
      output_start,
      tensor_bytes(tensor),
      "shard output offset for tensor '" + tensor.name + "'",
    )
    tensors.push({
      name: tensor.name,
      source_start: tensor.data_start,
      source_end: tensor.data_end,
      output_start,
      output_end,
    })
    output_start = output_end
  }
  { name, tensors, data_bytes: output_start }
}

///|
/// Create a deterministic whole-tensor split plan; no tensor is sliced.
pub fn plan_split(
  file : @model.ValidatedFile,
  policy : @model.ShardPolicy,
) -> @model.ShardPlan raise @model.SftkError {
  if policy.max_bytes == 0UL || policy.max_tensors <= 0 {
    raise @model.SftkError::InvalidOptions(
      message="split policy requires positive max_bytes and max_tensors",
    )
  }
  let ordered = file.header.tensors.copy()
  ordered.sort_by(compare_tensor_ranges)
  let groups : Array[Array[@model.TensorInfo]] = []
  let mut current : Array[@model.TensorInfo] = []
  let mut current_bytes = 0UL
  for tensor in ordered {
    let size = tensor_bytes(tensor)
    let exceeds_count = current.length() >= policy.max_tensors
    let exceeds_bytes = !current.is_empty() &&
      (size > policy.max_bytes || current_bytes > policy.max_bytes - size)
    if exceeds_count || exceeds_bytes {
      groups.push(current)
      current = []
      current_bytes = 0UL
    }
    current.push(tensor)
    current_bytes = checked_add(
      current_bytes,
      size,
      "split shard size for tensor '" + tensor.name + "'",
    )
  }
  if !current.is_empty() {
    groups.push(current)
  }
  let shards : Array[@model.PlannedShard] = []
  let mapping : Array[(String, String)] = []
  for index, group in groups {
    let name = shard_name(index + 1, groups.length())
    let shard = build_shard(group, name)
    for tensor in shard.tensors {
      mapping.push((tensor.name, name))
    }
    shards.push(shard)
  }
  mapping.sort_by((left, right) => left.0.lexical_compare(right.0))
  {
    source_data_bytes: file.data_len,
    source_tensor_count: file.header.tensors.length(),
    shards,
    index: {
      format: "huggingface-safetensors-index-v1",
      total_size: file.data_len,
      weight_map: mapping,
    },
  }
}
