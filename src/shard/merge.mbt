///|
/// Parse and validate the versioned supported shard-index JSON schema.
pub fn parse_index(
  index_bytes : Bytes,
  limits~ : @model.ParseLimits,
) -> @model.ShardIndex raise @model.SftkError {
  @codec.parse_shard_index(index_bytes, limits~)
}

///|
fn find_index_shard(index : @model.ShardIndex, name : String) -> String? {
  for pair in index.weight_map {
    if pair.0 == name {
      return Some(pair.1)
    }
  }
  None
}

///|
fn find_tensor(
  file : @model.ValidatedFile,
  name : String,
) -> @model.TensorInfo? {
  for tensor in file.header.tensors {
    if tensor.name == name {
      return Some(tensor)
    }
  }
  None
}

///|
fn contains_text(values : Array[String], wanted : String) -> Bool {
  for value in values {
    if value == wanted {
      return true
    }
  }
  false
}

///|
fn shard_names(
  index : @model.ShardIndex,
) -> Array[String] raise @model.SftkError {
  let names : Array[String] = []
  let tensor_names : Array[String] = []
  for pair in index.weight_map {
    if pair.0.is_empty() || pair.1.is_empty() {
      raise @model.SftkError::InvalidIndex(
        message="weight_map keys and values must be non-empty",
      )
    }
    if contains_text(tensor_names, pair.0) {
      raise @model.SftkError::InvalidIndex(
        message="weight_map contains a duplicate tensor name '" + pair.0 + "'",
      )
    }
    tensor_names.push(pair.0)
    if !contains_text(names, pair.1) {
      names.push(pair.1)
    }
  }
  names.sort_by((left, right) => left.lexical_compare(right))
  names
}

///|
fn file_for_shard(
  files : ArrayView[@model.ValidatedFile],
  shard_names : Array[String],
  wanted : String,
) -> @model.ValidatedFile raise @model.SftkError {
  for index, name in shard_names {
    if name == wanted {
      return files[index]
    }
  }
  raise @model.SftkError::InvalidIndex(
    message="index references unknown shard '" + wanted + "'",
  )
}

///|
/// Validate shard coverage and create canonical output offsets and ranges.
///
/// `files` must be passed in ascending lexical order of the distinct shard
/// basenames present in `index.weight_map`; native callers establish this from
/// their input paths before invoking the planner.
pub fn plan_merge(
  index : @model.ShardIndex,
  files : ArrayView[@model.ValidatedFile],
) -> @model.MergePlan raise @model.SftkError {
  if index.format != "huggingface-safetensors-index-v1" {
    raise @model.SftkError::InvalidIndex(
      message="unsupported shard index format",
    )
  }
  let names = shard_names(index)
  if files.length() != names.length() {
    raise @model.SftkError::InvalidIndex(
      message="provided shard snapshot count does not match the index",
    )
  }
  for file_index, file in files {
    let expected_shard = names[file_index]
    for tensor in file.header.tensors {
      match find_index_shard(index, tensor.name) {
        Some(actual_shard) if actual_shard == expected_shard => ()
        Some(_) =>
          raise @model.SftkError::InvalidIndex(
            message="tensor '" +
              tensor.name +
              "' is present in the wrong shard snapshot",
          )
        None =>
          raise @model.SftkError::InvalidIndex(
            message="shard snapshot contains tensor absent from index: '" +
              tensor.name +
              "'",
          )
      }
    }
  }
  let mappings = index.weight_map.copy()
  mappings.sort_by((left, right) => left.0.lexical_compare(right.0))
  let entries : Array[@model.MergeEntry] = []
  let mut output_start = 0UL
  for pair in mappings {
    let (name, shard_name) = pair
    let file = file_for_shard(files, names, shard_name)
    let tensor = match find_tensor(file, name) {
      Some(value) => value
      None =>
        raise @model.SftkError::InvalidIndex(
          message="index tensor '" +
            name +
            "' is absent from shard '" +
            shard_name +
            "'",
        )
    }
    let size = tensor.data_end - tensor.data_start
    let output_end = checked_add(
      output_start,
      size,
      "merged output offset for tensor '" + name + "'",
    )
    entries.push({
      name,
      shard_name,
      source_start: tensor.data_start,
      source_end: tensor.data_end,
      output_start,
      output_end,
    })
    output_start = output_end
  }
  if output_start != index.total_size {
    raise @model.SftkError::InvalidIndex(
      message="index metadata.total_size does not match covered tensor bytes",
    )
  }
  { entries, output_data_bytes: output_start }
}
