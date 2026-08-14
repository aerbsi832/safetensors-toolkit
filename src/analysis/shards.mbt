///|
fn mapping_for(index : @model.ShardIndex, tensor_name : String) -> String? {
  for entry in index.weight_map {
    if entry.0 == tensor_name {
      return Some(entry.1)
    }
  }
  None
}

///|
fn named_file(
  shards : ArrayView[(String, @model.ValidatedFile)],
  name : String,
) -> @model.ValidatedFile? {
  for entry in shards {
    if entry.0 == name {
      return Some(entry.1)
    }
  }
  None
}

///|
fn add_issue(
  issues : Array[@model.ShardSetIssue],
  shard_name : String?,
  tensor_name : String?,
  message : String,
) -> Unit {
  issues.push({ shard_name, tensor_name, message })
}

///|
fn metadata_equal(
  left : ArrayView[@model.MetadataEntry],
  right : ArrayView[@model.MetadataEntry],
) -> Bool {
  if left.length() != right.length() {
    return false
  }
  for entry in left {
    let mut found = false
    for other in right {
      if entry.key == other.key && entry.value == other.value {
        found = true
      }
    }
    if !found {
      return false
    }
  }
  true
}

///|
fn optional_add(left : UInt64, right : UInt64) -> UInt64? {
  let maximum = UInt64::lnot(0UL)
  if left > maximum - right {
    None
  } else {
    Some(left + right)
  }
}

///|
/// Validate an indexed, named shard set before planning a merge.  This adds
/// human-readable coverage/metadata diagnostics on top of the strict merge
/// planner and never opens files or reads payload bytes.
pub fn validate_shard_set(
  index : @model.ShardIndex,
  shards : ArrayView[(String, @model.ValidatedFile)],
) -> @model.ShardSetValidation {
  let issues : Array[@model.ShardSetIssue] = []
  let seen_shards : Map[String, Bool] = Map([])
  let seen_tensors : Map[String, Bool] = Map([])
  let mut total_data_bytes = 0UL
  let mut baseline_metadata : Array[@model.MetadataEntry]? = None
  for entry in shards {
    let (shard_name, file) = entry
    if seen_shards.contains(shard_name) {
      add_issue(issues, Some(shard_name), None, "duplicate named shard input")
      continue
    }
    seen_shards[shard_name] = true
    match optional_add(total_data_bytes, file.data_len) {
      Some(total) => total_data_bytes = total
      None =>
        add_issue(
          issues,
          Some(shard_name),
          None,
          "aggregate shard size overflow",
        )
    }
    match baseline_metadata {
      Some(metadata) if !metadata_equal(metadata, file.header.metadata) =>
        add_issue(
          issues,
          Some(shard_name),
          None,
          "metadata differs from earlier shard",
        )
      None => baseline_metadata = Some(file.header.metadata.copy())
      _ => ()
    }
    for tensor in file.header.tensors {
      if seen_tensors.contains(tensor.name) {
        add_issue(
          issues,
          Some(shard_name),
          Some(tensor.name),
          "tensor occurs in multiple shard inputs",
        )
      }
      seen_tensors[tensor.name] = true
      match mapping_for(index, tensor.name) {
        Some(mapped) if mapped == shard_name => ()
        Some(mapped) =>
          add_issue(
            issues,
            Some(shard_name),
            Some(tensor.name),
            "index maps tensor to '" + mapped + "'",
          )
        None =>
          add_issue(
            issues,
            Some(shard_name),
            Some(tensor.name),
            "tensor is absent from index weight_map",
          )
      }
    }
  }
  for mapping in index.weight_map {
    let (tensor_name, shard_name) = mapping
    if named_file(shards, shard_name) is None {
      add_issue(
        issues,
        Some(shard_name),
        Some(tensor_name),
        "index references a missing shard input",
      )
    } else if !seen_tensors.contains(tensor_name) {
      add_issue(
        issues,
        Some(shard_name),
        Some(tensor_name),
        "index tensor is missing from shard header",
      )
    }
  }
  if total_data_bytes != index.total_size {
    add_issue(
      issues,
      None,
      None,
      "aggregate shard data size differs from index metadata.total_size",
    )
  }
  issues.sort_by((left, right) => {
    let left_shard = match left.shard_name {
      Some(value) => value
      None => ""
    }
    let right_shard = match right.shard_name {
      Some(value) => value
      None => ""
    }
    let by_shard = left_shard.lexical_compare(right_shard)
    if by_shard != 0 {
      return by_shard
    }
    let left_tensor = match left.tensor_name {
      Some(value) => value
      None => ""
    }
    let right_tensor = match right.tensor_name {
      Some(value) => value
      None => ""
    }
    let by_tensor = left_tensor.lexical_compare(right_tensor)
    if by_tensor != 0 {
      by_tensor
    } else {
      left.message.lexical_compare(right.message)
    }
  })
  { valid: issues.is_empty(), issues, total_data_bytes }
}

///|
/// Return unique index shard basenames in deterministic lexical order.
pub fn indexed_shard_names(index : @model.ShardIndex) -> Array[String] {
  let names : Array[String] = []
  for entry in index.weight_map {
    if !names.contains(entry.1) {
      names.push(entry.1)
    }
  }
  names.sort_by((left, right) => left.lexical_compare(right))
  names
}

///|
/// Return all weight-map tensor names that are not present in the supplied
/// named headers. This is a small focused diagnostic for partial downloads.
pub fn missing_indexed_tensors(
  index : @model.ShardIndex,
  shards : ArrayView[(String, @model.ValidatedFile)],
) -> Array[String] {
  let present : Map[String, Bool] = Map([])
  for entry in shards {
    for tensor in entry.1.header.tensors {
      present[tensor.name] = true
    }
  }
  let missing : Array[String] = []
  for mapping in index.weight_map {
    if !present.contains(mapping.0) {
      missing.push(mapping.0)
    }
  }
  missing.sort_by((left, right) => left.lexical_compare(right))
  missing
}

///|
/// Return whether a shard-set validation contains any inconsistency.
pub fn has_shard_set_issues(validation : @model.ShardSetValidation) -> Bool {
  !validation.valid
}
