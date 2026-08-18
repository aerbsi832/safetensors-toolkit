///|
fn string_in(values : Array[String], wanted : String) -> Bool {
  for value in values {
    if value == wanted {
      return true
    }
  }
  false
}

///|
fn find_snapshot(
  snapshots : Array[(String, String, @model.ValidatedFile)],
  shard_name : String,
) -> (String, @model.ValidatedFile)? {
  for item in snapshots {
    let (name, path, file) = item
    if name == shard_name {
      return Some((path, file))
    }
  }
  None
}

///|
fn metadata_equal(
  left : Array[@model.MetadataEntry],
  right : Array[@model.MetadataEntry],
) -> Bool {
  if left.length() != right.length() {
    return false
  }
  for entry in left {
    let mut found = false
    for other in right {
      if entry.key == other.key && entry.value == other.value {
        found = true
        break
      }
    }
    if !found {
      return false
    }
  }
  true
}

///|
fn checked_merge_snapshot(
  plan : @model.MergePlan,
  snapshots : Array[(String, String, @model.ValidatedFile)],
) -> @model.ValidatedFile raise @model.SftkError {
  let tensors : Array[@model.TensorInfo] = []
  let mut output_start = 0UL
  for entry in plan.entries {
    if entry.output_start != output_start ||
      entry.output_end < entry.output_start {
      raise @model.SftkError::InvalidOptions(
        message="merge plan contains non-contiguous output ranges",
      )
    }
    let (_, source) = match find_snapshot(snapshots, entry.shard_name) {
      Some(value) => value
      None =>
        raise @model.SftkError::InvalidOptions(
          message="merge plan references an input shard that was not supplied",
        )
    }
    let tensor = match source_tensor(source, entry.name) {
      Some(value) => value
      None =>
        raise @model.SftkError::Conflict(
          path=entry.shard_name,
          message="merge source shard no longer contains planned tensor '" +
            entry.name +
            "'",
        )
    }
    if tensor.data_start != entry.source_start ||
      tensor.data_end != entry.source_end {
      raise @model.SftkError::Conflict(
        path=entry.shard_name,
        message="merge source range no longer matches its header",
      )
    }
    if tensors.contains(tensor) {
      raise @model.SftkError::InvalidOptions(
        message="merge plan repeats a source tensor",
      )
    }
    tensors.push({
      name: tensor.name,
      dtype: tensor.dtype,
      shape: tensor.shape,
      data_start: entry.output_start,
      data_end: entry.output_end,
      source_order: tensors.length(),
    })
    output_start = entry.output_end
  }
  if output_start != plan.output_data_bytes {
    raise @model.SftkError::InvalidOptions(
      message="merge plan byte count disagrees with output ranges",
    )
  }
  let metadata = match snapshots {
    [(_, _, first), ..] => {
      for index in 1..<snapshots.length() {
        let (shard_name, _, file) = snapshots[index]
        if !metadata_equal(first.header.metadata, file.header.metadata) {
          raise @model.SftkError::Conflict(
            path=shard_name,
            message="merge shards have inconsistent metadata",
          )
        }
      }
      first.header.metadata
    }
    [] => []
  }
  let header : @model.HeaderSnapshot = { tensors, metadata }
  @codec.validate_layout(header, data_len=plan.output_data_bytes)
}

///|
async fn write_merged_file(
  temporary : String,
  plan : @model.MergePlan,
  snapshot : @model.ValidatedFile,
  inputs : Array[(String, String, @fs.File)],
  options : @model.ExecutionOptions,
) -> Unit raise @model.SftkError {
  let output = create_output(temporary)
  defer output.close()
  write_header(output, temporary, snapshot)
  for entry in plan.entries {
    let mut selected : (String, @fs.File)? = None
    for item in inputs {
      let (name, path, file) = item
      if name == entry.shard_name {
        selected = Some((path, file))
        break
      }
    }
    let (path, file) = match selected {
      Some(value) => value
      None =>
        raise @model.SftkError::InvalidOptions(
          message="merge input handle is missing a required shard",
        )
    }
    copy_range(
      file,
      path,
      output,
      temporary,
      container_data_offset(file, path),
      entry.source_start,
      entry.source_end,
      options.buffer_bytes,
    )
  }
  output.sync() catch {
    _ =>
      raise @model.SftkError::IoFailure(
        operation="sync output",
        path=temporary,
        message="unable to sync staged merge output",
      )
  }
}

///|
/// Execute a validated merge plan without decoding tensor values.
pub async fn execute_merge(
  shard_paths : ArrayView[String],
  output_path : String,
  plan : @model.MergePlan,
  options : @model.ExecutionOptions,
) -> Unit raise @model.SftkError {
  validate_execution_options(options)
  if shard_paths.length() == 0 && !plan.entries.is_empty() {
    raise @model.SftkError::InvalidOptions(
      message="merge requires input shard paths",
    )
  }
  if !options.overwrite {
    let exists = @fs.exists(output_path) catch {
      _ =>
        raise @model.SftkError::IoFailure(
          operation="inspect output",
          path=output_path,
          message="unable to check output destination",
        )
    }
    if exists {
      raise @model.SftkError::Conflict(
        path=output_path,
        message="output already exists; pass force/overwrite explicitly",
      )
    }
  }
  let required : Array[String] = []
  for entry in plan.entries {
    if !safe_basename(entry.shard_name) {
      raise @model.SftkError::InvalidOptions(
        message="merge plan contains an unsafe shard name",
      )
    }
    if !string_in(required, entry.shard_name) {
      required.push(entry.shard_name)
    }
  }
  if required.length() != shard_paths.length() {
    raise @model.SftkError::InvalidOptions(
      message="input shard path count does not match merge plan",
    )
  }
  let snapshots : Array[(String, String, @model.ValidatedFile)] = []
  for path in shard_paths {
    let name = basename(path)
    if !string_in(required, name) || !(find_snapshot(snapshots, name) is None) {
      raise @model.SftkError::InvalidOptions(
        message="input shard paths do not uniquely match merge plan shard basenames",
      )
    }
    snapshots.push(
      (
        name,
        path,
        inspect_file_with_policy(
          path,
          @model.ParseLimits::default(),
          options.follow_symlinks,
        ),
      ),
    )
  }
  let snapshot = checked_merge_snapshot(plan, snapshots)
  let inputs : Array[(String, String, @fs.File)] = []
  for item in snapshots {
    let (name, path, _) = item
    inputs.push((name, path, open_regular(path, options.follow_symlinks)))
  }
  defer (for item in inputs {
    let (_, _, file) = item
    file.close()
  })
  let source_guards : Array[
    (String, String, @model.ValidatedFile, PayloadDigest),
  ] = []
  for index in 0..<snapshots.length() {
    let (name, path, expected) = snapshots[index]
    let (_, _, file) = inputs[index]
    source_guards.push(
      (
        name,
        path,
        expected,
        payload_digest(
          file,
          path,
          container_data_offset(file, path),
          expected.data_len,
          options.buffer_bytes,
        ),
      ),
    )
  }
  @fs.mkdir(parent_path(output_path), recursive=true) catch {
    _ => ()
  }
  let temporary = temporary_path(output_path)
  write_merged_file(temporary, plan, snapshot, inputs, options) catch {
    error => {
      cleanup_paths([temporary]) catch {
        _ => ()
      }
      raise error
    }
  }
  let checked = inspect_file(temporary, @model.ParseLimits::default()) catch {
    error => {
      cleanup_paths([temporary]) catch {
        _ => ()
      }
      raise error
    }
  }
  if checked.data_len != plan.output_data_bytes ||
    checked.header.tensors.length() != plan.entries.length() {
    cleanup_paths([temporary]) catch {
      _ => ()
    }
    raise @model.SftkError::Conflict(
      path=temporary,
      message="staged merge output failed structural verification",
    )
  }
  for item in source_guards {
    let (_, path, expected, digest) = item
    verify_source_unchanged(path, expected, digest, options) catch {
      error => {
        cleanup_paths([temporary]) catch {
          _ => ()
        }
        raise error
      }
    }
  }
  publish(temporary, output_path, options.overwrite) catch {
    error => {
      cleanup_paths([temporary]) catch {
        _ => ()
      }
      raise error
    }
  }
}
