///|
fn source_tensor(
  source : @model.ValidatedFile,
  name : String,
) -> @model.TensorInfo? {
  for tensor in source.header.tensors {
    if tensor.name == name {
      return Some(tensor)
    }
  }
  None
}

///|
fn index_shard(index : @model.ShardIndex, tensor_name : String) -> String? {
  for pair in index.weight_map {
    if pair.0 == tensor_name {
      return Some(pair.1)
    }
  }
  None
}

///|
fn checked_split_snapshot(
  source : @model.ValidatedFile,
  shard : @model.PlannedShard,
  index : @model.ShardIndex,
) -> @model.ValidatedFile raise @model.SftkError {
  if !safe_basename(shard.name) {
    raise @model.SftkError::InvalidOptions(
      message="split plan contains an unsafe shard name",
    )
  }
  let tensors : Array[@model.TensorInfo] = []
  let mut expected_start = 0UL
  for planned in shard.tensors {
    if planned.output_start != expected_start ||
      planned.output_end < planned.output_start {
      raise @model.SftkError::InvalidOptions(
        message="split plan contains non-contiguous output ranges",
      )
    }
    if index_shard(index, planned.name) != Some(shard.name) {
      raise @model.SftkError::InvalidOptions(
        message="split plan shard and weight_map disagree for tensor '" +
          planned.name +
          "'",
      )
    }
    let source_tensor = match source_tensor(source, planned.name) {
      Some(value) => value
      None =>
        raise @model.SftkError::InvalidOptions(
          message="split plan references a tensor absent from source: '" +
            planned.name +
            "'",
        )
    }
    if source_tensor.data_start != planned.source_start ||
      source_tensor.data_end != planned.source_end {
      raise @model.SftkError::Conflict(
        path=planned.name,
        message="split plan source range no longer matches the source header",
      )
    }
    tensors.push({
      name: source_tensor.name,
      dtype: source_tensor.dtype,
      shape: source_tensor.shape,
      data_start: planned.output_start,
      data_end: planned.output_end,
      source_order: tensors.length(),
    })
    expected_start = planned.output_end
  }
  if expected_start != shard.data_bytes {
    raise @model.SftkError::InvalidOptions(
      message="split plan shard byte count disagrees with tensor ranges",
    )
  }
  let header : @model.HeaderSnapshot = {
    tensors,
    metadata: source.header.metadata,
  }
  @codec.validate_layout(header, data_len=shard.data_bytes)
}

///|
fn ensure_unique_shards(plan : @model.ShardPlan) -> Unit raise @model.SftkError {
  let names : Array[String] = []
  let tensors : Array[String] = []
  for shard in plan.shards {
    if names.contains(shard.name) {
      raise @model.SftkError::InvalidOptions(
        message="split plan repeats a shard name",
      )
    }
    names.push(shard.name)
    for planned in shard.tensors {
      if tensors.contains(planned.name) {
        raise @model.SftkError::InvalidOptions(
          message="split plan repeats a tensor",
        )
      }
      tensors.push(planned.name)
    }
  }
  if tensors.length() != plan.source_tensor_count ||
    plan.index.weight_map.length() != plan.source_tensor_count {
    raise @model.SftkError::InvalidOptions(
      message="split plan has incomplete tensor coverage",
    )
  }
}

///|
async fn write_split_shard(
  source_file : @fs.File,
  source_path : String,
  temporary : String,
  snapshot : @model.ValidatedFile,
  shard : @model.PlannedShard,
  options : @model.ExecutionOptions,
) -> Unit raise @model.SftkError {
  let output = create_output(temporary)
  defer output.close()
  write_header(output, temporary, snapshot)
  let data_base = container_data_offset(source_file, source_path)
  for planned in shard.tensors {
    copy_range(
      source_file,
      source_path,
      output,
      temporary,
      data_base,
      planned.source_start,
      planned.source_end,
      options.buffer_bytes,
    )
  }
  output.sync() catch {
    _ =>
      raise @model.SftkError::IoFailure(
        operation="sync output",
        path=temporary,
        message="unable to sync staged shard",
      )
  }
}

///|
async fn ensure_output_directory(
  path : String,
  follow_symlinks : Bool,
) -> Unit raise @model.SftkError {
  @fs.mkdir(path, recursive=true) catch {
    _ => ()
  }
  let kind = @fs.kind(path, follow_symlink=follow_symlinks) catch {
    _ =>
      raise @model.SftkError::IoFailure(
        operation="inspect output directory",
        path~,
        message="unable to inspect output directory",
      )
  }
  if kind != @fs.FileKind::Directory {
    raise @model.SftkError::InvalidOptions(
      message="output_dir must be a directory",
    )
  }
}

///|
/// Execute a reviewed split plan by bounded opaque range copying and atomic
/// publication of all shards followed by their index.
pub async fn execute_split(
  source_path : String,
  output_dir : String,
  plan : @model.ShardPlan,
  options : @model.ExecutionOptions,
) -> Unit raise @model.SftkError {
  validate_execution_options(options)
  ensure_output_directory(output_dir, options.follow_symlinks)
  let source = inspect_file_with_policy(
    source_path,
    @model.ParseLimits::default(),
    options.follow_symlinks,
  )
  if source.data_len != plan.source_data_bytes ||
    source.header.tensors.length() != plan.source_tensor_count {
    raise @model.SftkError::Conflict(
      path=source_path,
      message="source layout changed after split planning",
    )
  }
  ensure_unique_shards(plan)
  let source_file = open_regular(source_path, options.follow_symlinks)
  defer source_file.close()
  let source_digest = payload_digest(
    source_file,
    source_path,
    container_data_offset(source_file, source_path),
    source.data_len,
    options.buffer_bytes,
  )
  let destinations : Array[(String, @model.ValidatedFile, @model.PlannedShard)] = []
  for shard in plan.shards {
    let destination = join_path(output_dir, shard.name)
    let snapshot = checked_split_snapshot(source, shard, plan.index)
    if basename(destination) != shard.name {
      raise @model.SftkError::InvalidOptions(
        message="split destination escaped output directory",
      )
    }
    destinations.push((destination, snapshot, shard))
  }
  let index_destination = join_path(output_dir, "model.safetensors.index.json")
  if !options.overwrite {
    for _, item in destinations {
      let (destination, _, _) = item
      let exists = @fs.exists(destination) catch {
        _ =>
          raise @model.SftkError::IoFailure(
            operation="inspect output",
            path=destination,
            message="unable to check output destination",
          )
      }
      if exists {
        raise @model.SftkError::Conflict(
          path=destination,
          message="output already exists; pass force/overwrite explicitly",
        )
      }
    }
    let index_exists = @fs.exists(index_destination) catch {
      _ =>
        raise @model.SftkError::IoFailure(
          operation="inspect output",
          path=index_destination,
          message="unable to check index destination",
        )
    }
    if index_exists {
      raise @model.SftkError::Conflict(
        path=index_destination,
        message="index already exists; pass force/overwrite explicitly",
      )
    }
  }
  let staged : Array[(String, String)] = []
  for _, item in destinations {
    let (destination, snapshot, shard) = item
    let temporary = temporary_path(destination) catch {
      error => {
        cleanup_paths(staged.map(item => item.0)) catch {
          _ => ()
        }
        raise error
      }
    }
    write_split_shard(
      source_file, source_path, temporary, snapshot, shard, options,
    ) catch {
      error => {
        cleanup_paths(staged.map(item => item.0) + [temporary]) catch {
          _ => ()
        }
        raise error
      }
    }
    let checked = inspect_file(temporary, @model.ParseLimits::default()) catch {
      error => {
        cleanup_paths(staged.map(item => item.0) + [temporary]) catch {
          _ => ()
        }
        raise error
      }
    }
    if checked.data_len != shard.data_bytes ||
      checked.header.tensors.length() != shard.tensors.length() {
      cleanup_paths(staged.map(item => item.0) + [temporary]) catch {
        _ => ()
      }
      raise @model.SftkError::Conflict(
        path=temporary,
        message="staged shard failed structural verification",
      )
    }
    staged.push((temporary, destination))
  }
  let index_temporary = temporary_path(index_destination) catch {
    error => {
      cleanup_paths(staged.map(item => item.0)) catch {
        _ => ()
      }
      raise error
    }
  }
  @fs.write_file(
    index_temporary,
    @codec.encode_shard_index(plan.index),
    create_mode=CreateNew,
    sync=Full,
  ) catch {
    _ => {
      cleanup_paths(staged.map(item => item.0) + [index_temporary]) catch {
        _ => ()
      }
      raise @model.SftkError::IoFailure(
        operation="write index",
        path=index_temporary,
        message="unable to stage shard index",
      )
    }
  }
  // Re-open and hash the source immediately before any output becomes
  // visible.  This catches same-size payload changes that header/size checks
  // cannot detect.  It does not claim cross-process compare-and-swap.
  verify_source_unchanged(source_path, source, source_digest, options) catch {
    error => {
      cleanup_paths(staged.map(item => item.0) + [index_temporary]) catch {
        _ => ()
      }
      raise error
    }
  }
  let published : Array[String] = []
  for index = 0; index < staged.length(); index = index + 1 {
    let (temporary, destination) = staged[index]
    publish(temporary, destination, options.overwrite) catch {
      error => {
        let remaining = staged[index:].map(item => item.0)
        cleanup_paths(remaining + [index_temporary]) catch {
          _ => ()
        }
        if !options.overwrite {
          cleanup_paths(published) catch {
            _ => ()
          }
        }
        raise error
      }
    }
    published.push(destination)
  }
  // The index is deliberately published last: consumers following it never
  // observe a successful index that points to unpublished shards.
  publish(index_temporary, index_destination, options.overwrite) catch {
    error => {
      cleanup_paths([index_temporary]) catch {
        _ => ()
      }
      if !options.overwrite {
        cleanup_paths(published) catch {
          _ => ()
        }
      }
      raise error
    }
  }
}
