///|
async fn verify_renamed_output(
  path : String,
  expected : @model.ValidatedFile,
) -> Unit raise @model.SftkError {
  let observed = inspect_file(path, @model.ParseLimits::default())
  if observed.data_len != expected.data_len ||
    observed.header.tensors.length() != expected.header.tensors.length() {
    raise @model.SftkError::Conflict(
      path~,
      message="staged renamed file failed structural verification",
    )
  }
  for index = 0; index < expected.header.tensors.length(); index = index + 1 {
    let wanted = expected.header.tensors[index]
    let actual = observed.header.tensors[index]
    if wanted.name != actual.name ||
      wanted.dtype != actual.dtype ||
      wanted.shape != actual.shape ||
      wanted.data_start != actual.data_start ||
      wanted.data_end != actual.data_end {
      raise @model.SftkError::Conflict(
        path~,
        message="staged renamed header does not match planned transformation",
      )
    }
  }
}

///|
/// Rewrite a SafeTensors file with a reviewed header-only rename plan. Payload
/// bytes are streamed exactly once in original physical order; only the header
/// changes, and publication uses the same staged atomic rename path as split.
pub async fn execute_rename(
  source_path : String,
  output_path : String,
  plan : @model.RenamePlan,
  options : @model.ExecutionOptions,
) -> Unit raise @model.SftkError {
  validate_execution_options(options)
  let source = inspect_file_with_policy(
    source_path,
    @model.ParseLimits::default(),
    options.follow_symlinks,
  )
  let renamed = @transform.apply_rename_plan(source, plan)
  if source_path == output_path && !options.overwrite {
    raise @model.SftkError::Conflict(
      path=output_path,
      message="in-place rename requires overwrite=true",
    )
  }
  let source_file = open_regular(source_path, options.follow_symlinks)
  defer source_file.close()
  let source_digest = payload_digest(
    source_file,
    source_path,
    container_data_offset(source_file, source_path),
    source.data_len,
    options.buffer_bytes,
  )
  let temporary = temporary_path(output_path)
  let output = create_output(temporary) catch {
    error => {
      cleanup_paths([temporary]) catch {
        _ => ()
      }
      raise error
    }
  }
  defer output.close()
  write_header(output, temporary, renamed) catch {
    error => {
      output.close()
      cleanup_paths([temporary]) catch {
        _ => ()
      }
      raise error
    }
  }
  let data_base = container_data_offset(source_file, source_path)
  copy_range(
    source_file,
    source_path,
    output,
    temporary,
    data_base,
    0UL,
    source.data_len,
    options.buffer_bytes,
  ) catch {
    error => {
      output.close()
      cleanup_paths([temporary]) catch {
        _ => ()
      }
      raise error
    }
  }
  output.sync() catch {
    _ => {
      cleanup_paths([temporary]) catch {
        _ => ()
      }
      raise @model.SftkError::IoFailure(
        operation="sync output",
        path=temporary,
        message="unable to sync staged renamed file",
      )
    }
  }
  verify_renamed_output(temporary, renamed) catch {
    error => {
      cleanup_paths([temporary]) catch {
        _ => ()
      }
      raise error
    }
  }
  verify_source_unchanged(source_path, source, source_digest, options) catch {
    error => {
      cleanup_paths([temporary]) catch {
        _ => ()
      }
      raise error
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

///|
/// Plan a source-file rename by exact prefix, validating the actual header
/// before any destination file is created.
pub async fn plan_prefix_rename_file(
  source_path : String,
  from_prefix : String,
  to_prefix : String,
  options : @model.ExecutionOptions,
) -> @model.RenamePlan raise @model.SftkError {
  validate_execution_options(options)
  let source = inspect_file_with_policy(
    source_path,
    @model.ParseLimits::default(),
    options.follow_symlinks,
  )
  @transform.plan_prefix_rename(source, from_prefix, to_prefix)
}
