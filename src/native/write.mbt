///|
fn header_prefix(header : Bytes) -> Bytes {
  let length = header.length().to_uint64()
  let bytes : Array[Byte] = []
  for index in 0..<8 {
    bytes.push(((length >> (index * 8)) & 0xffUL).to_int().to_byte())
  }
  Bytes::from_array(bytes)
}

///|
async fn write_bytes(
  file : @fs.File,
  path : String,
  data : Bytes,
) -> Unit raise @model.SftkError {
  file.write(data) catch {
    _ =>
      raise @model.SftkError::IoFailure(
        operation="write",
        path~,
        message="unable to write output bytes",
      )
  }
}

///|
async fn copy_range(
  input : @fs.File,
  input_path : String,
  output : @fs.File,
  output_path : String,
  data_base : UInt64,
  start : UInt64,
  end : UInt64,
  buffer_bytes : Int,
) -> Unit raise @model.SftkError {
  let mut offset = start
  while offset < end {
    let remaining = end - offset
    let chunk = if remaining < buffer_bytes.to_uint64() {
      remaining.to_int()
    } else {
      buffer_bytes
    }
    let absolute = data_base + offset
    if absolute < offset {
      raise @model.SftkError::ArithmeticOverflow(
        context="source file byte offset",
      )
    }
    let bytes = read_at(input, input_path, chunk, absolute)
    write_bytes(output, output_path, bytes)
    offset += chunk.to_uint64()
  }
}

///|
async fn create_output(path : String) -> @fs.File raise @model.SftkError {
  @fs.open(path, mode=WriteOnly, create_mode=CreateNew, sync=Full) catch {
    _ =>
      raise @model.SftkError::IoFailure(
        operation="create output",
        path~,
        message="unable to create output file",
      )
  }
}

///|
async fn write_header(
  output : @fs.File,
  output_path : String,
  file : @model.ValidatedFile,
) -> Unit raise @model.SftkError {
  let header = @codec.encode_header(file)
  write_bytes(output, output_path, header_prefix(header))
  write_bytes(output, output_path, header)
}

///|
async fn publish(
  temporary : String,
  destination : String,
  overwrite : Bool,
) -> Unit raise @model.SftkError {
  if !overwrite {
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
  @fs.rename(temporary, destination, replace=overwrite) catch {
    _ =>
      raise @model.SftkError::IoFailure(
        operation="publish output",
        path=destination,
        message="unable to atomically publish output",
      )
  }
}

///|
/// Best-effort cleanup for staging files.  Cleanup is deliberately separate
/// from the primary error so a failed copy/publish never masks its cause.
async fn cleanup_paths(paths : Array[String]) -> Unit {
  for path in paths {
    @fs.remove(path) catch {
      _ => ()
    }
  }
}
