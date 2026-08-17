///|
fn validate_execution_options(
  options : @model.ExecutionOptions,
) -> Unit raise @model.SftkError {
  if options.buffer_bytes <= 0 || options.buffer_bytes > 64 * 1024 * 1024 {
    raise @model.SftkError::InvalidOptions(
      message="buffer_bytes must be between 1 and 67108864",
    )
  }
}

///|
async fn open_regular(
  path : String,
  follow_symlinks : Bool,
) -> @fs.File raise @model.SftkError {
  let kind = @fs.kind(path, follow_symlink=follow_symlinks) catch {
    _ =>
      raise @model.SftkError::IoFailure(
        operation="inspect",
        path~,
        message="unable to inspect input path",
      )
  }
  if kind != @fs.FileKind::Regular {
    raise @model.SftkError::IoFailure(
      operation="inspect",
      path~,
      message="input must be a regular file under the selected symlink policy",
    )
  }
  @fs.open(path, mode=ReadOnly) catch {
    _ =>
      raise @model.SftkError::IoFailure(
        operation="open",
        path~,
        message="unable to open input file",
      )
  }
}

///|
fn checked_file_offset(
  value : UInt64,
  context : String,
) -> Int64 raise @model.SftkError {
  if value > 0x7fffffffffffffffUL {
    raise @model.SftkError::ArithmeticOverflow(context~)
  }
  value.reinterpret_as_int64()
}

///|
async fn read_at(
  file : @fs.File,
  path : String,
  length : Int,
  position : UInt64,
) -> Bytes raise @model.SftkError {
  file.read_exactly_at(
    length,
    position=checked_file_offset(position, "native file offset"),
  ) catch {
    _ =>
      raise @model.SftkError::IoFailure(
        operation="read",
        path~,
        message="unexpected EOF or unreadable input range",
      )
  }
}

///|
/// A bounded-memory payload digest.  This is deliberately a private
/// corruption/race guard rather than a cryptographic authenticity claim.
priv struct PayloadDigest {
  length : UInt64
  first : UInt64
  second : UInt64
} derive(Eq)

///|
/// Hash the opaque data buffer in bounded chunks.  This is intentionally a
/// payload-only guard: the header is checked separately by
/// `verify_source_unchanged`, while the digest prevents a same-size payload
/// replacement from being silently copied after planning.
async fn payload_digest(
  file : @fs.File,
  path : String,
  data_base : UInt64,
  data_len : UInt64,
  buffer_bytes : Int,
) -> PayloadDigest raise @model.SftkError {
  let mut first = 1469598103934665603UL
  let mut second = 1099511628211UL
  let mut offset = 0UL
  while offset < data_len {
    let remaining = data_len - offset
    let chunk = if remaining < buffer_bytes.to_uint64() {
      remaining.to_int()
    } else {
      buffer_bytes
    }
    let absolute = data_base + offset
    if absolute < data_base {
      raise @model.SftkError::ArithmeticOverflow(
        context="payload digest file offset",
      )
    }
    for byte in read_at(file, path, chunk, absolute) {
      let value = byte.to_int().to_uint64()
      first = (first ^ value) * 1099511628211UL
      second = (second ^ (value + 17UL)) * 14029467366897019727UL
    }
    offset += chunk.to_uint64()
  }
  { length: data_len, first, second }
}

///|
/// Verify that a path still denotes the validated source and that its opaque
/// payload is unchanged.  This is a pre-publication guard, not an
/// OS-level compare-and-swap: another writer can still race after this check.
async fn verify_source_unchanged(
  path : String,
  expected : @model.ValidatedFile,
  expected_digest : PayloadDigest,
  options : @model.ExecutionOptions,
) -> Unit raise @model.SftkError {
  let observed = inspect_file_with_policy(
    path,
    @model.ParseLimits::default(),
    options.follow_symlinks,
  )
  if observed != expected {
    raise @model.SftkError::Conflict(
      path~,
      message="source header or layout changed during transformation",
    )
  }
  let file = open_regular(path, options.follow_symlinks)
  defer file.close()
  let digest = payload_digest(
    file,
    path,
    container_data_offset(file, path),
    observed.data_len,
    options.buffer_bytes,
  )
  if digest != expected_digest {
    raise @model.SftkError::Conflict(
      path~,
      message="source payload changed during transformation",
    )
  }
}

///|
async fn container_data_offset(
  file : @fs.File,
  path : String,
) -> UInt64 raise @model.SftkError {
  let prefix = read_at(file, path, 8, 0UL)
  let header_length = @codec.decode_header_length(
    prefix,
    limits=@model.ParseLimits::default(),
  )
  let offset = 8UL + header_length
  if offset < header_length {
    raise @model.SftkError::ArithmeticOverflow(context="container data offset")
  }
  offset
}

///|
/// Read only the bounded prefix/header under the requested symlink policy.
async fn inspect_file_with_policy(
  path : String,
  limits : @model.ParseLimits,
  follow_symlinks : Bool,
) -> @model.ValidatedFile raise @model.SftkError {
  let file = open_regular(path, follow_symlinks)
  defer file.close()
  let signed_size = file.size() catch {
    _ =>
      raise @model.SftkError::IoFailure(
        operation="stat",
        path~,
        message="unable to determine input size",
      )
  }
  if signed_size < 8L {
    raise @model.SftkError::Truncated(
      phase="file prefix",
      expected=8UL,
      actual=if signed_size < 0L {
        0UL
      } else {
        signed_size.reinterpret_as_uint64()
      },
    )
  }
  let size = signed_size.reinterpret_as_uint64()
  let prefix = read_at(file, path, 8, 0UL)
  let header_length = @codec.decode_header_length(prefix, limits~)
  let header_end = 8UL + header_length
  if header_end < header_length || header_end > size {
    raise @model.SftkError::Truncated(
      phase="header",
      expected=header_end,
      actual=size,
    )
  }
  let header = read_at(file, path, header_length.to_int(), 8UL)
  let snapshot = @codec.parse_header(header, limits~)
  @codec.validate_layout(snapshot, data_len=size - header_end)
}

///|
/// Read only a bounded prefix/header and validate the file against its size.
pub async fn inspect_file(
  path : String,
  limits : @model.ParseLimits,
) -> @model.ValidatedFile raise @model.SftkError {
  inspect_file_with_policy(path, limits, false)
}
