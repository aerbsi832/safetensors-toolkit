///|
/// Return the exact element count, checking every unsigned multiplication.
fn element_count(tensor : @model.TensorInfo) -> UInt64 raise @model.SftkError {
  let maximum = UInt64::lnot(0UL)
  let mut count = 1UL
  for dimension in tensor.shape {
    if dimension != 0UL && count > maximum / dimension {
      raise @model.SftkError::ArithmeticOverflow(
        context="shape product for tensor '" + tensor.name + "'",
      )
    }
    count = count * dimension
  }
  count
}

///|
/// Return the exact byte size for a tensor without allocating data.
fn tensor_byte_length(
  tensor : @model.TensorInfo,
) -> UInt64 raise @model.SftkError {
  let maximum = UInt64::lnot(0UL)
  let elements = element_count(tensor)
  let bits = tensor.dtype.bit_width()
  if elements != 0UL && elements > maximum / bits {
    raise @model.SftkError::ArithmeticOverflow(
      context="bit size for tensor '" + tensor.name + "'",
    )
  }
  let total_bits = elements * bits
  if total_bits % 8UL != 0UL {
    raise @model.SftkError::InvalidTensor(
      name=tensor.name,
      message="dtype width and shape do not occupy a whole number of bytes",
    )
  }
  total_bits / 8UL
}

///|
/// Return checked logical element count for a validated or header snapshot
/// tensor.  This remains header-only and never reads payload bytes.
pub fn tensor_element_count(
  tensor : @model.TensorInfo,
) -> UInt64 raise @model.SftkError {
  element_count(tensor)
}

///|
/// Return checked storage length implied by dtype and shape.
pub fn tensor_storage_bytes(
  tensor : @model.TensorInfo,
) -> UInt64 raise @model.SftkError {
  tensor_byte_length(tensor)
}

///|
fn compare_ranges(left : @model.TensorInfo, right : @model.TensorInfo) -> Int {
  let by_start = left.data_start.compare(right.data_start)
  if by_start != 0 {
    by_start
  } else {
    let by_end = left.data_end.compare(right.data_end)
    if by_end != 0 {
      by_end
    } else {
      left.name.compare(right.name)
    }
  }
}

///|
/// Check dtype byte sizes, shape arithmetic, ranges, holes, and final length.
pub fn validate_layout(
  header : @model.HeaderSnapshot,
  data_len~ : UInt64,
) -> @model.ValidatedFile raise @model.SftkError {
  let ordered = header.tensors.copy()
  ordered.sort_by(compare_ranges)
  let mut cursor = 0UL
  for tensor in ordered {
    if tensor.data_end < tensor.data_start {
      raise @model.SftkError::InvalidTensor(
        name=tensor.name,
        message="data_offsets end precedes start",
      )
    }
    let actual = tensor.data_end - tensor.data_start
    let expected = tensor_byte_length(tensor)
    if actual != expected {
      raise @model.SftkError::InvalidTensor(
        name=tensor.name,
        message="data range length does not match dtype and shape",
      )
    }
    if tensor.data_start != cursor {
      raise @model.SftkError::InvalidLayout(
        message="tensor ranges must be contiguous from offset zero without overlap or holes",
      )
    }
    cursor = tensor.data_end
  }
  if cursor != data_len {
    raise @model.SftkError::InvalidLayout(
      message="tensor ranges do not cover the complete data buffer",
    )
  }
  { header, data_len }
}
