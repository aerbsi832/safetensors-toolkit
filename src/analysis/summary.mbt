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
fn dtype_summary_index(
  summaries : ArrayView[@model.DtypeSummary],
  dtype : @model.Dtype,
) -> Int? {
  for index = 0; index < summaries.length(); index = index + 1 {
    if summaries[index].dtype == dtype {
      return Some(index)
    }
  }
  None
}

///|
fn bigger(candidate : @model.TensorMatch, current : @model.TensorMatch) -> Bool {
  candidate.byte_count > current.byte_count ||
  (
    candidate.byte_count == current.byte_count &&
    candidate.tensor.name.lexical_compare(current.tensor.name) < 0
  )
}

///|
/// Compute deterministic dtype/rank/size aggregates from a validated header.
/// The result is useful for model inventory dashboards and avoids loading any
/// tensor payload bytes.
pub fn summarize_layout(
  file : @model.ValidatedFile,
) -> @model.LayoutSummary raise @model.SftkError {
  let dtype_summaries : Array[@model.DtypeSummary] = []
  let mut scalar_count = 0
  let mut max_rank = 0
  let mut total_elements = 0UL
  let mut total_bytes = 0UL
  let mut largest_tensor : @model.TensorMatch? = None
  for tensor in file.header.tensors {
    let elements = @codec.tensor_element_count(tensor)
    let bytes = @codec.tensor_storage_bytes(tensor)
    let matched : @model.TensorMatch = {
      tensor,
      element_count: elements,
      byte_count: bytes,
    }
    if tensor.shape.length() == 0 {
      scalar_count += 1
    }
    if tensor.shape.length() > max_rank {
      max_rank = tensor.shape.length()
    }
    total_elements = checked_add(
      total_elements, elements, "layout element total",
    )
    total_bytes = checked_add(total_bytes, bytes, "layout byte total")
    match largest_tensor {
      Some(current) if bigger(matched, current) =>
        largest_tensor = Some(matched)
      None => largest_tensor = Some(matched)
      _ => ()
    }
    match dtype_summary_index(dtype_summaries, tensor.dtype) {
      Some(index) => {
        let previous = dtype_summaries[index]
        dtype_summaries[index] = {
          dtype: previous.dtype,
          tensor_count: previous.tensor_count + 1,
          element_count: checked_add(
            previous.element_count,
            elements,
            "dtype element total",
          ),
          byte_count: checked_add(
            previous.byte_count,
            bytes,
            "dtype byte total",
          ),
        }
      }
      None =>
        dtype_summaries.push({
          dtype: tensor.dtype,
          tensor_count: 1,
          element_count: elements,
          byte_count: bytes,
        })
    }
  }
  dtype_summaries.sort_by((left, right) => {
    left.dtype.name().lexical_compare(right.dtype.name())
  })
  {
    tensor_count: file.header.tensors.length(),
    scalar_count,
    max_rank,
    total_elements,
    total_bytes,
    dtype_summaries,
    largest_tensor,
  }
}

///|
/// Return the number of tensors using a requested dtype without allocating a
/// query result.  It is equivalent to a dtype-only query for this common case.
pub fn count_dtype(file : @model.ValidatedFile, dtype : @model.Dtype) -> Int {
  let mut count = 0
  for tensor in file.header.tensors {
    if tensor.dtype == dtype {
      count += 1
    }
  }
  count
}

///|
/// Return the largest tensor by storage bytes, breaking ties by lexical name.
pub fn largest_tensor(
  file : @model.ValidatedFile,
) -> @model.TensorMatch? raise @model.SftkError {
  summarize_layout(file).largest_tensor
}
