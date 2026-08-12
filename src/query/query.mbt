///|
fn name_matches(rule : @model.TensorNameMatch, name : String) -> Bool {
  match rule {
    @model.TensorNameMatch::AnyName => true
    @model.TensorNameMatch::ExactName(value~) => name == value
    @model.TensorNameMatch::Prefix(value~) => name.has_prefix(value)
    @model.TensorNameMatch::Suffix(value~) => name.has_suffix(value)
    @model.TensorNameMatch::Contains(value~) => name.contains(value)
  }
}

///|
fn metadata_has_key(file : @model.ValidatedFile, required : String) -> Bool {
  for entry in file.header.metadata {
    if entry.key == required {
      return true
    }
  }
  false
}

///|
fn query_is_valid(query : @model.TensorQuery) -> Bool {
  match (query.min_bytes, query.max_bytes) {
    (Some(minimum), Some(maximum)) => minimum <= maximum
    _ => true
  }
}

///|
fn tensor_matches(
  tensor : @model.TensorInfo,
  bytes : UInt64,
  query : @model.TensorQuery,
) -> Bool {
  if !name_matches(query.name_match, tensor.name) {
    return false
  }
  if !query.dtypes.is_empty() && !query.dtypes.contains(tensor.dtype) {
    return false
  }
  match query.rank {
    Some(rank) if tensor.shape.length() != rank => return false
    _ => ()
  }
  match query.min_bytes {
    Some(minimum) if bytes < minimum => return false
    _ => ()
  }
  match query.max_bytes {
    Some(maximum) if bytes > maximum => return false
    _ => ()
  }
  true
}

///|
fn add_checked(
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
fn compare_matches(
  left : @model.TensorMatch,
  right : @model.TensorMatch,
) -> Int {
  left.tensor.name.lexical_compare(right.tensor.name)
}

///|
/// Query a validated header without inspecting tensor payload bytes.  Results
/// are sorted by name rather than source offset so users get stable output even
/// when containers use different physical tensor ordering.
pub fn query_tensors(
  file : @model.ValidatedFile,
  query : @model.TensorQuery,
) -> @model.TensorQueryResult raise @model.SftkError {
  if !query_is_valid(query) {
    raise @model.SftkError::InvalidOptions(
      message="tensor query min_bytes must not exceed max_bytes",
    )
  }
  match query.rank {
    Some(rank) if rank < 0 =>
      raise @model.SftkError::InvalidOptions(
        message="tensor query rank must be nonnegative",
      )
    _ => ()
  }
  match query.require_metadata_key {
    Some(key) if !metadata_has_key(file, key) =>
      return { matches: [], total_elements: 0UL, total_bytes: 0UL }
    _ => ()
  }
  let matches : Array[@model.TensorMatch] = []
  let mut total_elements = 0UL
  let mut total_bytes = 0UL
  for tensor in file.header.tensors {
    let bytes = @codec.tensor_storage_bytes(tensor)
    if tensor_matches(tensor, bytes, query) {
      let elements = @codec.tensor_element_count(tensor)
      matches.push({ tensor, element_count: elements, byte_count: bytes })
      total_elements = add_checked(
        total_elements, elements, "query element total",
      )
      total_bytes = add_checked(total_bytes, bytes, "query byte total")
    }
  }
  matches.sort_by(compare_matches)
  { matches, total_elements, total_bytes }
}

///|
/// Look up an exact tensor name and return checked counts, if the tensor is in
/// the validated header.  This avoids users rebuilding a query for a common
/// single-name inspection path.
pub fn find_tensor(
  file : @model.ValidatedFile,
  name : String,
) -> @model.TensorMatch? raise @model.SftkError {
  let query : @model.TensorQuery = {
    name_match: @model.TensorNameMatch::ExactName(value=name),
    dtypes: [],
    rank: None,
    min_bytes: None,
    max_bytes: None,
    require_metadata_key: None,
  }
  let result = query_tensors(file, query)
  if result.matches.is_empty() {
    None
  } else {
    Some(result.matches[0])
  }
}

///|
/// Return tensor names in deterministic lexical order for simple completion
/// and model-browser UIs.
pub fn tensor_names(file : @model.ValidatedFile) -> Array[String] {
  let names = file.header.tensors.map(tensor => tensor.name)
  names.sort_by((left, right) => left.lexical_compare(right))
  names
}

///|
/// Return metadata values by exact key in deterministic header order.  The
/// SafeTensors parser already rejects duplicate keys, so at most one value can
/// be returned for a valid header.
pub fn metadata_value(file : @model.ValidatedFile, key : String) -> String? {
  for entry in file.header.metadata {
    if entry.key == key {
      return Some(entry.value)
    }
  }
  None
}
