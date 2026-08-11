///|
/// Decode the little-endian u64 header length from exactly the file prefix.
pub fn decode_header_length(
  prefix : Bytes,
  limits~ : @model.ParseLimits,
) -> UInt64 raise @model.SftkError {
  if prefix.length() != 8 {
    raise @model.SftkError::Truncated(
      phase="header length prefix",
      expected=8UL,
      actual=prefix.length().to_uint64(),
    )
  }
  let mut length = 0UL
  for index in 0..<8 {
    let signed = prefix[index].to_int()
    let byte = if signed < 0 {
      (signed + 256).to_uint64()
    } else {
      signed.to_uint64()
    }
    length = length | (byte << (index * 8))
  }
  if length > limits.max_header_bytes {
    raise @model.SftkError::HeaderTooLarge(
      length~,
      limit=limits.max_header_bytes,
    )
  }
  length
}

///|
fn object_value(
  entries : Array[(String, JsonValue)],
  key : String,
) -> JsonValue? {
  for _, entry in entries {
    let (existing, value) = entry
    if existing == key {
      return Some(value)
    }
  }
  None
}

///|
fn require_string(
  value : JsonValue?,
  tensor : String,
  field : String,
) -> String raise @model.SftkError {
  match value {
    Some(JString(text)) => text
    Some(other) =>
      raise @model.SftkError::InvalidTensor(
        name=tensor,
        message="field '" +
          field +
          "' must be a string, not " +
          json_kind(other),
      )
    _ =>
      raise @model.SftkError::InvalidTensor(
        name=tensor,
        message="field '" + field + "' must be a string",
      )
  }
}

///|
fn parse_uint64(
  text : String,
  tensor : String,
  field : String,
) -> UInt64 raise @model.SftkError {
  if text.is_empty() {
    raise @model.SftkError::InvalidTensor(
      name=tensor,
      message="field '" + field + "' must be an unsigned integer",
    )
  }
  let mut value = 0UL
  let maximum = UInt64::lnot(0UL)
  for character in text {
    if character < '0' || character > '9' {
      raise @model.SftkError::InvalidTensor(
        name=tensor,
        message="field '" + field + "' must be an unsigned integer",
      )
    }
    let part = (character.to_int() - '0'.to_int()).to_uint64()
    if value > (maximum - part) / 10UL {
      raise @model.SftkError::ArithmeticOverflow(
        context="unsigned integer in " + tensor + "." + field,
      )
    }
    value = value * 10UL + part
  }
  value
}

///|
fn require_uint64(
  value : JsonValue,
  tensor : String,
  field : String,
) -> UInt64 raise @model.SftkError {
  match value {
    JNumber(text) => parse_uint64(text, tensor, field)
    other =>
      raise @model.SftkError::InvalidTensor(
        name=tensor,
        message="field '" +
          field +
          "' must contain unsigned integers, not " +
          json_kind(other),
      )
  }
}

///|
fn parse_shape(
  value : JsonValue?,
  tensor : String,
) -> Array[UInt64] raise @model.SftkError {
  match value {
    Some(JArray(items)) => {
      let shape : Array[UInt64] = []
      for item in items {
        shape.push(require_uint64(item, tensor, "shape"))
      }
      shape
    }
    _ =>
      raise @model.SftkError::InvalidTensor(
        name=tensor,
        message="field 'shape' must be an array",
      )
  }
}

///|
fn parse_offsets(
  value : JsonValue?,
  tensor : String,
) -> (UInt64, UInt64) raise @model.SftkError {
  match value {
    Some(JArray(items)) if items.length() == 2 =>
      (
        require_uint64(items[0], tensor, "data_offsets"),
        require_uint64(items[1], tensor, "data_offsets"),
      )
    _ =>
      raise @model.SftkError::InvalidTensor(
        name=tensor,
        message="field 'data_offsets' must contain exactly two unsigned integers",
      )
  }
}

///|
fn parse_tensor(
  name : String,
  value : JsonValue,
  source_order : Int,
) -> @model.TensorInfo raise @model.SftkError {
  match value {
    JObject(entries) => {
      for _, entry in entries {
        let (field, _) = entry
        if field != "dtype" && field != "shape" && field != "data_offsets" {
          raise @model.SftkError::InvalidTensor(
            name~,
            message="unexpected tensor field '" + field + "'",
          )
        }
      }
      let dtype_name = require_string(
        object_value(entries, "dtype"),
        name,
        "dtype",
      )
      let dtype = match @model.Dtype::parse(dtype_name) {
        Some(value) => value
        None => raise @model.SftkError::UnknownDtype(name~, dtype=dtype_name)
      }
      let shape = parse_shape(object_value(entries, "shape"), name)
      let (data_start, data_end) = parse_offsets(
        object_value(entries, "data_offsets"),
        name,
      )
      { name, dtype, shape, data_start, data_end, source_order }
    }
    _ =>
      raise @model.SftkError::InvalidTensor(
        name~,
        message="tensor entry must be an object",
      )
  }
}

///|
fn parse_metadata(
  value : JsonValue,
) -> Array[@model.MetadataEntry] raise @model.SftkError {
  match value {
    JObject(entries) => {
      let metadata : Array[@model.MetadataEntry] = []
      for _, entry in entries {
        let (key, value) = entry
        match value {
          JString(text) => metadata.push({ key, value: text })
          other =>
            raise @model.SftkError::InvalidJson(
              offset=0UL,
              message="__metadata__." +
                key +
                " must be a string, not " +
                json_kind(other),
            )
        }
      }
      metadata
    }
    _ =>
      raise @model.SftkError::InvalidJson(
        offset=0UL,
        message="__metadata__ must be an object",
      )
  }
}

///|
/// Strictly parse UTF-8 JSON while retaining enough structure to reject
/// duplicate keys instead of silently overwriting them.
pub fn parse_header(
  header_bytes : Bytes,
  limits~ : @model.ParseLimits,
) -> @model.HeaderSnapshot raise @model.SftkError {
  let length = header_bytes.length().to_uint64()
  if length > limits.max_header_bytes {
    raise @model.SftkError::HeaderTooLarge(
      length~,
      limit=limits.max_header_bytes,
    )
  }
  if header_bytes.length() == 0 || header_bytes[0].to_int() != 0x7b {
    raise @model.SftkError::InvalidJson(
      offset=0UL,
      message="SafeTensors header must begin with '{'",
    )
  }
  let text = @utf8.decode(header_bytes) catch {
    _ => raise @model.SftkError::InvalidUtf8(offset=0UL)
  }
  let root = parse_complete_json(text, limits)
  match root {
    JObject(entries) => {
      let tensors : Array[@model.TensorInfo] = []
      let mut metadata : Array[@model.MetadataEntry] = []
      for _, entry in entries {
        let (name, value) = entry
        if name == "__metadata__" {
          metadata = parse_metadata(value)
          if metadata.length() > limits.max_metadata_entries {
            raise @model.SftkError::InvalidJson(
              offset=0UL,
              message="metadata exceeds configured entry limit",
            )
          }
        } else {
          if tensors.length() >= limits.max_tensors {
            raise @model.SftkError::InvalidJson(
              offset=0UL,
              message="tensor count exceeds configured limit",
            )
          }
          tensors.push(parse_tensor(name, value, tensors.length()))
        }
      }
      { tensors, metadata }
    }
    _ =>
      raise @model.SftkError::InvalidJson(
        offset=0UL,
        message="SafeTensors header root must be an object",
      )
  }
}
