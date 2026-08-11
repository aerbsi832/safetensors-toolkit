///|
/// Emit a JSON string value without relying on a map serializer.
fn quoted(text : String) -> String {
  let output = StringBuilder()
  output.write_char('"')
  for character in text {
    match character {
      '"' => output.write_string("\\\"")
      '\\' => output.write_string("\\\\")
      '\b' => output.write_string("\\b")
      '\f' => output.write_string("\\f")
      '\n' => output.write_string("\\n")
      '\r' => output.write_string("\\r")
      '\t' => output.write_string("\\t")
      _ if character.to_int() < 0x20 => {
        let hexadecimal = "0123456789abcdef".to_array()
        let code = character.to_int()
        output.write_string("\\u00")
        output.write_char(hexadecimal[(code >> 4) & 15])
        output.write_char(hexadecimal[code & 15])
      }
      _ => output.write_char(character)
    }
  }
  output.write_char('"')
  output.to_string()
}

///|
fn render_shape(shape : Array[UInt64]) -> String {
  "[" + shape.map(value => value.to_string()).join(",") + "]"
}

///|
fn render_tensor(tensor : @model.TensorInfo) -> String {
  "{\"data_offsets\":[" +
  tensor.data_start.to_string() +
  "," +
  tensor.data_end.to_string() +
  "],\"dtype\":" +
  quoted(tensor.dtype.name()) +
  ",\"shape\":" +
  render_shape(tensor.shape) +
  "}"
}

///|
/// Emit a deterministic valid header for transformation output.
pub fn encode_header(file : @model.ValidatedFile) -> Bytes {
  let fields : Array[(String, String)] = []
  if !file.header.metadata.is_empty() {
    let metadata = file.header.metadata.copy()
    metadata.sort_by((left, right) => left.key.lexical_compare(right.key))
    let rendered = metadata.map(entry => {
      quoted(entry.key) + ":" + quoted(entry.value)
    })
    fields.push(("__metadata__", "{" + rendered.join(",") + "}"))
  }
  let tensors = file.header.tensors.copy()
  tensors.sort_by((left, right) => left.name.lexical_compare(right.name))
  for tensor in tensors {
    fields.push((tensor.name, render_tensor(tensor)))
  }
  fields.sort_by((left, right) => left.0.lexical_compare(right.0))
  let body = "{" +
    fields.map(field => quoted(field.0) + ":" + field.1).join(",") +
    "}"
  let bytes = @utf8.encode(body)
  let remainder = bytes.length() % 8
  if remainder == 0 {
    bytes
  } else {
    @utf8.encode(body + " ".repeat(8 - remainder))
  }
}

///|
/// Calculate header-only stable file statistics.
pub fn inspect(file : @model.ValidatedFile) -> @model.FileInfo {
  let counts : Array[(String, Int)] = []
  for tensor in file.header.tensors {
    let name = tensor.dtype.name()
    let mut matched = false
    for index, entry in counts {
      if entry.0 == name {
        counts[index] = (entry.0, entry.1 + 1)
        matched = true
        break
      }
    }
    if !matched {
      counts.push((name, 1))
    }
  }
  counts.sort_by((left, right) => left.0.lexical_compare(right.0))
  {
    tensor_count: file.header.tensors.length(),
    metadata_count: file.header.metadata.length(),
    data_bytes: file.data_len,
    dtype_counts: counts,
  }
}
