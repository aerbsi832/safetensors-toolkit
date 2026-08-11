///|
fn index_uint64(
  value : JsonValue?,
  field : String,
) -> UInt64 raise @model.SftkError {
  match value {
    Some(JNumber(text)) =>
      parse_uint64(text, "index", field) catch {
        _ =>
          raise @model.SftkError::InvalidIndex(
            message="field '" + field + "' must be an unsigned integer",
          )
      }
    Some(other) =>
      raise @model.SftkError::InvalidIndex(
        message="field '" +
          field +
          "' must be an unsigned integer, not " +
          json_kind(other),
      )
    None =>
      raise @model.SftkError::InvalidIndex(
        message="missing field '" + field + "'",
      )
  }
}

///|
fn ensure_shard_name(value : String) -> String raise @model.SftkError {
  if value.is_empty() ||
    value.contains("/") ||
    value.contains("\\") ||
    value == "." ||
    value == ".." {
    raise @model.SftkError::InvalidIndex(
      message="weight_map values must be safe shard basenames",
    )
  }
  value
}

///|
fn parse_weight_map(
  value : JsonValue?,
) -> Array[(String, String)] raise @model.SftkError {
  match value {
    Some(JObject(entries)) => {
      let weights : Array[(String, String)] = []
      for _, entry in entries {
        let (name, value) = entry
        if name.is_empty() {
          raise @model.SftkError::InvalidIndex(
            message="weight_map tensor names cannot be empty",
          )
        }
        match value {
          JString(shard) => weights.push((name, ensure_shard_name(shard)))
          other =>
            raise @model.SftkError::InvalidIndex(
              message="weight_map." +
                name +
                " must be a string, not " +
                json_kind(other),
            )
        }
      }
      weights.sort_by((left, right) => left.0.lexical_compare(right.0))
      weights
    }
    Some(other) =>
      raise @model.SftkError::InvalidIndex(
        message="weight_map must be an object, not " + json_kind(other),
      )
    None =>
      raise @model.SftkError::InvalidIndex(message="missing field 'weight_map'")
  }
}

///|
/// Parse the supported Hugging Face SafeTensors index schema while preserving
/// duplicate-key rejection from the strict JSON reader.
pub fn parse_shard_index(
  index_bytes : Bytes,
  limits~ : @model.ParseLimits,
) -> @model.ShardIndex raise @model.SftkError {
  if index_bytes.length().to_uint64() > limits.max_header_bytes {
    raise @model.SftkError::HeaderTooLarge(
      length=index_bytes.length().to_uint64(),
      limit=limits.max_header_bytes,
    )
  }
  let text = @utf8.decode(index_bytes) catch {
    _ => raise @model.SftkError::InvalidUtf8(offset=0UL)
  }
  match parse_complete_json(text, limits) {
    JObject(entries) => {
      for _, entry in entries {
        let (key, _) = entry
        if key != "metadata" && key != "weight_map" {
          raise @model.SftkError::InvalidIndex(
            message="unexpected top-level index field '" + key + "'",
          )
        }
      }
      let metadata = match object_value(entries, "metadata") {
        Some(JObject(values)) => values
        Some(other) =>
          raise @model.SftkError::InvalidIndex(
            message="metadata must be an object, not " + json_kind(other),
          )
        None =>
          raise @model.SftkError::InvalidIndex(
            message="missing field 'metadata'",
          )
      }
      for _, entry in metadata {
        let (key, _) = entry
        if key != "total_size" {
          raise @model.SftkError::InvalidIndex(
            message="unsupported metadata field '" + key + "'",
          )
        }
      }
      let total_size = index_uint64(
        object_value(metadata, "total_size"),
        "metadata.total_size",
      )
      {
        format: "huggingface-safetensors-index-v1",
        total_size,
        weight_map: parse_weight_map(object_value(entries, "weight_map")),
      }
    }
    other =>
      raise @model.SftkError::InvalidIndex(
        message="index root must be an object, not " + json_kind(other),
      )
  }
}

///|
/// Encode the supported index schema deterministically for native split output.
pub fn encode_shard_index(index : @model.ShardIndex) -> Bytes {
  let weights = index.weight_map.copy()
  weights.sort_by((left, right) => left.0.lexical_compare(right.0))
  let body = "{\"metadata\":{\"total_size\":" +
    index.total_size.to_string() +
    "},\"weight_map\":{" +
    weights.map(pair => quoted(pair.0) + ":" + quoted(pair.1)).join(",") +
    "}}"
  @utf8.encode(body)
}
