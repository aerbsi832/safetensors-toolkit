///|
fn stdout_line(text : String) -> Unit {
  println(text)
}

///|
fn stderr_line(text : String) -> Unit {
  let bytes = @utf8.encode(text + "\n")
  stderr_write(bytes, bytes.length())
}

///|
#borrow(bytes)
extern "C" fn stderr_write(bytes : Bytes, length : Int) = "sftk_stderr_write"

///|
fn quote_json(value : String) -> String {
  let output = StringBuilder()
  output.write_char('"')
  for character in value {
    match character {
      '"' => output.write_string("\\\"")
      '\\' => output.write_string("\\\\")
      '\n' => output.write_string("\\n")
      '\r' => output.write_string("\\r")
      '\t' => output.write_string("\\t")
      _ if character.to_int() < 0x20 => {
        let hex = "0123456789abcdef".to_array()
        let code = character.to_int()
        output.write_string("\\u00")
        output.write_char(hex[(code >> 4) & 15])
        output.write_char(hex[code & 15])
      }
      _ => output.write_char(character)
    }
  }
  output.write_char('"')
  output.to_string()
}

///|
fn usage() -> String {
  "sftk 0.1.0\n\nUsage:\n  sftk validate PATH [--format text|json]\n  sftk info PATH [--format text|json]\n  sftk query PATH SELECTOR [--format text|json]\n  sftk diff LEFT RIGHT [--format text|json]\n  sftk split PATH --max-bytes SIZE --output-dir DIR [--max-tensors N] [--dry-run|--write] [--force] [--format text|json]\n  sftk merge INDEX_PATH --output PATH [--dry-run|--write] [--force] [--format text|json]\n  sftk rename-prefix PATH --from-prefix TEXT --to-prefix TEXT --output PATH [--dry-run|--write] [--force] [--format text|json]\n\nSelectors use comma-separated fields, e.g. prefix=encoder.,dtype=F16|F32,rank=2.\n\nExit codes: 0 success, 1 validation failure or structural differences, 2 argument error, 3 I/O or publish failure."
}

///|
fn render_info(file : @model.ValidatedFile, format : OutputFormat) -> String {
  let info = @codec.inspect(file)
  match format {
    Text =>
      "valid: tensors=" +
      info.tensor_count.to_string() +
      " metadata=" +
      info.metadata_count.to_string() +
      " data_bytes=" +
      info.data_bytes.to_string() +
      " dtypes=" +
      info.dtype_counts.map(pair => pair.0 + ":" + pair.1.to_string()).join(",")
    Json =>
      "{\"schema_version\":\"sftk/v1\",\"valid\":true,\"tensor_count\":" +
      info.tensor_count.to_string() +
      ",\"metadata_count\":" +
      info.metadata_count.to_string() +
      ",\"data_bytes\":" +
      info.data_bytes.to_string() +
      ",\"dtype_counts\":[" +
      info.dtype_counts
      .map(pair => {
        "{\"dtype\":" +
        quote_json(pair.0) +
        ",\"count\":" +
        pair.1.to_string() +
        "}"
      })
      .join(",") +
      "]}"
  }
}

///|
fn diff_kind_name(kind : @model.DiffKind) -> String {
  match kind {
    TensorAdded => "tensor_added"
    TensorRemoved => "tensor_removed"
    DtypeChanged => "dtype_changed"
    ShapeChanged => "shape_changed"
    DataRangeChanged => "data_range_changed"
    MetadataAdded => "metadata_added"
    MetadataRemoved => "metadata_removed"
    MetadataChanged => "metadata_changed"
  }
}

///|
fn render_diff(report : @model.DiffReport, format : OutputFormat) -> String {
  match format {
    Text => {
      let lines = report.entries.map(entry => {
        diff_kind_name(entry.kind) +
        " " +
        entry.name +
        (match (entry.before, entry.after) {
          (Some(before), Some(after)) => ": " + before + " -> " + after
          (Some(before), None) => ": " + before + " -> <absent>"
          (None, Some(after)) => ": <absent> -> " + after
          (None, None) => ""
        })
      })
      if report.equal {
        "equal"
      } else {
        lines.join("\n")
      }
    }
    Json =>
      "{\"schema_version\":\"sftk/v1\",\"equal\":" +
      (if report.equal { "true" } else { "false" }) +
      ",\"changes\":[" +
      report.entries
      .map(entry => {
        "{\"kind\":" +
        quote_json(diff_kind_name(entry.kind)) +
        ",\"name\":" +
        quote_json(entry.name) +
        ",\"before\":" +
        (match entry.before {
          Some(value) => quote_json(value)
          None => "null"
        }) +
        ",\"after\":" +
        (match entry.after {
          Some(value) => quote_json(value)
          None => "null"
        }) +
        "}"
      })
      .join(",") +
      "]}"
  }
}

///|
fn render_split(
  plan : @model.ShardPlan,
  write : Bool,
  format : OutputFormat,
) -> String {
  match format {
    Text =>
      "split plan: mode=" +
      (if write { "write" } else { "dry-run" }) +
      " shards=" +
      plan.shards.length().to_string() +
      " tensors=" +
      plan.source_tensor_count.to_string() +
      " data_bytes=" +
      plan.source_data_bytes.to_string() +
      "\n" +
      plan.shards
      .map(shard => {
        shard.name +
        " bytes=" +
        shard.data_bytes.to_string() +
        " tensors=" +
        shard.tensors.length().to_string()
      })
      .join("\n")
    Json =>
      "{\"schema_version\":\"sftk/v1\",\"mode\":" +
      quote_json(if write { "write" } else { "dry-run" }) +
      ",\"shards\":[" +
      plan.shards
      .map(shard => {
        "{\"name\":" +
        quote_json(shard.name) +
        ",\"data_bytes\":" +
        shard.data_bytes.to_string() +
        ",\"tensor_count\":" +
        shard.tensors.length().to_string() +
        "}"
      })
      .join(",") +
      "],\"index\":{\"total_size\":" +
      plan.index.total_size.to_string() +
      ",\"weight_count\":" +
      plan.index.weight_map.length().to_string() +
      "}}"
  }
}

///|
fn render_merge(
  plan : @model.MergePlan,
  write : Bool,
  format : OutputFormat,
) -> String {
  match format {
    Text =>
      "merge plan: mode=" +
      (if write { "write" } else { "dry-run" }) +
      " tensors=" +
      plan.entries.length().to_string() +
      " data_bytes=" +
      plan.output_data_bytes.to_string()
    Json =>
      "{\"schema_version\":\"sftk/v1\",\"mode\":" +
      quote_json(if write { "write" } else { "dry-run" }) +
      ",\"tensor_count\":" +
      plan.entries.length().to_string() +
      ",\"data_bytes\":" +
      plan.output_data_bytes.to_string() +
      "}"
  }
}

///|
fn render_query(
  selector : String,
  result : @model.TensorQueryResult,
  format : OutputFormat,
) -> String {
  match format {
    Text => {
      let lines = result.matches.map(match_ => {
        match_.tensor.name +
        " dtype=" +
        match_.tensor.dtype.name() +
        " elements=" +
        match_.element_count.to_string() +
        " bytes=" +
        match_.byte_count.to_string()
      })
      let detail = if lines.is_empty() { "" } else { "\n" + lines.join("\n") }
      "selector=" +
      selector +
      " matches=" +
      result.matches.length().to_string() +
      " total_bytes=" +
      result.total_bytes.to_string() +
      detail
    }
    Json =>
      "{\"schema_version\":\"sftk/v1\",\"selector\":" +
      quote_json(selector) +
      ",\"total_elements\":" +
      result.total_elements.to_string() +
      ",\"total_bytes\":" +
      result.total_bytes.to_string() +
      ",\"matches\":[" +
      result.matches
      .map(match_ => {
        "{\"name\":" +
        quote_json(match_.tensor.name) +
        ",\"dtype\":" +
        quote_json(match_.tensor.dtype.name()) +
        ",\"shape\":[" +
        match_.tensor.shape.map(value => value.to_string()).join(",") +
        "],\"elements\":" +
        match_.element_count.to_string() +
        ",\"bytes\":" +
        match_.byte_count.to_string() +
        "}"
      })
      .join(",") +
      "]}"
  }
}

///|
fn render_rename_prefix(
  plan : @model.RenamePlan,
  write : Bool,
  format : OutputFormat,
) -> String {
  match format {
    Text => {
      let detail = if plan.renames.is_empty() {
        ""
      } else {
        "\n" +
        plan.renames.map(rename => rename.from + " -> " + rename.to).join("\n")
      }
      "rename-prefix plan: mode=" +
      (if write { "write" } else { "dry-run" }) +
      " changed=" +
      (if plan.changed { "true" } else { "false" }) +
      " renames=" +
      plan.renames.length().to_string() +
      detail
    }
    Json =>
      "{\"schema_version\":\"sftk/v1\",\"mode\":" +
      quote_json(if write { "write" } else { "dry-run" }) +
      ",\"changed\":" +
      (if plan.changed { "true" } else { "false" }) +
      ",\"renames\":[" +
      plan.renames
      .map(rename => {
        "{\"from\":" +
        quote_json(rename.from) +
        ",\"to\":" +
        quote_json(rename.to) +
        "}"
      })
      .join(",") +
      "]}"
  }
}

///|
fn error_text(error : @model.SftkError) -> String {
  match error {
    Truncated(phase~, expected~, actual~) =>
      phase +
      " truncated: expected " +
      expected.to_string() +
      ", got " +
      actual.to_string()
    HeaderTooLarge(length~, limit~) =>
      "header too large: " + length.to_string() + " > " + limit.to_string()
    InvalidUtf8(offset~) => "invalid UTF-8 at byte " + offset.to_string()
    InvalidJson(offset~, message~) =>
      "invalid JSON at offset " + offset.to_string() + ": " + message
    DuplicateKey(path~, key~) => "duplicate JSON key " + key + " at " + path
    InvalidTensor(name~, message~) => "invalid tensor " + name + ": " + message
    UnknownDtype(name~, dtype~) =>
      "unknown dtype " + dtype + " for tensor " + name
    ArithmeticOverflow(context~) => "arithmetic overflow: " + context
    InvalidLayout(message~) => "invalid layout: " + message
    InvalidIndex(message~) => "invalid shard index: " + message
    IoFailure(operation~, path~, message~) =>
      operation + " " + path + ": " + message
    Conflict(path~, message~) => "conflict " + path + ": " + message
    InvalidOptions(message~) => message
  }
}

///|
fn error_exit(error : @model.SftkError) -> Int {
  match error {
    IoFailure(..) | Conflict(..) => 3
    InvalidOptions(..) => 2
    _ => 1
  }
}
