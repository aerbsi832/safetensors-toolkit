///|
/// Internal JSON tree which retains every object member so duplicate keys can
/// be rejected before schema conversion.
priv enum JsonValue {
  JNull
  JBool(Bool)
  JNumber(String)
  JString(String)
  JArray(Array[JsonValue])
  JObject(Array[(String, JsonValue)])
}

///|
fn json_kind(value : JsonValue) -> String {
  match value {
    JNull => "null"
    JBool(value) => if value { "true" } else { "false" }
    JNumber(_) => "number"
    JString(_) => "string"
    JArray(_) => "array"
    JObject(_) => "object"
  }
}

///|
fn[T] json_error(offset : Int, message : String) -> T raise @model.SftkError {
  raise @model.SftkError::InvalidJson(offset=offset.to_uint64(), message~)
}

///|
fn whitespace(character : Char) -> Bool {
  character == ' ' ||
  character == '\n' ||
  character == '\r' ||
  character == '\t'
}

///|
fn skip_whitespace(chars : Array[Char], cursor : Int) -> Int {
  let mut next = cursor
  while next < chars.length() && whitespace(chars[next]) {
    next += 1
  }
  next
}

///|
fn digit(character : Char) -> Bool {
  character >= '0' && character <= '9'
}

///|
fn hex_value(character : Char) -> Int? {
  if character >= '0' && character <= '9' {
    Some(character.to_int() - '0'.to_int())
  } else if character >= 'a' && character <= 'f' {
    Some(character.to_int() - 'a'.to_int() + 10)
  } else if character >= 'A' && character <= 'F' {
    Some(character.to_int() - 'A'.to_int() + 10)
  } else {
    None
  }
}

///|
fn parse_hex4(
  chars : Array[Char],
  cursor : Int,
) -> (Int, Int) raise @model.SftkError {
  if cursor + 4 > chars.length() {
    json_error(cursor, "truncated unicode escape")
  }
  let mut value = 0
  for index in cursor..<(cursor + 4) {
    match hex_value(chars[index]) {
      Some(part) => value = value * 16 + part
      None => json_error(index, "invalid unicode escape")
    }
  }
  (value, cursor + 4)
}

///|
fn parse_json_string(
  chars : Array[Char],
  cursor : Int,
  limits : @model.ParseLimits,
) -> (String, Int) raise @model.SftkError {
  if cursor >= chars.length() || chars[cursor] != '"' {
    json_error(cursor, "expected JSON string")
  }
  let output = StringBuilder()
  let mut next = cursor + 1
  while next < chars.length() {
    let character = chars[next]
    if character == '"' {
      let value = output.to_string()
      if @utf8.encode(value).length() > limits.max_string_bytes {
        json_error(cursor, "JSON string exceeds configured byte limit")
      }
      return (value, next + 1)
    }
    if character.to_int() < 0x20 {
      json_error(next, "unescaped control character in JSON string")
    }
    if character != '\\' {
      output.write_char(character)
      next += 1
      continue
    }
    if next + 1 >= chars.length() {
      json_error(next, "truncated JSON string escape")
    }
    let escaped = chars[next + 1]
    match escaped {
      '"' => output.write_char('"')
      '\\' => output.write_char('\\')
      '/' => output.write_char('/')
      'b' => output.write_char('\b')
      'f' => output.write_char('\f')
      'n' => output.write_char('\n')
      'r' => output.write_char('\r')
      't' => output.write_char('\t')
      'u' => {
        let (codepoint, after_escape) = parse_hex4(chars, next + 2)
        if codepoint >= 0xd800 && codepoint <= 0xdbff {
          if after_escape + 2 > chars.length() ||
            chars[after_escape] != '\\' ||
            chars[after_escape + 1] != 'u' {
            json_error(
              next, "high surrogate is not followed by a low surrogate",
            )
          }
          let (low, after_low) = parse_hex4(chars, after_escape + 2)
          if low < 0xdc00 || low > 0xdfff {
            json_error(after_escape + 2, "invalid low surrogate")
          }
          let scalar = 0x10000 + (codepoint - 0xd800) * 0x400 + (low - 0xdc00)
          output.write_char(scalar.unsafe_to_char())
          next = after_low
        } else if codepoint >= 0xdc00 && codepoint <= 0xdfff {
          json_error(next, "unpaired low surrogate")
        } else {
          output.write_char(codepoint.unsafe_to_char())
          next = after_escape
        }
        continue
      }
      _ => json_error(next + 1, "invalid JSON string escape")
    }
    next += 2
  }
  json_error(cursor, "unterminated JSON string")
}

///|
fn parse_number(
  chars : Array[Char],
  cursor : Int,
) -> (JsonValue, Int) raise @model.SftkError {
  let start = cursor
  let mut next = cursor
  if next < chars.length() && chars[next] == '-' {
    next += 1
  }
  if next >= chars.length() {
    json_error(start, "truncated JSON number")
  }
  if chars[next] == '0' {
    next += 1
    if next < chars.length() && digit(chars[next]) {
      json_error(next, "JSON number has a leading zero")
    }
  } else if chars[next] >= '1' && chars[next] <= '9' {
    next += 1
    while next < chars.length() && digit(chars[next]) {
      next += 1
    }
  } else {
    json_error(next, "invalid JSON number")
  }
  if next < chars.length() && chars[next] == '.' {
    next += 1
    if next >= chars.length() || !digit(chars[next]) {
      json_error(next, "JSON fraction requires a digit")
    }
    while next < chars.length() && digit(chars[next]) {
      next += 1
    }
  }
  if next < chars.length() && (chars[next] == 'e' || chars[next] == 'E') {
    next += 1
    if next < chars.length() && (chars[next] == '+' || chars[next] == '-') {
      next += 1
    }
    if next >= chars.length() || !digit(chars[next]) {
      json_error(next, "JSON exponent requires a digit")
    }
    while next < chars.length() && digit(chars[next]) {
      next += 1
    }
  }
  (JNumber(String::from_array(chars[start:next])), next)
}

///|
fn parse_literal(
  chars : Array[Char],
  cursor : Int,
  text : String,
  value : JsonValue,
) -> (JsonValue, Int) raise @model.SftkError {
  let literal = text.to_array()
  if cursor + literal.length() > chars.length() {
    json_error(cursor, "truncated JSON literal")
  }
  for index in 0..<literal.length() {
    if chars[cursor + index] != literal[index] {
      json_error(cursor, "invalid JSON literal")
    }
  }
  (value, cursor + literal.length())
}

///|
fn parse_json_array(
  chars : Array[Char],
  cursor : Int,
  limits : @model.ParseLimits,
  depth : Int,
) -> (JsonValue, Int) raise @model.SftkError {
  let values : Array[JsonValue] = []
  let mut next = skip_whitespace(chars, cursor + 1)
  if next < chars.length() && chars[next] == ']' {
    return (JArray(values), next + 1)
  }
  while true {
    if values.length() >= limits.max_tensors {
      json_error(next, "JSON array exceeds configured item limit")
    }
    let (value, after_value) = parse_json_value(chars, next, limits, depth + 1)
    values.push(value)
    next = skip_whitespace(chars, after_value)
    if next >= chars.length() {
      json_error(next, "unterminated JSON array")
    }
    if chars[next] == ']' {
      return (JArray(values), next + 1)
    }
    if chars[next] != ',' {
      json_error(next, "expected comma or closing bracket")
    }
    next = skip_whitespace(chars, next + 1)
  }
  json_error(next, "unterminated JSON array")
}

///|
fn parse_json_object(
  chars : Array[Char],
  cursor : Int,
  limits : @model.ParseLimits,
  depth : Int,
) -> (JsonValue, Int) raise @model.SftkError {
  let entries : Array[(String, JsonValue)] = []
  let mut next = skip_whitespace(chars, cursor + 1)
  if next < chars.length() && chars[next] == '}' {
    return (JObject(entries), next + 1)
  }
  while true {
    let (key, after_key) = parse_json_string(chars, next, limits)
    for _, entry in entries {
      let (existing, _) = entry
      if existing == key {
        raise @model.SftkError::DuplicateKey(path="$", key~)
      }
    }
    next = skip_whitespace(chars, after_key)
    if next >= chars.length() || chars[next] != ':' {
      json_error(next, "expected colon after object key")
    }
    next = skip_whitespace(chars, next + 1)
    if entries.length() >= limits.max_tensors + limits.max_metadata_entries {
      json_error(next, "JSON object exceeds configured item limit")
    }
    let (value, after_value) = parse_json_value(chars, next, limits, depth + 1)
    entries.push((key, value))
    next = skip_whitespace(chars, after_value)
    if next >= chars.length() {
      json_error(next, "unterminated JSON object")
    }
    if chars[next] == '}' {
      return (JObject(entries), next + 1)
    }
    if chars[next] != ',' {
      json_error(next, "expected comma or closing brace")
    }
    next = skip_whitespace(chars, next + 1)
  }
  json_error(next, "unterminated JSON object")
}

///|
fn parse_json_value(
  chars : Array[Char],
  cursor : Int,
  limits : @model.ParseLimits,
  depth : Int,
) -> (JsonValue, Int) raise @model.SftkError {
  if depth > limits.max_depth {
    json_error(cursor, "JSON exceeds configured nesting depth")
  }
  let next = skip_whitespace(chars, cursor)
  if next >= chars.length() {
    json_error(next, "expected JSON value")
  }
  match chars[next] {
    '{' => parse_json_object(chars, next, limits, depth)
    '[' => parse_json_array(chars, next, limits, depth)
    '"' => {
      let (value, after_value) = parse_json_string(chars, next, limits)
      (JString(value), after_value)
    }
    't' => parse_literal(chars, next, "true", JBool(true))
    'f' => parse_literal(chars, next, "false", JBool(false))
    'n' => parse_literal(chars, next, "null", JNull)
    '-' | '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9' =>
      parse_number(chars, next)
    _ => json_error(next, "invalid JSON value")
  }
}

///|
fn parse_complete_json(
  text : String,
  limits : @model.ParseLimits,
) -> JsonValue raise @model.SftkError {
  let chars = text.to_array()
  let (value, cursor) = parse_json_value(chars, 0, limits, 0)
  let after = skip_whitespace(chars, cursor)
  if after != chars.length() {
    json_error(after, "trailing non-whitespace JSON content")
  }
  value
}
