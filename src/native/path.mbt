///|
fn join_path(left : String, right : String) -> String {
  if left.has_suffix("/") {
    left + right
  } else {
    left + "/" + right
  }
}

///|
fn basename(path : String) -> String {
  let chars = path.to_array()
  let mut start = 0
  for index in 0..<chars.length() {
    if chars[index] == '/' || chars[index] == '\\' {
      start = index + 1
    }
  }
  String::from_array(chars[start:])
}

///|
fn parent_path(path : String) -> String {
  let chars = path.to_array()
  let mut index = chars.length() - 1
  while index >= 0 && chars[index] != '/' && chars[index] != '\\' {
    index -= 1
  }
  if index < 0 {
    "."
  } else if index == 0 {
    "/"
  } else {
    String::from_array(chars[:index])
  }
}

///|
fn safe_basename(name : String) -> Bool {
  !name.is_empty() &&
  name != "." &&
  name != ".." &&
  !name.contains("/") &&
  !name.contains("\\")
}

///|
async fn temporary_path(destination : String) -> String raise @model.SftkError {
  for counter = 0; counter < 64; counter = counter + 1 {
    let candidate = destination + ".sftk-tmp-" + counter.to_string()
    let exists = @fs.exists(candidate) catch {
      _ =>
        raise @model.SftkError::IoFailure(
          operation="allocate temporary output",
          path=candidate,
          message="unable to inspect temporary output path",
        )
    }
    if !exists {
      return candidate
    }
  }
  raise @model.SftkError::Conflict(
    path=destination,
    message="unable to allocate a collision-free temporary output",
  )
}
