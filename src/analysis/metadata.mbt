///|
fn metadata_has(file : @model.ValidatedFile, key : String) -> Bool {
  for entry in file.header.metadata {
    if entry.key == key {
      return true
    }
  }
  false
}

///|
fn push_issue(
  issues : Array[@model.MetadataIssue],
  key : String?,
  message : String,
) -> Unit {
  issues.push({ key, message })
}

///|
fn unique_keys(
  keys : ArrayView[String],
  role : String,
) -> Unit raise @model.SftkError {
  let seen : Map[String, Bool] = Map([])
  for key in keys {
    if key.is_empty() {
      raise @model.SftkError::InvalidOptions(
        message=role + " metadata key cannot be empty",
      )
    }
    if seen.contains(key) {
      raise @model.SftkError::InvalidOptions(
        message=role + " metadata keys must be unique",
      )
    }
    seen[key] = true
  }
}

///|
/// Validate string metadata against a project-owned policy.  This never treats
/// arbitrary metadata as executable configuration and never reads data bytes.
pub fn validate_metadata(
  file : @model.ValidatedFile,
  policy : @model.MetadataPolicy,
) -> @model.MetadataValidation raise @model.SftkError {
  if policy.max_value_bytes < 0 {
    raise @model.SftkError::InvalidOptions(
      message="metadata max_value_bytes must be nonnegative",
    )
  }
  unique_keys(policy.required_keys, "required")
  unique_keys(policy.forbidden_keys, "forbidden")
  let issues : Array[@model.MetadataIssue] = []
  for key in policy.required_keys {
    if !metadata_has(file, key) {
      push_issue(issues, Some(key), "required metadata key is absent")
    }
  }
  for key in policy.forbidden_keys {
    if metadata_has(file, key) {
      push_issue(issues, Some(key), "forbidden metadata key is present")
    }
  }
  for entry in file.header.metadata {
    let byte_length = @utf8.encode(entry.value).length()
    if byte_length > policy.max_value_bytes {
      push_issue(
        issues,
        Some(entry.key),
        "metadata value exceeds configured byte limit",
      )
    }
    if !policy.allow_empty_values && entry.value.is_empty() {
      push_issue(issues, Some(entry.key), "empty metadata value is not allowed")
    }
  }
  issues.sort_by((left, right) => {
    let left_key = match left.key {
      Some(value) => value
      None => ""
    }
    let right_key = match right.key {
      Some(value) => value
      None => ""
    }
    let compared = left_key.lexical_compare(right_key)
    if compared != 0 {
      compared
    } else {
      left.message.lexical_compare(right.message)
    }
  })
  { valid: issues.is_empty(), issues }
}

///|
/// Return metadata keys in lexical order, suitable for deterministic display.
pub fn metadata_keys(file : @model.ValidatedFile) -> Array[String] {
  let keys = file.header.metadata.map(entry => entry.key)
  keys.sort_by((left, right) => left.lexical_compare(right))
  keys
}

///|
/// Return whether a metadata policy is satisfiable by construction: a key may
/// not simultaneously be required and forbidden.
pub fn policy_is_consistent(policy : @model.MetadataPolicy) -> Bool {
  for key in policy.required_keys {
    if policy.forbidden_keys.contains(key) {
      return false
    }
  }
  policy.max_value_bytes >= 0
}
