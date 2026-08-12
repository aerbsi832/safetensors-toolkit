///|
fn parse_uint64(
  value : String,
  field : String,
) -> UInt64 raise @model.SftkError {
  if value.is_empty() {
    raise @model.SftkError::InvalidOptions(
      message=field + " selector value cannot be empty",
    )
  }
  let maximum = UInt64::lnot(0UL)
  let mut result = 0UL
  for character in value {
    if character < '0' || character > '9' {
      raise @model.SftkError::InvalidOptions(
        message=field + " selector value must be unsigned decimal",
      )
    }
    let digit = (character.to_int() - '0'.to_int()).to_uint64()
    if result > (maximum - digit) / 10UL {
      raise @model.SftkError::InvalidOptions(
        message=field + " selector value overflows UInt64",
      )
    }
    result = result * 10UL + digit
  }
  result
}

///|
fn parse_rank(value : String) -> Int raise @model.SftkError {
  let number = parse_uint64(value, "rank")
  if number > 2147483647UL {
    raise @model.SftkError::InvalidOptions(
      message="rank selector value exceeds Int range",
    )
  }
  number.to_int()
}

///|
fn parse_dtypes(value : String) -> Array[@model.Dtype] raise @model.SftkError {
  if value.is_empty() {
    raise @model.SftkError::InvalidOptions(
      message="dtype selector needs at least one dtype",
    )
  }
  let dtypes : Array[@model.Dtype] = []
  for part in value.split("|") {
    let name = part.trim().to_owned()
    match @model.Dtype::parse(name) {
      Some(dtype) if !dtypes.contains(dtype) => dtypes.push(dtype)
      Some(_) => ()
      None => raise @model.SftkError::UnknownDtype(name="selector", dtype=name)
    }
  }
  if dtypes.is_empty() {
    raise @model.SftkError::InvalidOptions(
      message="dtype selector needs at least one dtype",
    )
  }
  dtypes.sort_by((left, right) => left.name().lexical_compare(right.name()))
  dtypes
}

///|
fn split_term(term : String) -> (String, String) raise @model.SftkError {
  match term.find("=") {
    Some(index) if index > 0 && index + 1 < term.length() =>
      (term[:index].trim().to_owned(), term[index + 1:].trim().to_owned())
    _ =>
      raise @model.SftkError::InvalidOptions(
        message="selector terms must have the form key=value",
      )
  }
}

///|
fn name_match_from(key : String, value : String) -> @model.TensorNameMatch? {
  match key {
    "name" => Some(@model.TensorNameMatch::ExactName(value~))
    "prefix" => Some(@model.TensorNameMatch::Prefix(value~))
    "suffix" => Some(@model.TensorNameMatch::Suffix(value~))
    "contains" => Some(@model.TensorNameMatch::Contains(value~))
    _ => None
  }
}

///|
/// Parse a compact comma-separated query selector, for example:
/// `prefix=encoder.,dtype=F16|F32,rank=2,min_bytes=1024`.
/// Dtype lists use `|` because comma is the outer selector delimiter. Repeated fields are rejected rather than silently
/// choosing one of several user intentions.
pub fn parse_selector(
  text : String,
) -> @model.TensorQuery raise @model.SftkError {
  let trimmed = text.trim().to_owned()
  if trimmed.is_empty() || trimmed == "all" {
    return @model.TensorQuery::all()
  }
  let mut name_match = @model.TensorNameMatch::AnyName
  let mut has_name_match = false
  let mut dtypes : Array[@model.Dtype] = []
  let mut rank : Int? = None
  let mut min_bytes : UInt64? = None
  let mut max_bytes : UInt64? = None
  let mut require_metadata_key : String? = None
  let terms = trimmed.split(",").to_array()
  let mut index = 0
  while index < terms.length() {
    let term = terms[index].trim().to_owned()
    if term.is_empty() {
      index += 1
      continue
    }
    let (key, value) = split_term(term)
    match name_match_from(key, value) {
      Some(rule) => {
        if has_name_match {
          raise @model.SftkError::InvalidOptions(
            message="selector may contain only one name match",
          )
        }
        if value.is_empty() {
          raise @model.SftkError::InvalidOptions(
            message="name selector value cannot be empty",
          )
        }
        name_match = rule
        has_name_match = true
      }
      None =>
        match key {
          "dtype" => {
            if !dtypes.is_empty() {
              raise @model.SftkError::InvalidOptions(
                message="selector may contain only one dtype field",
              )
            }
            dtypes = parse_dtypes(value)
          }
          "rank" => {
            if rank is Some(_) {
              raise @model.SftkError::InvalidOptions(
                message="selector may contain only one rank field",
              )
            }
            rank = Some(parse_rank(value))
          }
          "min_bytes" => {
            if min_bytes is Some(_) {
              raise @model.SftkError::InvalidOptions(
                message="selector may contain only one min_bytes field",
              )
            }
            min_bytes = Some(parse_uint64(value, "min_bytes"))
          }
          "max_bytes" => {
            if max_bytes is Some(_) {
              raise @model.SftkError::InvalidOptions(
                message="selector may contain only one max_bytes field",
              )
            }
            max_bytes = Some(parse_uint64(value, "max_bytes"))
          }
          "metadata" => {
            if require_metadata_key is Some(_) || value.is_empty() {
              raise @model.SftkError::InvalidOptions(
                message="metadata selector needs one nonempty key",
              )
            }
            require_metadata_key = Some(value)
          }
          _ =>
            raise @model.SftkError::InvalidOptions(
              message="unknown tensor selector field: " + key,
            )
        }
    }
    index += 1
  }
  match (min_bytes, max_bytes) {
    (Some(minimum), Some(maximum)) if minimum > maximum =>
      raise @model.SftkError::InvalidOptions(
        message="min_bytes must not exceed max_bytes",
      )
    _ => ()
  }
  { name_match, dtypes, rank, min_bytes, max_bytes, require_metadata_key }
}

///|
fn name_match_text(rule : @model.TensorNameMatch) -> String {
  match rule {
    @model.TensorNameMatch::AnyName => ""
    @model.TensorNameMatch::ExactName(value~) => "name=" + value
    @model.TensorNameMatch::Prefix(value~) => "prefix=" + value
    @model.TensorNameMatch::Suffix(value~) => "suffix=" + value
    @model.TensorNameMatch::Contains(value~) => "contains=" + value
  }
}

///|
/// Render a query into its deterministic selector spelling.  Values are not
/// escaped; this helper is intended for selectors produced by `parse_selector`
/// or programmatic callers using names without comma/equal delimiters.
pub fn render_selector(query : @model.TensorQuery) -> String {
  let terms : Array[String] = []
  let name = name_match_text(query.name_match)
  if !name.is_empty() {
    terms.push(name)
  }
  if !query.dtypes.is_empty() {
    let names = query.dtypes.map(dtype => dtype.name())
    terms.push("dtype=" + names.join("|"))
  }
  match query.rank {
    Some(value) => terms.push("rank=" + value.to_string())
    None => ()
  }
  match query.min_bytes {
    Some(value) => terms.push("min_bytes=" + value.to_string())
    None => ()
  }
  match query.max_bytes {
    Some(value) => terms.push("max_bytes=" + value.to_string())
    None => ()
  }
  match query.require_metadata_key {
    Some(value) => terms.push("metadata=" + value)
    None => ()
  }
  if terms.is_empty() {
    "all"
  } else {
    terms.join(",")
  }
}
