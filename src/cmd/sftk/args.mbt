///|
priv enum OutputFormat {
  Text
  Json
}

///|
priv enum Request {
  Help
  Version
  Validate(path~ : String, format~ : OutputFormat)
  Info(path~ : String, format~ : OutputFormat)
  Query(path~ : String, selector~ : String, format~ : OutputFormat)
  Diff(left~ : String, right~ : String, format~ : OutputFormat)
  Split(
    path~ : String,
    max_bytes~ : UInt64,
    max_tensors~ : Int,
    output_dir~ : String,
    write~ : Bool,
    force~ : Bool,
    format~ : OutputFormat
  )
  Merge(
    index_path~ : String,
    output~ : String,
    write~ : Bool,
    force~ : Bool,
    format~ : OutputFormat
  )
  RenamePrefix(
    path~ : String,
    from_prefix~ : String,
    to_prefix~ : String,
    output~ : String,
    write~ : Bool,
    force~ : Bool,
    format~ : OutputFormat
  )
}

///|
fn invalid_options(message : String) -> @model.SftkError {
  @model.SftkError::InvalidOptions(message~)
}

///|
fn parse_format(value : String) -> OutputFormat raise @model.SftkError {
  match value {
    "text" => Text
    "json" => Json
    _ => raise invalid_options("--format accepts only text or json")
  }
}

///|
fn parse_uint64(value : String, name : String) -> UInt64 raise @model.SftkError {
  if value.is_empty() {
    raise invalid_options(name + " requires a positive integer")
  }
  let maximum = UInt64::lnot(0UL)
  let mut output = 0UL
  for character in value {
    if character < '0' || character > '9' {
      raise invalid_options(name + " requires a decimal integer")
    }
    let digit = (character.to_int() - '0'.to_int()).to_uint64()
    if output > (maximum - digit) / 10UL {
      raise @model.SftkError::ArithmeticOverflow(context=name + " argument")
    }
    output = output * 10UL + digit
  }
  output
}

///|
fn parse_size(value : String) -> UInt64 raise @model.SftkError {
  let suffixes : Array[(String, UInt64)] = [
    ("GiB", 1024UL * 1024UL * 1024UL),
    ("MiB", 1024UL * 1024UL),
    ("KiB", 1024UL),
    ("GB", 1000UL * 1000UL * 1000UL),
    ("MB", 1000UL * 1000UL),
    ("KB", 1000UL),
    ("B", 1UL),
  ]
  for suffix in suffixes {
    let (text, multiplier) = suffix
    if value.has_suffix(text) {
      let amount = parse_uint64(
        value[:value.length() - text.length()].to_owned(),
        "--max-bytes",
      )
      if amount == 0UL || amount > UInt64::lnot(0UL) / multiplier {
        raise invalid_options("--max-bytes must be a positive in-range size")
      }
      return amount * multiplier
    }
  }
  let amount = parse_uint64(value, "--max-bytes")
  if amount == 0UL {
    raise invalid_options("--max-bytes must be positive")
  }
  amount
}

///|
fn parse_positive_int(
  value : String,
  name : String,
) -> Int raise @model.SftkError {
  let parsed = parse_uint64(value, name)
  if parsed == 0UL || parsed > 0x7fffffffUL {
    raise invalid_options(name + " must be a positive 32-bit integer")
  }
  parsed.to_int()
}

///|
fn parse_format_tail(
  args : ArrayView[String],
  start : Int,
  command : String,
) -> OutputFormat raise @model.SftkError {
  let mut format = Text
  let mut index = start
  while index < args.length() {
    if args[index] != "--format" || index + 1 >= args.length() {
      raise invalid_options("unknown " + command + " option: " + args[index])
    }
    index += 1
    format = parse_format(args[index])
    index += 1
  }
  format
}

///|
fn parse_split(args : ArrayView[String]) -> Request raise @model.SftkError {
  if args.length() < 2 {
    raise invalid_options("split requires PATH")
  }
  let mut max_bytes : UInt64? = None
  let mut max_tensors = 1_000_000
  let mut output_dir : String? = None
  let mut write = false
  let mut force = false
  let mut format = Text
  let mut index = 2
  while index < args.length() {
    match args[index] {
      "--max-bytes" => {
        index += 1
        if index >= args.length() {
          raise invalid_options("--max-bytes requires SIZE")
        }
        max_bytes = Some(parse_size(args[index]))
      }
      "--max-tensors" => {
        index += 1
        if index >= args.length() {
          raise invalid_options("--max-tensors requires COUNT")
        }
        max_tensors = parse_positive_int(args[index], "--max-tensors")
      }
      "--output-dir" => {
        index += 1
        if index >= args.length() {
          raise invalid_options("--output-dir requires PATH")
        }
        output_dir = Some(args[index])
      }
      "--write" => write = true
      "--dry-run" => write = false
      "--force" => force = true
      "--format" => {
        index += 1
        if index >= args.length() {
          raise invalid_options("--format requires text or json")
        }
        format = parse_format(args[index])
      }
      option => raise invalid_options("unknown split option: " + option)
    }
    index += 1
  }
  match (max_bytes, output_dir) {
    (Some(max_bytes), Some(output_dir)) =>
      Split(
        path=args[1],
        max_bytes~,
        max_tensors~,
        output_dir~,
        write~,
        force~,
        format~,
      )
    _ =>
      raise invalid_options(
        "split requires --max-bytes SIZE and --output-dir PATH",
      )
  }
}

///|
fn parse_merge(args : ArrayView[String]) -> Request raise @model.SftkError {
  if args.length() < 2 {
    raise invalid_options("merge requires INDEX_PATH")
  }
  let mut output : String? = None
  let mut write = false
  let mut force = false
  let mut format = Text
  let mut index = 2
  while index < args.length() {
    match args[index] {
      "--output" => {
        index += 1
        if index >= args.length() {
          raise invalid_options("--output requires PATH")
        }
        output = Some(args[index])
      }
      "--write" => write = true
      "--dry-run" => write = false
      "--force" => force = true
      "--format" => {
        index += 1
        if index >= args.length() {
          raise invalid_options("--format requires text or json")
        }
        format = parse_format(args[index])
      }
      option => raise invalid_options("unknown merge option: " + option)
    }
    index += 1
  }
  match output {
    Some(output) => Merge(index_path=args[1], output~, write~, force~, format~)
    None => raise invalid_options("merge requires --output PATH")
  }
}

///|
fn parse_query(args : ArrayView[String]) -> Request raise @model.SftkError {
  if args.length() < 3 {
    raise invalid_options("query requires PATH and SELECTOR")
  }
  Query(
    path=args[1],
    selector=args[2],
    format=parse_format_tail(args, 3, "query"),
  )
}

///|
fn parse_rename_prefix(
  args : ArrayView[String],
) -> Request raise @model.SftkError {
  if args.length() < 2 {
    raise invalid_options("rename-prefix requires PATH")
  }
  let mut from_prefix : String? = None
  let mut to_prefix : String? = None
  let mut output : String? = None
  let mut write = false
  let mut force = false
  let mut format = Text
  let mut index = 2
  while index < args.length() {
    match args[index] {
      "--from-prefix" => {
        index += 1
        if index >= args.length() {
          raise invalid_options("--from-prefix requires TEXT")
        }
        from_prefix = Some(args[index])
      }
      "--to-prefix" => {
        index += 1
        if index >= args.length() {
          raise invalid_options("--to-prefix requires TEXT")
        }
        to_prefix = Some(args[index])
      }
      "--output" => {
        index += 1
        if index >= args.length() {
          raise invalid_options("--output requires PATH")
        }
        output = Some(args[index])
      }
      "--write" => write = true
      "--dry-run" => write = false
      "--force" => force = true
      "--format" => {
        index += 1
        if index >= args.length() {
          raise invalid_options("--format requires text or json")
        }
        format = parse_format(args[index])
      }
      option => raise invalid_options("unknown rename-prefix option: " + option)
    }
    index += 1
  }
  match (from_prefix, to_prefix, output) {
    (Some(from_prefix), Some(to_prefix), Some(output)) =>
      RenamePrefix(
        path=args[1],
        from_prefix~,
        to_prefix~,
        output~,
        write~,
        force~,
        format~,
      )
    _ =>
      raise invalid_options(
        "rename-prefix requires --from-prefix TEXT --to-prefix TEXT --output PATH",
      )
  }
}

///|
fn parse_args(args : ArrayView[String]) -> Request raise @model.SftkError {
  if args.length() == 0 {
    return Help
  }
  match args[0] {
    "help" | "--help" | "-h" => Help
    "version" | "--version" | "-V" => Version
    "validate" if args.length() >= 2 =>
      Validate(path=args[1], format=parse_format_tail(args, 2, "validate"))
    "info" if args.length() >= 2 =>
      Info(path=args[1], format=parse_format_tail(args, 2, "info"))
    "query" => parse_query(args)
    "diff" if args.length() >= 3 =>
      Diff(
        left=args[1],
        right=args[2],
        format=parse_format_tail(args, 3, "diff"),
      )
    "split" => parse_split(args)
    "merge" => parse_merge(args)
    "rename-prefix" => parse_rename_prefix(args)
    "validate" => raise invalid_options("validate requires PATH")
    "info" => raise invalid_options("info requires PATH")
    "diff" => raise invalid_options("diff requires LEFT and RIGHT")
    command => raise invalid_options("unknown command: " + command)
  }
}
