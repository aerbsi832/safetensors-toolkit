///|
fn limits() -> @model.ParseLimits {
  @model.ParseLimits::default()
}

///|
fn valid_header() -> Bytes {
  b"{\"__metadata__\":{\"format\":\"pt\"},\"z\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[8,16]},\"a\":{\"dtype\":\"I32\",\"shape\":[2],\"data_offsets\":[0,8]}}"
}

///|
test "decode prefix and validate a complete contiguous layout" {
  let length = decode_header_length(
    b"\x80\x00\x00\x00\x00\x00\x00\x00",
    limits=limits(),
  )
  assert_eq(length, 128UL)
  let header = parse_header(valid_header(), limits=limits())
  assert_eq(header.tensors.length(), 2)
  assert_eq(header.metadata[0].key, "format")
  let file = validate_layout(header, data_len=16UL)
  let info = inspect(file)
  assert_eq(info.tensor_count, 2)
  assert_eq(info.data_bytes, 16UL)
  assert_eq(info.dtype_counts, [("F32", 1), ("I32", 1)])
}

///|
test "strict parser rejects nested duplicate keys and malformed schema values" {
  let duplicate = try
    parse_header(
      b"{\"x\":{\"dtype\":\"F32\",\"dtype\":\"I32\",\"shape\":[],\"data_offsets\":[0,4]}}",
      limits=limits(),
    )
  catch {
    @model.SftkError::DuplicateKey(path=_, key="dtype") => true
    _ => false
  } noraise {
    _ => false
  }
  assert_true(duplicate)
  let invalid_shape = try
    parse_header(
      b"{\"x\":{\"dtype\":\"F32\",\"shape\":[1.5],\"data_offsets\":[0,4]}}",
      limits=limits(),
    )
  catch {
    @model.SftkError::InvalidTensor(name="x", message=_) => true
    _ => false
  } noraise {
    _ => false
  }
  assert_true(invalid_shape)
}

///|
test "layout catches holes and canonical headers are reparsable" {
  let hole = try
    validate_layout(
      parse_header(
        b"{\"x\":{\"dtype\":\"U8\",\"shape\":[2],\"data_offsets\":[1,3]}}",
        limits=limits(),
      ),
      data_len=3UL,
    )
  catch {
    @model.SftkError::InvalidLayout(message=_) => true
    _ => false
  } noraise {
    _ => false
  }
  assert_true(hole)
  let file = validate_layout(
    parse_header(valid_header(), limits=limits()),
    data_len=16UL,
  )
  let encoded = encode_header(file)
  assert_eq(encoded.length() % 8, 0)
  let reparsed = parse_header(encoded, limits=limits())
  assert_eq(reparsed.tensors[0].name, "a")
  assert_eq(reparsed.tensors[1].name, "z")
}

///|
test "packed dtypes require byte-aligned tensor element counts" {
  let misaligned = try
    validate_layout(
      parse_header(
        b"{\"x\":{\"dtype\":\"F4\",\"shape\":[1],\"data_offsets\":[0,1]}}",
        limits=limits(),
      ),
      data_len=1UL,
    )
  catch {
    @model.SftkError::InvalidTensor(name="x", message=_) => true
    _ => false
  } noraise {
    _ => false
  }
  assert_true(misaligned)
}

///|
test "supported packed and wide dtypes validate their checked storage widths" {
  let f4 = validate_layout(
    parse_header(
      b"{\"x\":{\"dtype\":\"F4\",\"shape\":[2],\"data_offsets\":[0,1]}}",
      limits=limits(),
    ),
    data_len=1UL,
  )
  assert_eq(f4.data_len, 1UL)
  let f6 = validate_layout(
    parse_header(
      b"{\"x\":{\"dtype\":\"F6_E2M3\",\"shape\":[4],\"data_offsets\":[0,3]}}",
      limits=limits(),
    ),
    data_len=3UL,
  )
  assert_eq(f6.data_len, 3UL)
  let boolean = validate_layout(
    parse_header(
      b"{\"x\":{\"dtype\":\"BOOL\",\"shape\":[2],\"data_offsets\":[0,2]}}",
      limits=limits(),
    ),
    data_len=2UL,
  )
  assert_eq(boolean.data_len, 2UL)
  let bf16 = validate_layout(
    parse_header(
      b"{\"x\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,4]}}",
      limits=limits(),
    ),
    data_len=4UL,
  )
  assert_eq(bf16.data_len, 4UL)
  let complex = validate_layout(
    parse_header(
      b"{\"x\":{\"dtype\":\"C64\",\"shape\":[1],\"data_offsets\":[0,8]}}",
      limits=limits(),
    ),
    data_len=8UL,
  )
  assert_eq(complex.data_len, 8UL)
}
