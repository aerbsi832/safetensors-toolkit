///|
fn diff_file(
  header : Bytes,
  data_len : UInt64,
) -> @model.ValidatedFile raise @model.SftkError {
  let limits = @model.ParseLimits::default()
  @codec.validate_layout(@codec.parse_header(header, limits~), data_len~)
}

///|
test "diff has deterministic tensor and metadata changes" {
  let left = diff_file(
    b"{\"__metadata__\":{\"format\":\"pt\"},\"a\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}",
    4UL,
  )
  let right = diff_file(
    b"{\"__metadata__\":{\"format\":\"mlx\",\"v\":\"1\"},\"a\":{\"dtype\":\"I32\",\"shape\":[1],\"data_offsets\":[0,4]}}",
    4UL,
  )
  let report = diff(left, right)
  assert_false(report.equal)
  assert_eq(report.entries.length(), 3)
  assert_eq(report.entries[0].name, "metadata:format")
  assert_eq(report.entries[1].name, "metadata:v")
  assert_eq(report.entries[2].name, "tensor:a")
}
