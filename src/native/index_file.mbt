///|
/// Read a bounded local shard index before applying the strict index parser.
pub async fn load_index_file(
  path : String,
  limits : @model.ParseLimits,
) -> @model.ShardIndex raise @model.SftkError {
  let file = open_regular(path, false)
  defer file.close()
  let size = file.size() catch {
    _ =>
      raise @model.SftkError::IoFailure(
        operation="stat index",
        path~,
        message="unable to determine shard index size",
      )
  }
  if size < 0L || size.reinterpret_as_uint64() > limits.max_header_bytes {
    raise @model.SftkError::HeaderTooLarge(
      length=if size < 0L { 0UL } else { size.reinterpret_as_uint64() },
      limit=limits.max_header_bytes,
    )
  }
  @codec.parse_shard_index(read_at(file, path, size.to_int(), 0UL), limits~)
}
