///|
/// Stable public snapshots, reports, plans, policies, limits, and errors.
pub using @model {
  type DiffEntry,
  type DiffKind,
  type DiffReport,
  type Dtype,
  type DtypeSummary,
  type ExecutionOptions,
  type FileInfo,
  type HeaderSnapshot,
  type LayoutSummary,
  type MergeEntry,
  type MergePlan,
  type MetadataEntry,
  type MetadataIssue,
  type MetadataPolicy,
  type MetadataValidation,
  type ParseLimits,
  type PlannedShard,
  type PlannedTensor,
  type SftkError,
  type ShardIndex,
  type ShardPlan,
  type ShardPolicy,
  type TensorInfo,
  type TensorMatch,
  type TensorNameMatch,
  type TensorQuery,
  type TensorQueryResult,
  type TensorRename,
  type RenamePlan,
  type ShardSummary,
  type ShardSetIssue,
  type ShardSetValidation,
  type SplitSummary,
  type ValidatedFile,
}

///|
/// Parse a header and validate it against the complete data-buffer length.
pub fn parse_and_validate(
  header_bytes : Bytes,
  data_len~ : UInt64,
  limits~ : @model.ParseLimits,
) -> @model.ValidatedFile raise @model.SftkError {
  let header = @codec.parse_header(header_bytes, limits~)
  @codec.validate_layout(header, data_len~)
}

///|
/// Calculate header-only information for a validated container.
pub fn inspect(file : @model.ValidatedFile) -> @model.FileInfo {
  @codec.inspect(file)
}

///|
/// Produce typed structural changes without reading tensor data.
pub fn diff_files(
  left : @model.ValidatedFile,
  right : @model.ValidatedFile,
) -> @model.DiffReport {
  @diff.diff(left, right)
}

///|
/// Plan deterministic whole-tensor shards.
pub fn plan_split(
  file : @model.ValidatedFile,
  policy : @model.ShardPolicy,
) -> @model.ShardPlan raise @model.SftkError {
  @shard.plan_split(file, policy)
}

///|
/// Validate an index and its shard snapshots, then plan a canonical merge.
pub fn plan_merge(
  index : @model.ShardIndex,
  files : ArrayView[@model.ValidatedFile],
) -> @model.MergePlan raise @model.SftkError {
  @shard.plan_merge(index, files)
}

///|
/// Query tensor declarations and checked sizes without reading payload data.
pub fn query_tensors(
  file : @model.ValidatedFile,
  query : @model.TensorQuery,
) -> @model.TensorQueryResult raise @model.SftkError {
  @query.query_tensors(file, query)
}

///|
/// Find one exact tensor declaration and its header-only size information.
pub fn find_tensor(
  file : @model.ValidatedFile,
  name : String,
) -> @model.TensorMatch? raise @model.SftkError {
  @query.find_tensor(file, name)
}

///|
/// Summarize dtype, shape rank, scalar, and largest-tensor information.
pub fn summarize_layout(
  file : @model.ValidatedFile,
) -> @model.LayoutSummary raise @model.SftkError {
  @analysis.summarize_layout(file)
}

///|
/// Validate reserved string metadata against a caller-owned policy.
pub fn validate_metadata(
  file : @model.ValidatedFile,
  policy : @model.MetadataPolicy,
) -> @model.MetadataValidation raise @model.SftkError {
  @analysis.validate_metadata(file, policy)
}

///|
/// Validate a transactional header-only tensor rename operation.
pub fn plan_renames(
  file : @model.ValidatedFile,
  renames : ArrayView[@model.TensorRename],
) -> @model.RenamePlan raise @model.SftkError {
  @transform.plan_renames(file, renames)
}

///|
/// Apply a validated rename plan while preserving every payload range.
pub fn apply_rename_plan(
  file : @model.ValidatedFile,
  plan : @model.RenamePlan,
) -> @model.ValidatedFile raise @model.SftkError {
  @transform.apply_rename_plan(file, plan)
}

///|
/// Summarize tensor distribution in a deterministic split plan.
pub fn summarize_split(plan : @model.ShardPlan) -> @model.SplitSummary {
  @shard.summarize_split(plan)
}

///|
/// Verify a named shard set against its index before a merge is attempted.
pub fn validate_shard_set(
  index : @model.ShardIndex,
  shards : ArrayView[(String, @model.ValidatedFile)],
) -> @model.ShardSetValidation {
  @analysis.validate_shard_set(index, shards)
}
