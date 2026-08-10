///|
/// Stable failures for untrusted container parsing, planning, and native I/O.
pub(all) suberror SftkError {
  Truncated(phase~ : String, expected~ : UInt64, actual~ : UInt64)
  HeaderTooLarge(length~ : UInt64, limit~ : UInt64)
  InvalidUtf8(offset~ : UInt64)
  InvalidJson(offset~ : UInt64, message~ : String)
  DuplicateKey(path~ : String, key~ : String)
  InvalidTensor(name~ : String, message~ : String)
  UnknownDtype(name~ : String, dtype~ : String)
  ArithmeticOverflow(context~ : String)
  InvalidLayout(message~ : String)
  InvalidIndex(message~ : String)
  IoFailure(operation~ : String, path~ : String, message~ : String)
  Conflict(path~ : String, message~ : String)
  InvalidOptions(message~ : String)
} derive(Debug, Eq)

///|
/// The SafeTensors dtype matrix supported by format revision 1.
///
/// Widths are measured in bits because the current upstream format also
/// defines packed F4 and F6 element formats.
pub(all) enum Dtype {
  BOOL
  U8
  I8
  F4
  F6_E2M3
  F6_E3M2
  F8_E5M2
  F8_E4M3
  F8_E8M0
  F8_E4M3FNUZ
  F8_E5M2FNUZ
  U16
  I16
  F16
  BF16
  U32
  I32
  F32
  C64
  U64
  I64
  F64
} derive(Debug, Eq)

///|
/// Hard limits for untrusted SafeTensors headers.
pub(all) struct ParseLimits {
  max_header_bytes : UInt64
  max_tensors : Int
  max_metadata_entries : Int
  max_depth : Int
  max_string_bytes : Int
} derive(Debug, Eq)

///|
/// One tensor declaration before layout validation.
pub(all) struct TensorInfo {
  name : String
  dtype : Dtype
  shape : Array[UInt64]
  data_start : UInt64
  data_end : UInt64
  source_order : Int
} derive(Debug, Eq)

///|
/// One arbitrary string metadata value from the reserved metadata object.
pub(all) struct MetadataEntry {
  key : String
  value : String
} derive(Debug, Eq)

///|
/// Strictly parsed header with source-order tensor entries and metadata.
pub(all) struct HeaderSnapshot {
  tensors : Array[TensorInfo]
  metadata : Array[MetadataEntry]
} derive(Debug, Eq)

///|
/// Header plus checked dtype/shape sizes and complete data-buffer layout.
pub(all) struct ValidatedFile {
  header : HeaderSnapshot
  data_len : UInt64
} derive(Debug, Eq)

///|
/// Header-only statistics that never include tensor data bytes.
pub(all) struct FileInfo {
  tensor_count : Int
  metadata_count : Int
  data_bytes : UInt64
  dtype_counts : Array[(String, Int)]
} derive(Debug, Eq)

///|
/// One typed structural difference between valid containers.
pub(all) enum DiffKind {
  TensorAdded
  TensorRemoved
  DtypeChanged
  ShapeChanged
  DataRangeChanged
  MetadataAdded
  MetadataRemoved
  MetadataChanged
} derive(Debug, Eq)

///|
/// A named structural difference with deterministic old and new summaries.
pub(all) struct DiffEntry {
  kind : DiffKind
  name : String
  before : String?
  after : String?
} derive(Debug, Eq)

///|
/// Typed, deterministically ordered structural changes.
pub(all) struct DiffReport {
  entries : Array[DiffEntry]
  equal : Bool
} derive(Debug, Eq)

///|
/// Deterministic maximum-bytes and maximum-tensors split policy.
pub(all) struct ShardPolicy {
  max_bytes : UInt64
  max_tensors : Int
} derive(Debug, Eq)

///|
/// One whole-tensor byte range copied into an output shard.
pub(all) struct PlannedTensor {
  name : String
  source_start : UInt64
  source_end : UInt64
  output_start : UInt64
  output_end : UInt64
} derive(Debug, Eq)

///|
/// A deterministic output shard and its original tensor ranges.
pub(all) struct PlannedShard {
  name : String
  tensors : Array[PlannedTensor]
  data_bytes : UInt64
} derive(Debug, Eq)

///|
/// Versioned tensor-name to shard-name mapping.
pub(all) struct ShardIndex {
  format : String
  total_size : UInt64
  weight_map : Array[(String, String)]
} derive(Debug, Eq)

///|
/// Source-identity-bound whole-tensor byte-range split plan.
pub(all) struct ShardPlan {
  source_data_bytes : UInt64
  source_tensor_count : Int
  shards : Array[PlannedShard]
  index : ShardIndex
} derive(Debug, Eq)

///|
/// A tensor's source shard and canonical range in a merged output file.
pub(all) struct MergeEntry {
  name : String
  shard_name : String
  source_start : UInt64
  source_end : UInt64
  output_start : UInt64
  output_end : UInt64
} derive(Debug, Eq)

///|
/// Validated multi-shard merge plan with canonical output offsets and ranges.
pub(all) struct MergePlan {
  entries : Array[MergeEntry]
  output_data_bytes : UInt64
} derive(Debug, Eq)

///|
/// Native buffer, overwrite, and symbolic-link policy.
pub(all) struct ExecutionOptions {
  buffer_bytes : Int
  overwrite : Bool
  follow_symlinks : Bool
} derive(Debug, Eq)

///|
/// Conservative default limits, including a 100 MiB bounded header.
pub fn ParseLimits::default() -> ParseLimits {
  {
    max_header_bytes: 100UL * 1024UL * 1024UL,
    max_tensors: 1_000_000,
    max_metadata_entries: 100_000,
    max_depth: 64,
    max_string_bytes: 16 * 1024 * 1024,
  }
}

///|
/// Construct a deterministic tensor-granularity split policy.
pub fn ShardPolicy::new(
  max_bytes : UInt64,
  max_tensors : Int,
) -> ShardPolicy raise SftkError {
  if max_bytes == 0UL || max_tensors <= 0 {
    raise InvalidOptions(
      message="split policy requires positive max_bytes and max_tensors",
    )
  }
  { max_bytes, max_tensors }
}

///|
/// Construct conservative native execution options without overwrite.
pub fn ExecutionOptions::default() -> ExecutionOptions {
  { buffer_bytes: 1024 * 1024, overwrite: false, follow_symlinks: false }
}

///|
/// Return the canonical SafeTensors spelling of a dtype.
pub fn Dtype::name(self : Dtype) -> String {
  match self {
    BOOL => "BOOL"
    U8 => "U8"
    I8 => "I8"
    F4 => "F4"
    F6_E2M3 => "F6_E2M3"
    F6_E3M2 => "F6_E3M2"
    F8_E5M2 => "F8_E5M2"
    F8_E4M3 => "F8_E4M3"
    F8_E8M0 => "F8_E8M0"
    F8_E4M3FNUZ => "F8_E4M3FNUZ"
    F8_E5M2FNUZ => "F8_E5M2FNUZ"
    U16 => "U16"
    I16 => "I16"
    F16 => "F16"
    BF16 => "BF16"
    U32 => "U32"
    I32 => "I32"
    F32 => "F32"
    C64 => "C64"
    U64 => "U64"
    I64 => "I64"
    F64 => "F64"
  }
}

///|
/// Return the exact storage width of one dtype element in bits.
pub fn Dtype::bit_width(self : Dtype) -> UInt64 {
  match self {
    F4 => 4UL
    F6_E2M3 | F6_E3M2 => 6UL
    BOOL | U8 | I8 | F8_E5M2 | F8_E4M3 | F8_E8M0 | F8_E4M3FNUZ | F8_E5M2FNUZ =>
      8UL
    U16 | I16 | F16 | BF16 => 16UL
    U32 | I32 | F32 => 32UL
    C64 | U64 | I64 | F64 => 64UL
  }
}

///|
/// Parse an exact canonical dtype name from the supported format revision.
pub fn Dtype::parse(name : String) -> Dtype? {
  match name {
    "BOOL" => Some(BOOL)
    "U8" => Some(U8)
    "I8" => Some(I8)
    "F4" => Some(F4)
    "F6_E2M3" => Some(F6_E2M3)
    "F6_E3M2" => Some(F6_E3M2)
    "F8_E5M2" => Some(F8_E5M2)
    "F8_E4M3" => Some(F8_E4M3)
    "F8_E8M0" => Some(F8_E8M0)
    "F8_E4M3FNUZ" => Some(F8_E4M3FNUZ)
    "F8_E5M2FNUZ" => Some(F8_E5M2FNUZ)
    "U16" => Some(U16)
    "I16" => Some(I16)
    "F16" => Some(F16)
    "BF16" => Some(BF16)
    "U32" => Some(U32)
    "I32" => Some(I32)
    "F32" => Some(F32)
    "C64" => Some(C64)
    "U64" => Some(U64)
    "I64" => Some(I64)
    "F64" => Some(F64)
    _ => None
  }
}

///|
/// A composable name predicate for tensor discovery without loading tensor data.
pub(all) enum TensorNameMatch {
  AnyName
  ExactName(value~ : String)
  Prefix(value~ : String)
  Suffix(value~ : String)
  Contains(value~ : String)
} derive(Debug, Eq)

///|
/// Header-only tensor query with optional dtype/rank/byte-size restrictions.
pub(all) struct TensorQuery {
  name_match : TensorNameMatch
  dtypes : Array[Dtype]
  rank : Int?
  min_bytes : UInt64?
  max_bytes : UInt64?
  require_metadata_key : String?
} derive(Debug, Eq)

///|
/// One query match annotated with checked logical element and storage counts.
pub(all) struct TensorMatch {
  tensor : TensorInfo
  element_count : UInt64
  byte_count : UInt64
} derive(Debug, Eq)

///|
/// Deterministic query output and query-wide aggregate counts.
pub(all) struct TensorQueryResult {
  matches : Array[TensorMatch]
  total_elements : UInt64
  total_bytes : UInt64
} derive(Debug, Eq)

///|
/// A dtype-level aggregate useful for size and model architecture inspection.
pub(all) struct DtypeSummary {
  dtype : Dtype
  tensor_count : Int
  element_count : UInt64
  byte_count : UInt64
} derive(Debug, Eq)

///|
/// Shape and layout aggregates which never read tensor data payloads.
pub(all) struct LayoutSummary {
  tensor_count : Int
  scalar_count : Int
  max_rank : Int
  total_elements : UInt64
  total_bytes : UInt64
  dtype_summaries : Array[DtypeSummary]
  largest_tensor : TensorMatch?
} derive(Debug, Eq)

///|
/// A metadata policy evaluated only against the reserved string map.
pub(all) struct MetadataPolicy {
  required_keys : Array[String]
  forbidden_keys : Array[String]
  max_value_bytes : Int
  allow_empty_values : Bool
} derive(Debug, Eq)

///|
/// One metadata policy violation with no tensor-data disclosure.
pub(all) struct MetadataIssue {
  key : String?
  message : String
} derive(Debug, Eq)

///|
/// Deterministic metadata policy result.
pub(all) struct MetadataValidation {
  issues : Array[MetadataIssue]
  valid : Bool
} derive(Debug, Eq)

///|
/// A header-only rename declaration.  It is applied transactionally by a
/// rename plan and never changes payload byte order.
pub(all) struct TensorRename {
  from : String
  to : String
} derive(Debug, Eq)

///|
/// Validated rename transformation for a SafeTensors header.
pub(all) struct RenamePlan {
  renames : Array[TensorRename]
  changed : Bool
} derive(Debug, Eq)

///|
/// One deterministic shard size/proportion entry for split-plan review.
pub(all) struct ShardSummary {
  name : String
  tensor_count : Int
  data_bytes : UInt64
  min_tensor_bytes : UInt64
  max_tensor_bytes : UInt64
} derive(Debug, Eq)

///|
/// Aggregate split-plan balance evidence, independent from disk I/O.
pub(all) struct SplitSummary {
  shard_count : Int
  data_bytes : UInt64
  largest_shard_bytes : UInt64
  smallest_shard_bytes : UInt64
  shards : Array[ShardSummary]
} derive(Debug, Eq)

///|
/// One structural inconsistency detected across an indexed shard set.
pub(all) struct ShardSetIssue {
  shard_name : String?
  tensor_name : String?
  message : String
} derive(Debug, Eq)

///|
/// Deterministic validation result for named validated shard snapshots.
pub(all) struct ShardSetValidation {
  issues : Array[ShardSetIssue]
  total_data_bytes : UInt64
  valid : Bool
} derive(Debug, Eq)

///|
/// Construct an unconstrained header-only tensor query.
pub fn TensorQuery::all() -> TensorQuery {
  {
    name_match: AnyName,
    dtypes: [],
    rank: None,
    min_bytes: None,
    max_bytes: None,
    require_metadata_key: None,
  }
}

///|
/// Construct a metadata policy which permits any string metadata map.
pub fn MetadataPolicy::default() -> MetadataPolicy {
  {
    required_keys: [],
    forbidden_keys: [],
    max_value_bytes: 16 * 1024 * 1024,
    allow_empty_values: true,
  }
}
