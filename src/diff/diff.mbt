///|
fn find_tensor(
  tensors : Array[@model.TensorInfo],
  name : String,
) -> @model.TensorInfo? {
  for tensor in tensors {
    if tensor.name == name {
      return Some(tensor)
    }
  }
  None
}

///|
fn find_metadata(
  entries : Array[@model.MetadataEntry],
  key : String,
) -> String? {
  for entry in entries {
    if entry.key == key {
      return Some(entry.value)
    }
  }
  None
}

///|
fn render_shape(shape : Array[UInt64]) -> String {
  "[" + shape.map(value => value.to_string()).join(",") + "]"
}

///|
fn range_text(tensor : @model.TensorInfo) -> String {
  "[" + tensor.data_start.to_string() + "," + tensor.data_end.to_string() + ")"
}

///|
fn kind_name(kind : @model.DiffKind) -> String {
  match kind {
    TensorAdded => "tensor_added"
    TensorRemoved => "tensor_removed"
    DtypeChanged => "dtype_changed"
    ShapeChanged => "shape_changed"
    DataRangeChanged => "data_range_changed"
    MetadataAdded => "metadata_added"
    MetadataRemoved => "metadata_removed"
    MetadataChanged => "metadata_changed"
  }
}

///|
fn compare_entries(left : @model.DiffEntry, right : @model.DiffEntry) -> Int {
  let by_name = left.name.lexical_compare(right.name)
  if by_name != 0 {
    by_name
  } else {
    kind_name(left.kind).lexical_compare(kind_name(right.kind))
  }
}

///|
/// Compare two valid containers without reading or interpreting tensor data.
pub fn diff(
  left : @model.ValidatedFile,
  right : @model.ValidatedFile,
) -> @model.DiffReport {
  let entries : Array[@model.DiffEntry] = []
  for tensor in left.header.tensors {
    match find_tensor(right.header.tensors, tensor.name) {
      None =>
        entries.push({
          kind: TensorRemoved,
          name: "tensor:" + tensor.name,
          before: Some(tensor.dtype.name() + render_shape(tensor.shape)),
          after: None,
        })
      Some(other) => {
        if tensor.dtype != other.dtype {
          entries.push({
            kind: DtypeChanged,
            name: "tensor:" + tensor.name,
            before: Some(tensor.dtype.name()),
            after: Some(other.dtype.name()),
          })
        }
        if tensor.shape != other.shape {
          entries.push({
            kind: ShapeChanged,
            name: "tensor:" + tensor.name,
            before: Some(render_shape(tensor.shape)),
            after: Some(render_shape(other.shape)),
          })
        }
        if tensor.data_start != other.data_start ||
          tensor.data_end != other.data_end {
          entries.push({
            kind: DataRangeChanged,
            name: "tensor:" + tensor.name,
            before: Some(range_text(tensor)),
            after: Some(range_text(other)),
          })
        }
      }
    }
  }
  for tensor in right.header.tensors {
    if find_tensor(left.header.tensors, tensor.name) is None {
      entries.push({
        kind: TensorAdded,
        name: "tensor:" + tensor.name,
        before: None,
        after: Some(tensor.dtype.name() + render_shape(tensor.shape)),
      })
    }
  }
  for entry in left.header.metadata {
    match find_metadata(right.header.metadata, entry.key) {
      None =>
        entries.push({
          kind: MetadataRemoved,
          name: "metadata:" + entry.key,
          before: Some(entry.value),
          after: None,
        })
      Some(other) if other != entry.value =>
        entries.push({
          kind: MetadataChanged,
          name: "metadata:" + entry.key,
          before: Some(entry.value),
          after: Some(other),
        })
      _ => ()
    }
  }
  for entry in right.header.metadata {
    if find_metadata(left.header.metadata, entry.key) is None {
      entries.push({
        kind: MetadataAdded,
        name: "metadata:" + entry.key,
        before: None,
        after: Some(entry.value),
      })
    }
  }
  entries.sort_by(compare_entries)
  { equal: entries.is_empty(), entries }
}
