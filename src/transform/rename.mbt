///|
fn find_tensor(
  file : @model.ValidatedFile,
  name : String,
) -> @model.TensorInfo? {
  for tensor in file.header.tensors {
    if tensor.name == name {
      return Some(tensor)
    }
  }
  None
}

///|
fn valid_tensor_name(name : String) -> Bool {
  !name.is_empty() && name != "__metadata__" && !name.contains("\u0000")
}

///|
fn map_renames(plan : @model.RenamePlan) -> Map[String, String] {
  let mapping : Map[String, String] = Map([])
  for rename in plan.renames {
    mapping[rename.from] = rename.to
  }
  mapping
}

///|
/// Validate a complete header-only tensor rename transaction.  Every source
/// name must exist exactly once, no target may collide with unchanged names,
/// and the reserved metadata key is never accepted as a tensor target.
pub fn plan_renames(
  file : @model.ValidatedFile,
  renames : ArrayView[@model.TensorRename],
) -> @model.RenamePlan raise @model.SftkError {
  let source_names : Map[String, Bool] = Map([])
  let target_names : Map[String, Bool] = Map([])
  for rename in renames {
    if !valid_tensor_name(rename.from) || !valid_tensor_name(rename.to) {
      raise @model.SftkError::InvalidOptions(
        message="tensor rename names must be nonempty and cannot be __metadata__ or contain NUL",
      )
    }
    if !source_names.contains(rename.from) {
      match find_tensor(file, rename.from) {
        Some(_) => ()
        None =>
          raise @model.SftkError::InvalidTensor(
            name=rename.from,
            message="rename source tensor does not exist",
          )
      }
    } else {
      raise @model.SftkError::InvalidOptions(
        message="rename sources must be unique",
      )
    }
    if target_names.contains(rename.to) {
      raise @model.SftkError::InvalidOptions(
        message="rename targets must be unique",
      )
    }
    source_names[rename.from] = true
    target_names[rename.to] = true
  }
  for tensor in file.header.tensors {
    if !source_names.contains(tensor.name) && target_names.contains(tensor.name) {
      raise @model.SftkError::InvalidTensor(
        name=tensor.name,
        message="rename target collides with an unchanged tensor",
      )
    }
  }
  let ordered : Array[@model.TensorRename] = []
  for rename in renames {
    if rename.from != rename.to {
      ordered.push(rename)
    }
  }
  ordered.sort_by((left, right) => left.from.lexical_compare(right.from))
  { changed: !ordered.is_empty(), renames: ordered }
}

///|
/// Apply a validated rename plan to a header snapshot.  Tensor payload ranges,
/// dtype, shape, metadata and source order are preserved exactly.
pub fn apply_rename_plan(
  file : @model.ValidatedFile,
  plan : @model.RenamePlan,
) -> @model.ValidatedFile raise @model.SftkError {
  let mapping = map_renames(plan)
  let proposed : Array[@model.TensorRename] = []
  for rename in plan.renames {
    proposed.push(rename)
  }
  let canonical = plan_renames(file, proposed)
  if canonical.renames != plan.renames || canonical.changed != plan.changed {
    raise @model.SftkError::InvalidOptions(
      message="rename plan is not canonical for this validated file",
    )
  }
  let tensors : Array[@model.TensorInfo] = []
  for tensor in file.header.tensors {
    let name = match mapping.get(tensor.name) {
      Some(value) => value
      None => tensor.name
    }
    tensors.push({
      name,
      dtype: tensor.dtype,
      shape: tensor.shape.copy(),
      data_start: tensor.data_start,
      data_end: tensor.data_end,
      source_order: tensor.source_order,
    })
  }
  {
    header: { tensors, metadata: file.header.metadata.copy() },
    data_len: file.data_len,
  }
}

///|
/// Build rename declarations for every tensor with an exact textual prefix.
/// The declarations are still passed through `plan_renames`, ensuring collision
/// checks are identical to explicitly supplied renames.
pub fn plan_prefix_rename(
  file : @model.ValidatedFile,
  from_prefix : String,
  to_prefix : String,
) -> @model.RenamePlan raise @model.SftkError {
  if from_prefix.is_empty() {
    raise @model.SftkError::InvalidOptions(
      message="rename source prefix cannot be empty",
    )
  }
  let renames : Array[@model.TensorRename] = []
  for tensor in file.header.tensors {
    if tensor.name.has_prefix(from_prefix) {
      renames.push({
        from: tensor.name,
        to: to_prefix + tensor.name[from_prefix.length():].to_owned(),
      })
    }
  }
  plan_renames(file, renames)
}

///|
/// Return a lexical list of target tensor names after a validated rename plan.
pub fn renamed_tensor_names(
  file : @model.ValidatedFile,
  plan : @model.RenamePlan,
) -> Array[String] raise @model.SftkError {
  let output = apply_rename_plan(file, plan).header.tensors.map(tensor => {
    tensor.name
  })
  output.sort_by((left, right) => left.lexical_compare(right))
  output
}
