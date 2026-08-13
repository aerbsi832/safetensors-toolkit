///|
fn summarize_shard(shard : @model.PlannedShard) -> @model.ShardSummary {
  let mut minimum : UInt64? = None
  let mut maximum = 0UL
  for tensor in shard.tensors {
    let bytes = tensor.output_end - tensor.output_start
    match minimum {
      Some(value) if bytes >= value => ()
      _ => minimum = Some(bytes)
    }
    if bytes > maximum {
      maximum = bytes
    }
  }
  let min_tensor_bytes = match minimum {
    Some(value) => value
    None => 0UL
  }
  {
    name: shard.name,
    tensor_count: shard.tensors.length(),
    data_bytes: shard.data_bytes,
    min_tensor_bytes,
    max_tensor_bytes: maximum,
  }
}

///|
/// Summarize a split plan's shard balance without changing its ordering or
/// attempting to infer payload compressibility.
pub fn summarize_split(plan : @model.ShardPlan) -> @model.SplitSummary {
  let shards : Array[@model.ShardSummary] = []
  let mut largest = 0UL
  let mut smallest : UInt64? = None
  for shard in plan.shards {
    let summary = summarize_shard(shard)
    if summary.data_bytes > largest {
      largest = summary.data_bytes
    }
    match smallest {
      Some(value) if summary.data_bytes >= value => ()
      _ => smallest = Some(summary.data_bytes)
    }
    shards.push(summary)
  }
  let smallest_shard_bytes = match smallest {
    Some(value) => value
    None => 0UL
  }
  {
    shard_count: shards.length(),
    data_bytes: plan.source_data_bytes,
    largest_shard_bytes: largest,
    smallest_shard_bytes,
    shards,
  }
}

///|
/// Find a tensor's planned output shard by exact name.  The index mapping is
/// authoritative for a plan, but this helper exposes the corresponding rich
/// range information without an external map implementation.
pub fn planned_tensor(
  plan : @model.ShardPlan,
  name : String,
) -> (@model.PlannedShard, @model.PlannedTensor)? {
  for shard in plan.shards {
    for tensor in shard.tensors {
      if tensor.name == name {
        return Some((shard, tensor))
      }
    }
  }
  None
}
