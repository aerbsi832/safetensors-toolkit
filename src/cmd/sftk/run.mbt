///|
fn index_parent(path : String) -> String {
  let chars = path.to_array()
  let mut index = chars.length() - 1
  while index >= 0 && chars[index] != '/' && chars[index] != '\\' {
    index -= 1
  }
  if index < 0 {
    "."
  } else if index == 0 {
    "/"
  } else {
    String::from_array(chars[:index])
  }
}

///|
fn join_path(left : String, right : String) -> String {
  if left.has_suffix("/") {
    left + right
  } else {
    left + "/" + right
  }
}

///|
fn contains_text(values : Array[String], wanted : String) -> Bool {
  for value in values {
    if value == wanted {
      return true
    }
  }
  false
}

///|
fn ordered_shard_paths(
  index_path : String,
  index : @model.ShardIndex,
) -> Array[String] {
  let names : Array[String] = []
  for pair in index.weight_map {
    if !contains_text(names, pair.1) {
      names.push(pair.1)
    }
  }
  names.sort_by((left, right) => left.lexical_compare(right))
  let parent = index_parent(index_path)
  names.map(name => join_path(parent, name))
}

///|
/// Execute the CLI contract and return its documented process exit code.
pub async fn run(args : ArrayView[String]) -> Int {
  let request = parse_args(args) catch {
    error => {
      stderr_line("error: " + error_text(error))
      return 2
    }
  }
  execute(request) catch {
    error => {
      stderr_line("error: " + error_text(error))
      return error_exit(error)
    }
  }
}

///|
async fn execute(request : Request) -> Int raise @model.SftkError {
  match request {
    Help => {
      stdout_line(usage())
      0
    }
    Version => {
      stdout_line("sftk 0.1.0")
      0
    }
    Validate(path~, format~) | Info(path~, format~) => {
      let file = @native.inspect_file(path, @model.ParseLimits::default())
      stdout_line(render_info(file, format))
      0
    }
    Query(path~, selector~, format~) => {
      let file = @native.inspect_file(path, @model.ParseLimits::default())
      let query = @query.parse_selector(selector)
      stdout_line(
        render_query(
          @query.render_selector(query),
          @query.query_tensors(file, query),
          format,
        ),
      )
      0
    }
    Diff(left~, right~, format~) => {
      let report = @diff.diff(
        @native.inspect_file(left, @model.ParseLimits::default()),
        @native.inspect_file(right, @model.ParseLimits::default()),
      )
      stdout_line(render_diff(report, format))
      if report.equal {
        0
      } else {
        1
      }
    }
    Split(path~, max_bytes~, max_tensors~, output_dir~, write~, force~, format~) => {
      let file = @native.inspect_file(path, @model.ParseLimits::default())
      let plan = @shard.plan_split(
        file,
        @model.ShardPolicy::new(max_bytes, max_tensors),
      )
      stdout_line(render_split(plan, write, format))
      if write {
        let options = @model.ExecutionOptions::default()
        @native.execute_split(path, output_dir, plan, {
          ..options,
          overwrite: force,
        })
      }
      0
    }
    Merge(index_path~, output~, write~, force~, format~) => {
      let index = @native.load_index_file(
        index_path,
        @model.ParseLimits::default(),
      )
      let paths = ordered_shard_paths(index_path, index)
      let files : Array[@model.ValidatedFile] = []
      for path in paths {
        files.push(@native.inspect_file(path, @model.ParseLimits::default()))
      }
      let plan = @shard.plan_merge(index, files)
      stdout_line(render_merge(plan, write, format))
      if write {
        let options = @model.ExecutionOptions::default()
        @native.execute_merge(paths, output, plan, {
          ..options,
          overwrite: force,
        })
      }
      0
    }
    RenamePrefix(
      path~,
      from_prefix~,
      to_prefix~,
      output~,
      write~,
      force~,
      format~
    ) => {
      let options = @model.ExecutionOptions::default()
      let plan = @native.plan_prefix_rename_file(
        path, from_prefix, to_prefix, options,
      )
      stdout_line(render_rename_prefix(plan, write, format))
      if write {
        @native.execute_rename(path, output, plan, {
          ..options,
          overwrite: force,
        })
      }
      0
    }
  }
}
