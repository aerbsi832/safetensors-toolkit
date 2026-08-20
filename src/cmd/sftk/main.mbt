///|
/// Native executable entrypoint. `run` owns all user-facing diagnostics.
async fn main {
  let argv = @env.args()
  let args : Array[String] = []
  for index = 1; index < argv.length(); index = index + 1 {
    args.push(argv[index])
  }
  @sys.exit(run(args))
}
