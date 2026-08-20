///|
/// A minimal native consumer: validate a real SafeTensors path through the
/// public native API and print header-only facts.
async fn main {
  let argv = @env.args()
  if argv.length() < 2 {
    println("usage: moon run src/examples/validate --target native -- FILE")
    return
  }
  let file = @native.inspect_file(argv[1], @model.ParseLimits::default()) catch {
    _ => {
      println("invalid SafeTensors input")
      return
    }
  }
  println(
    "valid SafeTensors: tensors=" +
    file.header.tensors.length().to_string() +
    " metadata=" +
    file.header.metadata.length().to_string() +
    " data_bytes=" +
    file.data_len.to_string(),
  )
}
