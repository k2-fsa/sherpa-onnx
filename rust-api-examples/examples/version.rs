use sherpa_onnx;

fn main() {
    println!("version:  {}", sherpa_onnx::version());
    println!("Git SHA1: {}", sherpa_onnx::git_sha1());
    println!("Git date: {}", sherpa_onnx::git_date());
}
