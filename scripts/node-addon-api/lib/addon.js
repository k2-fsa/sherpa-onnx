const os = require('os');
const platform_arch = `${os.platform()}-${os.arch()}`;
const possible_paths = [
  '../build/Release/sherpa-onnx.node',
  '../build/Debug/sherpa-onnx.node',
  `./node_modules/sherpa-onnx-${platform_arch}/sherpa-onnx.node`,
];

let found = false;
for (const p of possible_paths) {
  try {
    console.log(p);
    module.exports = require(p);
    found = true;
    break;
  } catch (error) {
    // do nothing; try the next option
    ;
  }
}

if (!found) {
  throw new Error(
      `Could not find sherpa-onnx. Tried\n\n  ${possible_paths.join('\n  ')}\n`)
}
