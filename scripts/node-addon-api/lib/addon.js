const os = require('os');
const platform_arch = `${os.platform()}-${os.arch()}`;
const possible_paths = [
  '../build/Release/sherpa-onnx.node',
  '../build/Debug/sherpa-onnx.node',
  `./node_modules/sherpa-onnx-${platform_arch}/sherpa-onnx.node`,
  `../sherpa-onnx-${platform_arch}/sherpa-onnx.node`,
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
  let msg =
      `Could not find sherpa-onnx. Tried\n\n  ${possible_paths.join('\n  ')}\n`
  if (os.platform() == 'darwin' &&
      !process.env.DYLD_LIBRARY_PATH.includes(
          `node_modules/sherpa-onnx-${platform_arch}`)) {
    msg +=
        'Please remeber to set the following environment variable and try again:\n';

    msg += `export DYLD_LIBRARY_PATH=${
        process.env.PWD}/node_modules/sherpa-onnx-${platform_arch}`;

    msg += ':$DYLD_LIBRARY_PATH\n';
  }

  if (os.platform() == 'linux' &&
      !process.env.LD_LIBRARY_PATH.includes(
          `node_modules/sherpa-onnx-${platform_arch}`)) {
    msg +=
        'Please remeber to set the following environment variable and try again:\n';

    msg += `export LD_LIBRARY_PATH=${
        process.env.PWD}/node_modules/sherpa-onnx-${platform_arch}`;

    msg += ':$LD_LIBRARY_PATH\n';
  }

  throw new Error(msg)
}
