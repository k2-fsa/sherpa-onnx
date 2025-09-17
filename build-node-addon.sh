# Build Sherpa-Onnx + Node Addon

rm -rf build
mkdir build
cd build
cmake -DCMAKE_INSTALL_PREFIX=./install -DBUILD_SHARED_LIBS=ON ..
make -j install
export SHERPA_ONNX_INSTALL_DIR=$PWD/install #
export PKG_CONFIG_PATH=$PWD/install:$PKG_CONFIG_PATH
cd ../scripts/node-addon-api/
npm i
./node_modules/.bin/cmake-js compile --log-level verbose