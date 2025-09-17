# PHASE ONE - Build Sherpa-Onnx

rm -rf build-sherpa-onnx
mkdir build-sherpa-onnx
cd build-sherpa-onnx
cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=./install ..
make install
export SHERPA_ONNX_INSTALL_DIR=$PWD/install

# PHASE TWO - Build Node Addon

cd ..
rm -rf build-node-addon
mkdir build-node-addon
cd build-node-addon
cmake -DCMAKE_INSTALL_PREFIX=./install -DBUILD_SHARED_LIBS=ON ..
make -j install
export PKG_CONFIG_PATH=$PWD/install:$PKG_CONFIG_PATH
cd ../scripts/node-addon-api/
npm i
./node_modules/.bin/cmake-js compile --log-level verbose