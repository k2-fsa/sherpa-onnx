#!/usr/bin/env bash
#
# This scripts shows how to test java for sherpa-onnx
# Note: This scripts runs only on Linux and macOS

set -e

log() {
  # This function is from espnet
  local fname=${BASH_SOURCE[1]##*/}
  echo -e "$(date '+%Y-%m-%d %H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}



 
echo "PATH: $PATH"



 

log "------------------------------------------------------------"
log "Run download model"
log "------------------------------------------------------------"

repo_url=https://huggingface.co/csukuangfj/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20
log "Start testing ${repo_url}"
repo=$(basename $repo_url)
log "download dir is $(basename $repo_url)" 
if [ ! -d $repo ];then
	log "Download pretrained model and test-data from $repo_url"

	GIT_LFS_SKIP_SMUDGE=1 git clone $repo_url
	pushd $repo
	git lfs pull --include "*.onnx"
	ls -lh *.onnx
	popd
	ln -s $repo/test_wavs/0.wav hotwords.wav

fi

log $(pwd)

sed -e 's?/sherpa/?'$(pwd)'/?g'  modelconfig.cfg > modeltest.cfg

log "display model cfg"
cat modeltest.cfg

cd ..

export JAVA_HOME=$(readlink -f /usr/bin/javac | sed "s:/bin/javac::")

mkdir -p build
cd build

cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON -DSHERPA_ONNX_ENABLE_JNI=ON ..

make -j4
ls -lh lib

export LD_LIBRARY_PATH=$PWD/build/lib:$LD_LIBRARY_PATH

cd ../java-api-examples

make all

make runfile

echo "礼 拜 二" > hotwords.txt

sed -i 's/hotwords_file=/hotwords_file=hotwords.txt/g' modeltest.cfg

make runhotwords
