# Assets

Please download a forced-alignment model package from
https://github.com/k2-fsa/sherpa-onnx/releases/tag/fa-models
and extract it here. The model directory must be a direct child of `assets/`:

```
assets/
  zh-aishell3-pinyin/
    meta.json
    final.alimdl
    tree
    lda.mat
    phones.txt      # optional
    lexicon.txt
    zh.wav          # the package's test wav
    README.md
```

For example:

```bash
cd wasm/forced-alignment/assets
wget -q https://github.com/k2-fsa/sherpa-onnx/releases/download/fa-models/zh-aishell3-pinyin.tar.bz2
tar xvf zh-aishell3-pinyin.tar.bz2
rm zh-aishell3-pinyin.tar.bz2
```

You may put several model directories here; `app-fa.js` (and the web app)
select one of them by its index (see `modelIndex` in `app-fa.js`). For a
published app, put only the model it should ship with.

The contents of `assets/` are packed into `sherpa-onnx-wasm-main-fa.data` at
build time (`--preload-file`) and mounted at `/` in the wasm virtual FS.
