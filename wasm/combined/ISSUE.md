# Sherpa-ONNX WASM Combined Module Issue: Inconsistent Shared Module Context & HEAPF32 Access Failure

## Problem Description

The core issue is a fundamental limitation in the current Sherpa-ONNX WASM combined module architecture: **it fails to establish a reliably shared and synchronized WebAssembly (WASM) runtime context across the multiple, sequentially loaded JavaScript component files** (`sherpa-onnx-combined-core.js`, `sherpa-onnx-combined-asr.js`, etc.). Specifically, essential JavaScript views onto the WASM memory, like `HEAPF32`, are not consistently accessible across these script boundaries.

### Background: WASM Memory and HEAP Views

-   **WASM Linear Memory**: WebAssembly modules operate on a contiguous block of memory.
-   **Emscripten HEAP Views**: To allow JavaScript to interact with this memory, Emscripten (the compiler used) creates typed array views (e.g., `Float32Array`, `Int8Array`) pointing directly into this memory block. These views are assigned to the global `Module` object as properties like `Module.HEAPF32`, `Module.HEAP8`, `Module.HEAPU8`, etc.
-   **Initialization**: These `HEAP*` views are crucial for JS-WASM communication. They are normally initialized by the main Emscripten glue code (`sherpa-onnx-wasm-combined.js` in this case) *after* the WASM memory buffer is allocated but *before* or *during* the `Module.onRuntimeInitialized` callback, signifying the runtime is ready.

### Detailed Explanation of the Failure

1.  **Context/Scope Separation & HEAP Inaccessibility**: Despite all scripts referencing the global `window.Module`, they appear to operate within distinct execution contexts. Crucially, the standard `HEAP*` memory views (especially `HEAPF32`, essential for ASR audio data transfer) that *should* be initialized on `window.Module` by the main glue code are **not accessible or visible** within the context of the subsequently loaded component scripts (e.g., `sherpa-onnx-combined-core.js`). The repeated log messages `No suitable memory buffer found` and `HEAPF32 exists: false` from within `sherpa-onnx-combined-core.js` are direct evidence of this failure.

2.  **Sequential Loading Barrier**: The architecture loads functional components (ASR, VAD, etc.) as separate JS files *after* the main WASM module and its memory are expected to initialize. This sequential loading creates context boundaries that prevent the component scripts from accessing the already initialized `HEAP*` views on the `Module` object.

3.  **Initialization Callbacks Ineffective Across Contexts**: Callbacks like `onRuntimeInitialized` might fire in the main glue code's context, but this readiness state (including the availability of initialized `HEAP*` views) does not reliably propagate to the separate contexts of the component scripts.

4.  **Runtime Errors**: Consequently, operations requiring direct JS interaction with WASM memory via these views fail. For example, `OnlineStream.acceptWaveform` in ASR needs to write to `HEAPF32`. Since `HEAPF32` is inaccessible in the `asr.js` or `core.js` context, this fails, leading to downstream errors like `TypeError: asr.createStream is not a function` (as the recognizer likely failed during its own initialization which might require memory access).

5.  **Selective Functionality Failure (Evidence)**: Functionalities like TTS (`tts.html`) appear less affected. This suggests their JS-WASM interaction pattern doesn't critically rely on the *JavaScript context* having direct write access to `HEAPF32` in the same way streaming ASR does, further supporting that the issue is specific to the accessibility of these memory views across script contexts.

### Impact

-   **Unreliable Functionality**: Core features requiring JS access to WASM memory views (like streaming ASR via `HEAPF32`) fail reliably.
-   **Debugging Dead End**: Standard synchronization techniques are ineffective because the fundamental issue is the inaccessibility of necessary `HEAP*` views due to context separation.

### Architectural Root Cause

The multi-file JavaScript approach, combined with Emscripten's standard output, fails to guarantee that the essential `HEAP*` memory views initialized on the `Module` object are accessible from the separate JavaScript files loaded later. Each script effectively gets a view of the `Module` object that might lack these critical, dynamically initialized properties.

### Potential Solutions

1.  **Unified Script (Likely Viable but with Drawbacks)**: Combine *all* JavaScript glue code (core, ASR, VAD, TTS, etc.) and the main Emscripten `Module` interaction into a **single, large JavaScript file**. This forces all code into the same execution context, ensuring consistent access to the initialized `Module` object and its `HEAP*` views. **Drawback**: Creates a potentially very large initial JS file, impacting load performance.

2.  **WASM Module Re-architecture (Complex)**: Fundamentally change how the C++ code is compiled, perhaps using Emscripten features explicitly designed for better JS module interoperability (e.g., `MODULARIZE=1`, ES6 modules output) that might handle state sharing differently. This likely requires significant changes to the build process and C++/JS interface.

3.  ~~Delayed Functionality Binding~~ (Proven Ineffective): Delaying execution doesn't solve the problem that the necessary `HEAP*` views are fundamentally inaccessible from within the component script contexts.

This issue highlights a significant architectural challenge. The **Unified Script** approach appears the most practical path forward within the existing build system, despite performance implications.
