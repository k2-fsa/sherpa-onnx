// Native stub for PunctWorker (not used on native).

typedef OnReadyCallback = void Function();
typedef OnResultCallback = void Function(String result, double elapsed);
typedef OnErrorCallback = void Function(String message);

class PunctWorker {
  PunctWorker({
    required OnReadyCallback onReady,
    required OnResultCallback onResult,
    required OnErrorCallback onError,
  });

  Future<void> init() async {}
  void punctuate({required String text}) {}
  void dispose() {}
}
