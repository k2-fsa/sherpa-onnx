// Copyright (c)  2026  Xiaomi Corporation
import 'package:flutter/material.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();

  // Works on all platforms: loads native lib on desktop/mobile, WASM on web.
  // IMPORTANT: You must call initBindingsAsync() in every isolate that uses
  // sherpa-onnx APIs — including the main isolate and any worker isolates.
  await initBindingsAsync();

  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'sherpa-onnx hello world',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const VersionPage(),
    );
  }
}

class VersionPage extends StatelessWidget {
  const VersionPage({super.key});

  @override
  Widget build(BuildContext context) {
    final version = getVersion();
    final gitSha1 = getGitSha1();
    final gitDate = getGitDate();
    final onnxruntimeVersion = getOnnxruntimeVersion();

    return Scaffold(
      appBar: AppBar(
        title: const Text('sherpa-onnx hello world'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text('sherpa-onnx version: $version',
                style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            Text('Git SHA1: $gitSha1',
                style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            Text('Git date: $gitDate',
                style: Theme.of(context).textTheme.titleMedium),
            const SizedBox(height: 8),
            Text('onnxruntime version: $onnxruntimeVersion',
                style: Theme.of(context).textTheme.titleMedium),
          ],
        ),
      ),
    );
  }
}
