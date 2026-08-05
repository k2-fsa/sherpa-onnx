// Copyright (c)  2024  Xiaomi Corporation
import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';

import './tts_screen.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'sherpa-onnx TTS Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  int _currentIndex = 0;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: IndexedStack(
        index: _currentIndex,
        children: const [
          TtsScreen(),
          InfoScreen(),
        ],
      ),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _currentIndex,
        onTap: (i) => setState(() => _currentIndex = i),
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.home), label: 'TTS'),
          BottomNavigationBarItem(icon: Icon(Icons.info), label: 'Info'),
        ],
      ),
    );
  }
}

class InfoScreen extends StatelessWidget {
  const InfoScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Info')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('sherpa-onnx TTS Demo'),
            const SizedBox(height: 12),
            InkWell(
              child: const Text(
                'https://github.com/k2-fsa/sherpa-onnx',
                style: TextStyle(color: Colors.blue, decoration: TextDecoration.underline),
              ),
              onTap: () => launchUrl(Uri.parse('https://github.com/k2-fsa/sherpa-onnx')),
            ),
            const SizedBox(height: 12),
            InkWell(
              child: const Text(
                'Doc: https://k2-fsa.github.io/sherpa/onnx/',
                style: TextStyle(color: Colors.blue, decoration: TextDecoration.underline),
              ),
              onTap: () => launchUrl(Uri.parse('https://k2-fsa.github.io/sherpa/onnx/')),
            ),
          ],
        ),
      ),
    );
  }
}
