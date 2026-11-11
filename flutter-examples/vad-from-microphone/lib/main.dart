// Copyright (c)  2026  Xiaomi Corporation
import 'package:flutter/material.dart';
import 'package:sherpa_onnx/sherpa_onnx.dart' as sherpa_onnx;
import 'package:url_launcher/url_launcher.dart';

import './vad_screen.dart';
import './model_config.dart' show modelFile, modelUrl, modelDocUrl, modelName;

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await sherpa_onnx.initBindingsAsync();
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'sherpa-onnx VAD from Microphone',
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
          VadScreen(),
          InfoScreen(),
        ],
      ),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _currentIndex,
        onTap: (i) => setState(() => _currentIndex = i),
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.mic), label: 'VAD'),
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
    final theme = Theme.of(context);
    final linkStyle = TextStyle(
      color: theme.colorScheme.primary,
      fontSize: 13,
    );

    final version = sherpa_onnx.getVersion();
    final gitSha1 = sherpa_onnx.getGitSha1();
    final gitDate = sherpa_onnx.getGitDate();
    final onnxruntimeVersion = sherpa_onnx.getOnnxruntimeVersion();

    return Scaffold(
      appBar: AppBar(title: const Text('Info')),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(Icons.info_outline,
                          color: theme.colorScheme.primary),
                      const SizedBox(width: 8),
                      Text('Version',
                          style: theme.textTheme.titleMedium
                              ?.copyWith(fontWeight: FontWeight.bold)),
                    ],
                  ),
                  const Divider(),
                  _InfoRow('sherpa-onnx', version),
                  _InfoRow('Git SHA1', gitSha1),
                  _InfoRow('Git date', gitDate),
                  _InfoRow('onnxruntime', onnxruntimeVersion),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(Icons.mic, color: theme.colorScheme.primary),
                      const SizedBox(width: 8),
                      Text('Current Model',
                          style: theme.textTheme.titleMedium
                              ?.copyWith(fontWeight: FontWeight.bold)),
                    ],
                  ),
                  const Divider(),
                  Text(modelFile,
                      style: theme.textTheme.bodyLarge
                          ?.copyWith(fontWeight: FontWeight.w600)),
                  const SizedBox(height: 8),
                  _LinkRow(
                    icon: Icons.download,
                    label: 'Download',
                    url: modelUrl,
                    style: linkStyle,
                  ),
                  _LinkRow(
                    icon: Icons.menu_book,
                    label: 'Documentation',
                    url: modelDocUrl,
                    style: linkStyle,
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(Icons.link, color: theme.colorScheme.primary),
                      const SizedBox(width: 8),
                      Text('Resources',
                          style: theme.textTheme.titleMedium
                              ?.copyWith(fontWeight: FontWeight.bold)),
                    ],
                  ),
                  const Divider(),
                  _LinkRow(
                    icon: Icons.code,
                    label: 'GitHub',
                    url: 'https://github.com/k2-fsa/sherpa-onnx',
                    style: linkStyle,
                  ),
                  _LinkRow(
                    icon: Icons.menu_book,
                    label: 'Documentation',
                    url: 'https://k2-fsa.github.io/sherpa/onnx/',
                    style: linkStyle,
                  ),
                  _LinkRow(
                    icon: Icons.cloud_download,
                    label: 'Model Releases',
                    url: 'https://github.com/k2-fsa/sherpa-onnx/releases',
                    style: linkStyle,
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Icon(Icons.group, color: theme.colorScheme.primary),
                      const SizedBox(width: 8),
                      Text('Community',
                          style: theme.textTheme.titleMedium
                              ?.copyWith(fontWeight: FontWeight.bold)),
                    ],
                  ),
                  const Divider(),
                  _InfoRow('QQ Group', '744602236'),
                  _LinkRow(
                    icon: Icons.chat,
                    label: 'WeChat Groups',
                    url: 'https://k2-fsa.github.io/sherpa/social-groups.html',
                    style: linkStyle,
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 16),
          Center(
            child: GestureDetector(
              onTap: () =>
                  launchUrl(Uri.parse('https://github.com/k2-fsa/sherpa-onnx')),
              child: Text.rich(
                TextSpan(
                  text: 'Powered by ',
                  style: theme.textTheme.bodySmall
                      ?.copyWith(color: theme.colorScheme.outline),
                  children: [
                    TextSpan(
                      text: 'sherpa-onnx',
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: theme.colorScheme.primary,
                        decoration: TextDecoration.underline,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _InfoRow extends StatelessWidget {
  final String label;
  final String value;

  const _InfoRow(this.label, this.value);

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 2),
      child: Row(
        children: [
          SizedBox(
            width: 110,
            child: Text(label,
                style: const TextStyle(fontSize: 13, color: Colors.grey)),
          ),
          Expanded(
            child: Text(value, style: const TextStyle(fontSize: 13)),
          ),
        ],
      ),
    );
  }
}

class _LinkRow extends StatelessWidget {
  final IconData icon;
  final String label;
  final String url;
  final TextStyle style;

  const _LinkRow({
    required this.icon,
    required this.label,
    required this.url,
    required this.style,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: InkWell(
        onTap: () => launchUrl(Uri.parse(url)),
        borderRadius: BorderRadius.circular(8),
        child: Padding(
          padding: const EdgeInsets.symmetric(vertical: 6, horizontal: 4),
          child: Row(
            children: [
              Icon(icon, size: 18, color: style.color),
              const SizedBox(width: 10),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(label, style: style.copyWith(fontSize: 14)),
                    Text(url,
                        style: style.copyWith(fontSize: 11, color: Colors.grey),
                        overflow: TextOverflow.ellipsis),
                  ],
                ),
              ),
              Icon(Icons.open_in_new, size: 14, color: style.color),
            ],
          ),
        ),
      ),
    );
  }
}
