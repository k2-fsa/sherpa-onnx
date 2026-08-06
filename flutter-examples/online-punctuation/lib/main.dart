// Copyright (c)  2026  Xiaomi Corporation
import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';

import './punct_screen.dart';
import './model_config.dart' show modelDir, modelUrl, modelDocUrl;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'sherpa-onnx Online Punctuation',
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
          PunctScreen(),
          InfoScreen(),
        ],
      ),
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _currentIndex,
        onTap: (i) => setState(() => _currentIndex = i),
        items: const [
          BottomNavigationBarItem(
              icon: Icon(Icons.text_fields), label: 'Punctuation'),
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
                      Icon(Icons.text_fields,
                          color: theme.colorScheme.primary),
                      const SizedBox(width: 8),
                      Text('Current Model',
                          style: theme.textTheme.titleMedium
                              ?.copyWith(fontWeight: FontWeight.bold)),
                    ],
                  ),
                  const Divider(),
                  Text(modelDir,
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
                    icon: Icons.text_fields,
                    label: 'All Punctuation Models',
                    url:
                        'https://k2-fsa.github.io/sherpa/onnx/punctuation/pretrained_models.html',
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
                        style:
                            style.copyWith(fontSize: 11, color: Colors.grey),
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
