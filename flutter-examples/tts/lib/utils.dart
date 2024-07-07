// Copyright (c)  2024  Xiaomi Corporation
import 'dart:io';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';

Future<String> generateWaveFilename([String suffix = '']) async {
  final Directory directory = await getApplicationDocumentsDirectory();
  DateTime now = DateTime.now();
  final filename =
      '${now.year.toString()}-${now.month.toString().padLeft(2, '0')}-${now.day.toString().padLeft(2, '0')}-${now.hour.toString().padLeft(2, '0')}-${now.minute.toString().padLeft(2, '0')}-${now.second.toString().padLeft(2, '0')}$suffix.wav';

  return p.join(directory.path, filename);
}

// https://stackoverflow.com/questions/68862225/flutter-how-to-get-all-files-from-assets-folder-in-one-list
Future<List<String>> getAllAssetFiles() async {
  final AssetManifest assetManifest =
      await AssetManifest.loadFromAssetBundle(rootBundle);
  final List<String> assets = assetManifest.listAssets();
  return assets;
}

String stripLeadingDirectory(String src, {int n = 1}) {
  return p.joinAll(p.split(src).sublist(n));
}

Future<void> copyAllAssetFiles() async {
  final allFiles = await getAllAssetFiles();
  for (final src in allFiles) {
    final dst = stripLeadingDirectory(src);
    await copyAssetFile(src, dst);
  }
}

// Copy the asset file from src to dst.
// If dst already exists, then just skip the copy
Future<String> copyAssetFile(String src, [String? dst]) async {
  final Directory directory = await getApplicationDocumentsDirectory();
  if (dst == null) {
    dst = p.basename(src);
  }
  final target = p.join(directory.path, dst);
  bool exists = await new File(target).exists();

  if (!exists) {
    final data = await rootBundle.load(src);
    final List<int> bytes =
        data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
    await (await File(target).create(recursive: true)).writeAsBytes(bytes);
  }

  return target;
}
