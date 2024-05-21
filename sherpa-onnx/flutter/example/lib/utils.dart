// Copyright (c)  2024  Xiaomi Corporation
import 'package:path/path.dart';
import 'package:path_provider/path_provider.dart';
import 'package:flutter/services.dart' show rootBundle;
import "dart:io";

// Copy the asset file from src to dst
Future<String> copyAssetFile({required String src, required String dst}) async {
  final Directory directory = await getApplicationDocumentsDirectory();
  final target = join(directory.path, dst);

  final data = await rootBundle.load(src);
  final List<int> bytes =
      data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
  await File(target).writeAsBytes(bytes);

  return target;
}
