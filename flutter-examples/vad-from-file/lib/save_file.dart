// Native file save helper with file picker dialog.
import 'dart:io';
import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';

/// Show a native save dialog and write the file.
/// Returns the chosen path, or null if cancelled.
Future<String?> saveFileAs(String sourcePath, String suggestedName) async {
  try {
    final bytes = File(sourcePath).readAsBytesSync();

    final result = await FilePicker.platform.saveFile(
      dialogTitle: 'Save audio file',
      fileName: suggestedName,
      type: FileType.custom,
      allowedExtensions: ['wav'],
    );

    if (result == null) return null;

    await File(result).writeAsBytes(bytes);
    return result;
  } catch (e, st) {
    print('Error in saveFileAs: $e');
    print(st);
    rethrow;
  }
}

/// Save WAV bytes directly via a save dialog.
/// Returns the chosen path, or null if cancelled.
Future<String?> saveWavBytes(Uint8List wavBytes, String suggestedName) async {
  try {
    final result = await FilePicker.platform.saveFile(
      dialogTitle: 'Save audio file',
      fileName: suggestedName,
      type: FileType.custom,
      allowedExtensions: ['wav'],
    );

    if (result == null) return null;

    await File(result).writeAsBytes(wavBytes);
    return result;
  } catch (e, st) {
    print('Error in saveWavBytes: $e');
    print(st);
    rethrow;
  }
}
