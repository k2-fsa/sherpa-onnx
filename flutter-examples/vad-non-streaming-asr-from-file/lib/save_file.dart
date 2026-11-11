// Native file save helper with file picker dialog.
import 'dart:io';
import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';

/// Show a native save dialog and write the file.
/// Returns the chosen path, or null if cancelled.
Future<String?> saveFileAs(String sourcePath, String suggestedName) async {
  try {
    final bytes = File(sourcePath).readAsBytesSync();
    return await _saveBytes(bytes, suggestedName);
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
    return await _saveBytes(wavBytes, suggestedName);
  } catch (e, st) {
    print('Error in saveWavBytes: $e');
    print(st);
    rethrow;
  }
}

/// Save text content via a save dialog.
/// Returns the chosen path, or null if cancelled.
Future<String?> saveTextContent(
    String content, String suggestedName, String ext) async {
  try {
    if (Platform.isAndroid) {
      final dir = await FilePicker.platform.getDirectoryPath(
        dialogTitle: 'Select folder to save file',
      );
      if (dir == null) return null;
      final file = File('$dir/$suggestedName');
      await file.writeAsString(content);
      return file.path;
    }

    final result = await FilePicker.platform.saveFile(
      dialogTitle: 'Save file',
      fileName: suggestedName,
      type: FileType.custom,
      allowedExtensions: [ext],
    );

    if (result == null) return null;

    await File(result).writeAsString(content);
    return result;
  } catch (e) {
    print('Error saving text: $e');
    rethrow;
  }
}

Future<String?> _saveBytes(Uint8List bytes, String suggestedName) async {
  if (Platform.isAndroid) {
    // On Android, FilePicker.saveFile() doesn't reliably show a dialog.
    // Use getDirectoryPath() to let the user pick a directory, then save.
    final dir = await FilePicker.platform.getDirectoryPath(
      dialogTitle: 'Select folder to save audio',
    );
    if (dir == null) return null;
    final file = File('$dir/$suggestedName');
    await file.writeAsBytes(bytes);
    return file.path;
  }

  // Desktop (macOS, Linux, Windows): use native save dialog.
  final result = await FilePicker.platform.saveFile(
    dialogTitle: 'Save audio file',
    fileName: suggestedName,
    type: FileType.custom,
    allowedExtensions: ['wav'],
  );

  if (result == null) return null;

  await File(result).writeAsBytes(bytes);
  return result;
}
