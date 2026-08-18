// Native file save helper with file picker dialog.
import 'dart:io';

import 'package:file_picker/file_picker.dart';

/// Show a native save dialog and write the file.
/// Returns the chosen path, or null if cancelled.
Future<String?> saveFileAs(String sourcePath, String suggestedName) async {
  try {
    final bytes = File(sourcePath).readAsBytesSync();

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

    print('FilePicker result: $result');

    if (result == null) return null;

    await File(result).writeAsBytes(bytes);
    return result;
  } catch (e, st) {
    print('Error in saveFileAs: $e');
    print(st);
    rethrow;
  }
}
