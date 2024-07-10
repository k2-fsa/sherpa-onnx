// Copyright (c)  2024  Xiaomi Corporation
import 'dart:convert';
import 'dart:ffi';
import 'dart:typed_data';

import 'package:ffi/ffi.dart';

int _strLen(Pointer<Uint8> codeUnits) {
  // this function is copied from
  // https://github.com/dart-archive/ffi/blob/main/lib/src/utf8.dart#L52
  var length = 0;
  while (codeUnits[length] != 0) {
    length++;
  }
  return length;
}

// This function is modified from
// https://github.com/dart-archive/ffi/blob/main/lib/src/utf8.dart#L41
// It ignores invalid utf8 sequence
String toDartString(Pointer<Utf8> s) {
  final codeUnits = s.cast<Uint8>();
  final length = _strLen(codeUnits);
  return utf8.decode(codeUnits.asTypedList(length), allowMalformed: true);
}
