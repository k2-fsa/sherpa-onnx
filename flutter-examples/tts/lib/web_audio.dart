// Copyright (c)  2026  Xiaomi Corporation
// Web audio playback using Web Audio API.
import 'dart:js_interop';
import 'dart:js_interop_unsafe';
import 'dart:typed_data';

import './wav_encoder.dart';
export './wav_encoder.dart' show encodeWav;

/// Download WAV bytes as a file in the browser.
void downloadWavBytes(Uint8List wavBytes, String filename) {
  globalContext['_sherpaDownloadBytes'] = wavBytes.toJS;
  globalContext['_sherpaDownloadFilename'] = filename.toJS;

  final eval = globalContext.getProperty('eval'.toJS) as JSFunction;
  eval.callAsFunction(null, '''
    (function() {
      var bytes = window._sherpaDownloadBytes;
      var name = window._sherpaDownloadFilename || 'audio.wav';
      var blob = new Blob([bytes], {type: 'audio/wav'});
      var url = URL.createObjectURL(blob);
      var a = document.createElement('a');
      a.href = url;
      a.download = name;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
      window._sherpaDownloadBytes = null;
      window._sherpaDownloadFilename = null;
    })()
  '''.toJS);
}

/// Play a chunk of audio samples using the Web Audio API for streaming playback.
/// Chunks are scheduled sequentially so they play without gaps.
/// Call [resetChunkPlayback] before starting a new generation.
void playAudioChunk(Float32List samples, int sampleRate) {
  // Create a unique ID for this chunk to avoid race conditions.
  final id = _chunkId++;
  globalContext['_sherpaChunk_$id'] = samples.toJS;

  final eval = globalContext.getProperty('eval'.toJS) as JSFunction;
  eval.callAsFunction(null, '''
    (function() {
      var id = $id;
      var sr = $sampleRate;
      // Capture data immediately in closure to avoid race with next chunk.
      var samples = window['_sherpaChunk_' + id];
      setTimeout(function() {
        if (!window._sherpaAudioCtx) {
          window._sherpaAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
          window._sherpaNextTime = 0;
        }
        var ctx = window._sherpaAudioCtx;
        if (!samples) return;
        var buf = ctx.createBuffer(1, samples.length, sr);
        buf.getChannelData(0).set(samples);
        var source = ctx.createBufferSource();
        source.buffer = buf;
        source.connect(ctx.destination);
        var startTime = Math.max(ctx.currentTime, window._sherpaNextTime);
        source.start(startTime);
        window._sherpaNextTime = startTime + buf.duration;
        delete window['_sherpaChunk_' + id];
      }, 0);
    })()
  '''.toJS);
}

int _chunkId = 0;

/// Reset the chunk playback scheduler. Call before starting a new generation.
/// Closes the old AudioContext to stop any previously scheduled chunks.
void resetChunkPlayback() {
  final eval = globalContext.getProperty('eval'.toJS) as JSFunction;
  eval.callAsFunction(null, '''
    (function() {
      if (window._sherpaAudioCtx) {
        window._sherpaAudioCtx.close();
        window._sherpaAudioCtx = null;
      }
      window._sherpaNextTime = 0;
    })()
  '''.toJS);
}

/// Play WAV bytes using the browser's Audio API.
/// Stops any previously playing audio first.
void playWavBytes(Uint8List wavBytes) {
  globalContext['_sherpaWavBytes'] = wavBytes.toJS;

  final eval = globalContext.getProperty('eval'.toJS) as JSFunction;
  eval.callAsFunction(null, '''
    (function() {
      // Stop previous audio.
      if (window._sherpaCurrentAudio) {
        window._sherpaCurrentAudio.pause();
        window._sherpaCurrentAudio.currentTime = 0;
      }
      var bytes = window._sherpaWavBytes;
      var blob = new Blob([bytes], {type: 'audio/wav'});
      var url = URL.createObjectURL(blob);
      var audio = new Audio(url);
      window._sherpaCurrentAudio = audio;
      audio.play();
      window._sherpaWavBytes = null;
    })()
  '''.toJS);
}

/// Stop all audio playback (both Audio elements and AudioContext).
void stopPlayback() {
  final eval = globalContext.getProperty('eval'.toJS) as JSFunction;
  eval.callAsFunction(null, '''
    (function() {
      // Stop Audio element playback.
      if (window._sherpaCurrentAudio) {
        window._sherpaCurrentAudio.pause();
        window._sherpaCurrentAudio.currentTime = 0;
        window._sherpaCurrentAudio = null;
      }
      // Stop AudioContext (streaming chunks).
      if (window._sherpaAudioCtx) {
        window._sherpaAudioCtx.close();
        window._sherpaAudioCtx = null;
        window._sherpaNextTime = 0;
      }
    })()
  '''.toJS);
}
