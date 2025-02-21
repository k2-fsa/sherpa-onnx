// Copyright (c) 2023 Xiaomi Corporation (authors: Fangjun Kuang)
const mic = require('mic'); // It uses `mic` for better compatibility, do check its [npm](https://www.npmjs.com/package/mic) before running it.
const sherpa_onnx = require('sherpa-onnx');

function createOnlineRecognizer() {
  let onlineParaformerModelConfig = {
    encoder: './sherpa-onnx-streaming-paraformer-bilingual-zh-en/encoder.int8.onnx',
    decoder: './sherpa-onnx-streaming-paraformer-bilingual-zh-en/decoder.int8.onnx',
  };

  let onlineModelConfig = {
    paraformer: onlineParaformerModelConfig,
    tokens: './sherpa-onnx-streaming-paraformer-bilingual-zh-en/tokens.txt',
  };

  let recognizerConfig = {
    modelConfig: onlineModelConfig,
    enableEndpoint: 1,
    rule1MinTrailingSilence: 2.4,
    rule2MinTrailingSilence: 1.2,
    rule3MinUtteranceLength: 20,
  };

  return sherpa_onnx.createOnlineRecognizer(recognizerConfig);
}

/**
 * SpeechSession class, work as a session manager with the formatOutput function
 * Sample output:
=== Automated Speech Recognition ===
Current Session #1
Time: 8:44:46 PM
------------------------
Recognized Sentences:
[8:44:43 PM] 1. it's so great three result is great great 她还支持中文
[8:44:46 PM] 2. 很厉
------------------------
Recognizing: 真的很厉害太厉害

*/
class SpeechSession {
  constructor() {
    this.startTime = Date.now();
    this.sentences = [];
    this.currentText = '';
    this.lastUpdateTime = Date.now();
  }

  addOrUpdateText(text) {
    this.currentText = text;
    this.lastUpdateTime = Date.now();
  }

  finalizeSentence() {
    if (this.currentText.trim()) {
      this.sentences.push({
        text: this.currentText.trim(),
        timestamp: new Date().toLocaleTimeString()
      });
    }
    this.currentText = '';
  }

  shouldStartNewSession() {
    return Date.now() - this.lastUpdateTime > 10000; // 10 seconds of silence
  }
}

function formatOutput() {
    clearConsole();
    console.log('\n=== Automated Speech Recognition ===');
    console.log(`Current Session #${sessionCount}`);
    console.log('Time:', new Date().toLocaleTimeString());
    console.log('------------------------');
    
    // 显示历史句子
    if (currentSession.sentences.length > 0) {
      console.log('Recognized Sentences:');
      currentSession.sentences.forEach((sentence, index) => {
        console.log(`[${sentence.timestamp}] ${index + 1}. ${sentence.text}`);
      });
      console.log('------------------------');
    }
    
    // 显示当前正在识别的内容
    if (currentSession.currentText) {
      console.log('Recognizing:', currentSession.currentText);
    }
  }
  

const recognizer = createOnlineRecognizer();
const stream = recognizer.createStream();
let currentSession = new SpeechSession();
let sessionCount = 1;

function clearConsole() {
  process.stdout.write('\x1B[2J\x1B[0f');
}


function exitHandler(options, exitCode) {
  if (options.cleanup) {
    console.log('\nCleaned up resources...');
    micInstance.stop();
    stream.free();
    recognizer.free();
  }
  if (exitCode || exitCode === 0) console.log('Exit code:', exitCode);
  if (options.exit) process.exit();
}

const micInstance = mic({
  rate: recognizer.config.featConfig.sampleRate,
  channels: 1,
  debug: false, // 关闭调试输出
  device: 'default',
  bitwidth: 16,
  encoding: 'signed-integer',
  exitOnSilence: 6,
  fileType: 'raw'
});

const micInputStream = micInstance.getAudioStream();

function startMic() {
  return new Promise((resolve, reject) => {
    micInputStream.once('startComplete', () => {
      console.log('Mic phone started.');
      resolve();
    });
    
    micInputStream.once('error', (err) => {
      console.error('Mic phone start error:', err);
      reject(err);
    });
    
    micInstance.start();
  });
}

micInputStream.on('data', buffer => {
  const int16Array = new Int16Array(buffer.buffer);
  const samples = new Float32Array(int16Array.length);
  
  for (let i = 0; i < int16Array.length; i++) {
    samples[i] = int16Array[i] / 32768.0;
  }

  stream.acceptWaveform(recognizer.config.featConfig.sampleRate, samples);

  while (recognizer.isReady(stream)) {
    recognizer.decode(stream);
  }

  const isEndpoint = recognizer.isEndpoint(stream);
  const text = recognizer.getResult(stream).text;

  if (text.length > 0) {
    // 检查是否需要开始新会话
    if (currentSession.shouldStartNewSession()) {
      currentSession.finalizeSentence();
      sessionCount++;
      currentSession = new SpeechSession();
    }
    
    currentSession.addOrUpdateText(text);
    formatOutput();
  }

  if (isEndpoint) {
    if (text.length > 0) {
      currentSession.finalizeSentence();
      formatOutput();
    }
    recognizer.reset(stream);
  }
});

micInputStream.on('error', err => {
  console.error('Audio stream error:', err);
});

micInputStream.on('close', () => {
  console.log('Mic phone closed.');
});

process.on('exit', exitHandler.bind(null, {cleanup: true}));
process.on('SIGINT', exitHandler.bind(null, {exit: true}));
process.on('SIGUSR1', exitHandler.bind(null, {exit: true}));
process.on('SIGUSR2', exitHandler.bind(null, {exit: true}));
process.on('uncaughtException', exitHandler.bind(null, {exit: true}));

async function main() {
  try {
    console.log('Starting ...');
    await startMic();
    console.log('Initialized, waiting for speech ...');
    formatOutput();
  } catch (err) {
    console.error('Failed to initialize:', err);
    process.exit(1);
  }
}

main();