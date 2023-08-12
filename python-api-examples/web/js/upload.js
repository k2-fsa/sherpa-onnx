/**
References
https://developer.mozilla.org/en-US/docs/Web/API/FileList
https://developer.mozilla.org/en-US/docs/Web/API/FileReader
https://javascript.info/arraybuffer-binary-arrays
https://developer.mozilla.org/zh-CN/docs/Web/API/WebSocket
https://developer.mozilla.org/en-US/docs/Web/API/WebSocket/send
*/

var socket;

const serverIpInput = document.getElementById('server-ip');
const serverPortInput = document.getElementById('server-port');

const connectBtn = document.getElementById('connect');
const uploadBtn = document.getElementById('file');

function initWebSocket() {
  let protocol = 'ws://';
  if (window.location.protocol == 'https:') {
    protocol = 'wss://'
  }
  let server_ip = serverIpInput.value;
  let server_port = serverPortInput.value;
  console.log('protocol: ', protocol);
  console.log('server_ip: ', server_ip);
  console.log('server_port: ', server_port);


  let uri = protocol + server_ip + ':' + server_port;
  console.log('uri', uri);
  socket = new WebSocket(uri);

  // Connection opened
  socket.addEventListener('open', function(event) {
    console.log('connected');
    uploadBtn.disabled = false;
    connectBtn.disabled = true;
    connectBtn.innerHTML = 'Connected!';
  });

  // Connection closed
  socket.addEventListener('close', function(event) {
    console.log('disconnected');
    uploadBtn.disabled = true;
    connectBtn.disabled = false;
    connectBtn.innerHTML = 'Click me to connect!';
  });

  // Listen for messages
  socket.addEventListener('message', function(event) {
    console.log('Received message: ', event.data);

    document.getElementById('results').value = event.data;
    socket.send('Done');
    console.log('Sent Done');
    socket.close();
  });
}

window.onload = (event) => {
  console.log('page is fully loaded');
  console.log('protocol', window.location.protocol);
  console.log('port', window.location.port);
  if (window.location.protocol == 'https:') {
    document.getElementById('ws-protocol').textContent = 'wss://';
  }
  serverIpInput.value = window.location.hostname;
  serverPortInput.value = window.location.port;
};

connectBtn.onclick = function() {
  initWebSocket();
};

function send_header(n) {
  const header = new ArrayBuffer(8);
  // assume the uploaded wave is 16000 Hz
  new DataView(header).setInt32(0, 16000, true /* littleEndian */);
  new DataView(header).setInt32(4, n, true /* littleEndian */);
  socket.send(new Int32Array(header, 0, 2));
}

function onFileChange() {
  var files = document.getElementById('file').files;

  if (files.length == 0) {
    console.log('No file selected');
    return;
  }

  console.log('files: ' + files);

  const file = files[0];
  console.log(file);
  console.log('file.name ' + file.name);
  console.log('file.type ' + file.type);
  console.log('file.size ' + file.size);

  let reader = new FileReader();
  reader.onload = function() {
    console.log('reading file!');
    let view = new Int16Array(reader.result);
    // we assume the input file is a wav file.
    // TODO: add some checks here.
    let int16_samples = view.subarray(22);  // header has 44 bytes == 22 shorts
    let num_samples = int16_samples.length;
    let float32_samples = new Float32Array(num_samples);
    console.log('num_samples ' + num_samples)

    for (let i = 0; i < num_samples; ++i) {
      float32_samples[i] = int16_samples[i] / 32768.
    }

    // Send 1024 audio samples per request.
    //
    // It has two purposes:
    //  (1) Simulate streaming
    //  (2) There is a limit on the number of bytes in the payload that can be
    //      sent by websocket, which is 1MB, I think. We can send a large
    //      audio file for decoding in this approach.
    let buf = float32_samples.buffer
    let n = 1024 * 4;  // send this number of bytes per request.
    console.log('buf length, ' + buf.byteLength);
    send_header(buf.byteLength);
    for (let start = 0; start < buf.byteLength; start += n) {
      socket.send(buf.slice(start, start + n));
    }
  };

  reader.readAsArrayBuffer(file);
}

const clearBtn = document.getElementById('clear');
clearBtn.onclick = function() {
  console.log('clicked');
  document.getElementById('results').value = '';
};
