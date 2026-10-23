import Recorder from 'recorder-core'; // 注意如果未引用Recorder变量，可能编译时会被优化删除
import RecordApp from 'recorder-core/src/app-support/app';
import '@/uni_modules/Recorder-UniCore/app-uni-support.js';
import 'recorder-core/src/engine/mp3';
import 'recorder-core/src/engine/wav';
import 'recorder-core/src/engine/mp3-engine';

import 'recorder-core/src/extensions/waveview';
var chunk = null;
var pcmBuffer = new Int16Array(0);
var world = "";
var pcmindex = 0;
var socket = {};
class RecorderManager {

	constructor() {
		this.startindex = -1;
		this.isMounted = false;
		//this.socket = {};
		this.webSocketTask = null;
		this.waveView = null;
		this.clearBufferIdx = 0;
		this.pcmBuffer = new Int16Array(0);

	}
to16BitPCM(input) {
	  var dataLength = input.length * (16 / 8);
	  var dataBuffer = new ArrayBuffer(dataLength);
	  var dataView = new DataView(dataBuffer);
	  var offset = 0;
	
	  for (var i = 0; i < input.length; i++, offset += 2) {
	    var s = Math.max(-1, Math.min(1, input[i]));
	    dataView.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
	  }
	
	  return dataView;
	}
	
	 to16kHz(audioData) {
	  var sampleRate = arguments.length > 1 && arguments[1] !== undefined ? arguments[1] : 44100;
	  var data = new Float32Array(audioData);
	  var fitCount = Math.round(data.length * (16000 / sampleRate));
	  var newData = new Float32Array(fitCount);
	  var springFactor = (data.length - 1) / (fitCount - 1);
	  newData[0] = data[0];
	
	  for (var i = 1; i < fitCount - 1; i++) {
	    var tmp = i * springFactor;
	    var before = Math.floor(tmp).toFixed();
	    var after = Math.ceil(tmp).toFixed();
	    var atPoint = tmp - before;
	    newData[i] = data[before] + (data[after] - data[before]) * atPoint;
	  }
	
	  newData[fitCount - 1] = data[data.length - 1];
	  return newData;
	}

	init(url) {
		var that = this;

		this.url = url;

		socket = new WebSocket(url);
		socket.onopen = () => {
			console.log('WebSocket connection opened');
		};

		socket.onerror = error => {
			console.error('WebSocket error:', error);
		};

		socket.onclose = () => {
			this.reconnectWebSocket();
			console.log('WebSocket connection closed');
		};
		console.log(socket);

		const receiveTask = this.receiveResults(socket);

	}
	reconnectWebSocket() {
		// 重新创建 WebSocket 实例
		socket = new WebSocket(this.url);

		socket.onopen = () => {

			console.log('WebSocket reconnected');
		};
		socket.onclose = () => {
			this.reconnectWebSocket();
			console.log('WebSocket connection closed');
		};
		const receiveTask = this.receiveResults(socket);
	}

	mounted() {
		this.isMounted = true;
		RecordApp.UniPageOnShow(this);
	}

	onLoad() {

	}

	// socket() {
	// 	this.webSocketTask = uni.connectSocket({
	// 		url: this.url,
	// 		header: {
	// 			'content-type': 'application/json'
	// 		},
	// 		success(res) {
	// 			console.log('成功', res);
	// 		}
	// 	});

	// 	this.webSocketTask.onOpen((res) => {
	// 		this.webSocketTask.send({
	// 			data: JSON.stringify({
	// 				type: 'pong'
	// 			})
	// 		});
	// 		console.info("监听WebSocket连接打开事件", res);
	// 	});

	// 	uni.onSocketError((res) => {
	// 		console.info("监听WebSocket错误" + res);
	// 	});
	// }

	onShow() {

	}

	onUnload() {
		uni.closeSocket({
			success: () => {
				console.info("退出成功");
			}
		});
	}

	recReq() {
		RecordApp.UniWebViewActivate(this);
		RecordApp.RequestPermission(() => {
			console.log("已获得录音权限，可以开始录音了");
		}, (msg, isUserNotAllow) => {
			if (isUserNotAllow) {
				// 用户拒绝了录音权限
			}
			console.error("请求录音权限失败：" + msg);
		});
	}
	async receiveResults(socket) {
		let lastMessage = '';
		while (true) {
			const message = await new Promise(resolve => socket.onmessage = e => resolve(e.data));
			// const message = socket.data;
			if (message !== 'Done!') {
				if (lastMessage !== message) {
					lastMessage = message;
					if (lastMessage) {
						//console.log(lastMessage);
						this.talkStart(lastMessage);
						lastMessage = JSON.parse(lastMessage);
						//console.log(lastMessage.is_final);
						if (lastMessage.is_final) {
							this.talkEnd(lastMessage.text);
						}

					}
				}
			} else {
				return lastMessage;
			}
		}
	}
	startIdentify(res) {
		chunk = null;
		//console.log("开始识别");
	}

	talkStart(res) {
		//console.log("一句话开始");
		console.log(res);
	}
	identifyChange(res) {
		//console.log("识别变化时");
	}

	talkEnd(res) {
		console.log("一句话结束")
		console.log(res);

	}



	callback(buffers, powerLevel, duration, sampleRate, newBufferIdx, asyncEnd) {
		//console.log(powerLevel);
		// this.waveView.input(buffers[buffers.length-1],powerLevel,sampleRate);
		//console.log("数据：" + this.startindex);



		if (powerLevel >= 0) {
			chunk = Recorder.SampleData(buffers, sampleRate, 16000, chunk)
			//	chunk = Recorder.SampleData(buffers, sampleRate, 16000, chunk)
			//console.log(chunk);
			var pcm = chunk.data;
			var tmp = new Int16Array(pcmBuffer.length + pcm.length);
			tmp.set(pcmBuffer, 0);
			tmp.set(pcm, pcmBuffer.length);
			pcmBuffer = tmp;


			this.identifyChange(powerLevel);
			var int16Data = pcmBuffer;
			const float32Data = new Float32Array(int16Data.length);
			for (let i = 0; i < int16Data.length; i++) {
				float32Data[i] = int16Data[i] / 32768.0;
			}

			this.startIdentify(powerLevel);


			socket.send(float32Data);

			pcmBuffer = new Int16Array(0);
			//	chunk =[];
			if (this.clearBufferIdx > newBufferIdx) {
				this.clearBufferIdx = 0;
			}
			for (let i = this.clearBufferIdx || 0; i < newBufferIdx; i++) {
				buffers[i] = null;
			}
			this.clearBufferIdx = newBufferIdx;
		}
		//	this.chunk=RecordApp.SampleData(buffers,bufferSampleRate,pcmBufferSampleRate,chunk);
		//chunk=Recorder.SampleData(rec2.buffers,rec2.srcSampleRate,pcmBufferSampleRate,chunk); //直接使用rec2.buffers来处理也是一样的，rec2.buffers的采样率>=buffers的采样率
		//这个就是当前最新的pcm，采样率已转成16000，Int16Array可以直接发送使用，或发送pcm.buffer是ArrayBuffer


	}
	recStart() {

		const set = {
			type: "wav",
			sampleRate: 16000,
			bitRate: 16,
			// disableEnvInFix:true,
			onProcess: (buffers, powerLevel, duration, sampleRate, newBufferIdx, asyncEnd) => {
				//   console.log(powerLevel);
				this.callback(buffers, powerLevel, duration, sampleRate, newBufferIdx, asyncEnd);






			}

		};

		RecordApp.UniWebViewActivate(this);
		RecordApp.Start(set, () => {
			console.log("已开始录音");
		}, (msg) => {
			console.error("开始录音失败：" + msg);
		});
	}

	recPause() {
		if (RecordApp.GetCurrentRecOrNull()) {
			RecordApp.Pause();
			console.log("已暂停");
		}
	}

	recResume() {
		if (RecordApp.GetCurrentRecOrNull()) {
			RecordApp.Resume();
			console.log("继续录音中...");
		}
	}

	recStop() {
		chunk = null;
		RecordApp.Stop((arrayBuffer, duration, mime) => {

			//console.log(arrayBuffer, (window.URL || webkitURL).createObjectURL(arrayBuffer));
		}, (msg) => {
			console.error("结束录音失败：" + msg);
		});
	}

	setupWaveView(canvasId) {
		RecordApp.UniFindCanvas(this, [canvasId], `
            this.waveView = Recorder.WaveView({ compatibleCanvas: canvas1, width: 300, height: 100 });
        `, (canvas1) => {
			this.waveView = Recorder.WaveView({
				compatibleCanvas: canvas1,
				width: 300,
				height: 100
			});
		});
	}
}

export default RecorderManager;