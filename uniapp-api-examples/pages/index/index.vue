<template>
	<view class="content">
		<image class="logo" src="/static/logo3.png"></image>
		<view style="font-size: 18px; " class="">曲流觞</view>
	 <view class="text-area">

	<view style="width: 60vw; display: flex; justify-content: space-around; align-items: center; height: 140rpx;" >
		<button style="background-color: #007aff;color: #fff;" @click="startMp">开始识别</button>
		<button style="background-color: #ff0000; color: #fff;" @click="stop">停止识别</button>
	</view>
			
		</view>
			<view style="width: 100%; height: 200rpx;display: flex; justify-content: center; margin: 30rpx auto; flex-direction: column; align-items: center;">
				<text>实时识别内容：</text>
				<textarea style="border: 1px #bebebe solid; " :value="word"  cols="60" rows="20">
					
				</textarea>
			</view>
	</view>
</template>

<script>
	import RecorderManager from '../../common/RecorderManager.js';
	export default {
		data() {
			return {
				recorderManager: null,
				url: '你的服务器地址 ',	//你的服务器地址 怎么编译请看这个文档 https://www.bilibili.com/read/cv22438156/
				word: '',
			}
		},
		onShow() {

			this.recorderManager.onShow();
			this.recorderManager.recReq();
		},
		onLoad() {
			this.recorderManager = new RecorderManager();
			this.recorderManager.init(this.url);
			this.recorderManager.mounted();
			this.recorderManager.onLoad();


		},
		methods: {
			stop(){
					if (this.recorder) this.recorder.recStop();
			},
			startMp() {
				this.recorderManager.recStart();
				var t = this;
				t.recorder = this.recorderManager;

				t.recorder.startIdentify = (res) => {
					console.log('开始识别', res);
				
				};
				// 一句话开始
				t.recorder.talkStart = (res) => {
					console.log('一句话开始', res);
					var data=JSON.parse(res);
				
					if(data.text.trim()!=""){
						this.word=data.text;
					}
				};
				// 识别变化时
				t.recorder.identifyChange = (res) => {
					if (res > 3) {
						console.log('识别变化时', res);
					}
				};
				// 一句话结束
				t.recorder.talkEnd = (res) => {
					console.log('一句话结束', res);
					t.word = res;
					//语音转文字内容
					console.log(t.word);
				};
				// 识别结束
				t.recorder.OnRecognitionComplete = (res) => {
					console.log('识别结束', res);
				};
				// 识别错误
				t.recorder.OnError = (res) => {
					console.log('识别失败', res);
				};
			},
		}
	}
</script>

<style>
	.content {
		display: flex;
		flex-direction: column;
		align-items: center;
		justify-content: center;
	}

	.logo {
		height: 200rpx;
		width: 200rpx;
		margin-top: 200rpx;
		margin-left: auto;
		margin-right: auto;
		/* margin-bottom: 50rpx; */
	}

	.text-area {
		display: flex;
		justify-content: center;
	}

	.title {
		font-size: 36rpx;
		color: #8f8f94;
	}
</style>