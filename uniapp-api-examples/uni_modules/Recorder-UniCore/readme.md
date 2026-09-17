*⠀*

*⠀*

# Recorder-UniCore组件：uni-app内使用RecordApp录音

本组件使用`Recorder`开源库来进行录音和音频数据处理，使用`RecordApp`和本组件内的`app-uni-support.js`来适配到不同平台环境下进行录音。

- 支持vue2、vue3、nvue
- 支持编译成：H5、Android App、iOS App、微信小程序
- 支持已有的大部分录音格式：mp3、wav、pcm、amr、ogg、g711a、g711u等
- 支持实时处理，包括变速变调、实时上传、ASR语音转文字
- 支持可视化波形显示；可配置回声消除、降噪；**注意：不支持通话时录音**
- 支持离线使用，本组件和配套原生插件均不依赖网络
- App端有配套的[原生录音插件](https://ext.dcloud.net.cn/plugin?name=Recorder-NativePlugin)可供搭配使用，兼容性和体验更好

**详细文档(含Demo项目)：** [https://github.com/xiangyuecn/Recorder/tree/master/app-support-sample/demo_UniApp](https://github.com/xiangyuecn/Recorder/tree/master/app-support-sample/demo_UniApp)

**Recorder开源库地址：** [https://github.com/xiangyuecn/Recorder](https://github.com/xiangyuecn/Recorder)

如果github打不开，可以[点此访问Gitee仓库地址](https://gitee.com/xiangyuecn/Recorder/tree/master/app-support-sample/demo_UniApp) 。


*⠀*

## 测试方法
**示例项目如果在HBuilder中编译失败，请删掉node_modules目录重新手动执行npm install（偶尔出现HBuilder自动创建项目依赖包不完整，导致无法编译）**

1. 在本插件市场页面右侧下载或导入示例项目（或打开上面详细文档链接中的Demo源码）
2. 在测试项目根目录执行 `npm install --registry=https://registry.npmmirror.com/` ，完成`recorder-core`依赖的安装
3. 在HBuilder中打开本测试项目文件夹
4. 在HBuilder中运行到浏览器、手机、微信小程序，即可在不同环境下测试
5. 测试中提供了：基础录音、播放、上传、WebSocket实时语音通话对讲、ASR语音识别等功能



*⠀*

*⠀*

# 集成到自己项目中

你可以直接参考上面的测试示例项目源码，里面的`main_recTest.vue`更容易入门；示例项目中已经实现了很多功能，简单使用可直接照抄Demo代码到你的项目中。


## 一、引入js文件
1. 在你的项目根目录安装`recorder-core`：`npm install recorder-core --registry=https://registry.npmmirror.com/`
2. 导入Recorder-UniCore组件：插件市场下载本组件，然后添加到你的项目中 `/uni_modules/Recorder-UniCore`
3. 在需要录音的vue文件内编写以下代码，按需引入需要的js

``` html
<script> /**这里是逻辑层**/
//必须引入的Recorder核心（文件路径是 /src/recorder-core.js 下同），使用import、require都行
import Recorder from 'recorder-core' //注意如果未引用Recorder变量，可能编译时会被优化删除（如vue3 tree-shaking），请改成 import 'recorder-core'，或随便调用一下 Recorder.a=1 保证强引用

//必须引入的RecordApp核心文件（文件路径是 /src/app-support/app.js）
import RecordApp from 'recorder-core/src/app-support/app'

//所有平台必须引入的uni-app支持文件（如果编译出现路径错误，请把@换成 ../../ 这种）
import '@/uni_modules/Recorder-UniCore/app-uni-support.js'

/** 需要编译成微信小程序时，引入微信小程序支持文件 **/
// #ifdef MP-WEIXIN
    import 'recorder-core/src/app-support/app-miniProgram-wx-support.js'
// #endif


/** H5、小程序环境中：引入需要的格式编码器、可视化插件，App环境中在renderjs中引入 **/
// 注意：如果App中需要在逻辑层中调用Recorder的编码/转码功能，需要去掉此条件编译，否则会报未加载编码器的错误
// #ifdef H5 || MP-WEIXIN
    //按需引入你需要的录音格式支持文件，如果需要多个格式支持，把这些格式的编码引擎js文件统统引入进来即可
    import 'recorder-core/src/engine/mp3'
    import 'recorder-core/src/engine/mp3-engine' //如果此格式有额外的编码引擎（*-engine.js）的话，必须要加上
    
    //可选的插件支持项，把需要的插件按需引入进来即可
    import 'recorder-core/src/extensions/waveview'
// #endif

// ... 这后面写页面代码，用选项式API风格（vue2、vue3）、setup组合式API风格（仅vue3）都可以
</script>
```

``` html
<!-- #ifdef APP -->
<script module="yourModuleName" lang="renderjs"> //此模块内部只能用选项式API风格，vue2、vue3均可用，请照抄这段代码；不可改成setup组合式API风格，否则可能不能import vue导致编译失败
/**需要编译成App时，你需要添加一个renderjs模块，然后一模一样的import上面那些js（微信的js除外）
    ，因为App中默认是在renderjs（WebView）中进行录音和音频编码**/
import 'recorder-core'
import RecordApp from 'recorder-core/src/app-support/app'
import '../../uni_modules/Recorder-UniCore/app-uni-support.js' //renderjs中似乎不支持"@/"打头的路径，如果编译路径错误请改正路径即可

//按需引入你需要的录音格式支持文件，和插件
import 'recorder-core/src/engine/mp3'
import 'recorder-core/src/engine/mp3-engine' 

import 'recorder-core/src/extensions/waveview'

export default {
    mounted(){
        //App的renderjs必须调用的函数，传入当前模块this
        RecordApp.UniRenderjsRegister(this);
    },
    methods: {
        //这里定义的方法，在逻辑层中可通过 RecordApp.UniWebViewVueCall(this,'this.xxxFunc()') 直接调用
        //调用逻辑层的方法，请直接用 this.$ownerInstance.callMethod("xxxFunc",{args}) 调用，二进制数据需转成base64来传递
    }
}
</script>
<!-- #endif -->
```


*⠀*

*⠀*

## 二、调用录音
``` javascript
/**在逻辑层中编写**/
//import ... 上面那些import代码

//var vue3This=getCurrentInstance().proxy; //当用vue3 setup组合式 API (Composition API) 编写时，直接在import后面取到当前实例this，在需要this的地方传vue3This变量即可，其他的和选项式 API (Options API) 没有任何区别；import {getCurrentInstance} from 'vue'；详细可以参考Demo项目中的 page_vue3____composition_api.vue

//RecordApp.UniNativeUtsPlugin={ nativePlugin:true };  //App中启用配套的原生录音插件支持，配置后会使用原生插件进行录音，没有原生插件时依旧使用renderjs H5录音
//App中提升后台录音的稳定性：配置了原生插件后，可配置 `RecordApp.UniWithoutAppRenderjs=true` 禁用renderjs层音频编码（WebWorker加速），变成逻辑层中直接编码（但会降低逻辑层性能），后台运行时可避免部分手机WebView运行受限的影响
//App中提升后台录音的稳定性：需要启用后台录音保活服务（iOS不需要），Android 9开始，锁屏或进入后台一段时间后App可能会被禁止访问麦克风导致录音静音、无法录音（renderjs中H5录音也受影响），请调用配套原生插件的`androidNotifyService`接口，或使用第三方保活插件

export default {
data() { return {} } //视图没有引用到的变量无需放data里，直接this.xxx使用

,mounted() {
    this.isMounted=true;
    //页面onShow时【必须调用】的函数，传入当前组件this
    RecordApp.UniPageOnShow(this);
}
,onShow(){ //onShow可能比mounted先执行，页面可能还未准备好
    if(this.isMounted) RecordApp.UniPageOnShow(this);
}

,methods:{
    //请求录音权限
    recReq(){
        //编译成App时提供的授权许可（编译成H5、小程序为免费授权可不填写）；如果未填写授权许可，将会在App打开后第一次调用请求录音权限时，弹出“未获得商用授权时，App上仅供测试”提示框
        //RecordApp.UniAppUseLicense='我已获得UniAppID=*****的商用授权';
        
        //RecordApp.RequestPermission_H5OpenSet={ audioTrackSet:{ noiseSuppression:true,echoCancellation:true,autoGainControl:true } }; //这个是Start中的audioTrackSet配置，在h5（H5、App+renderjs）中必须提前配置，因为h5中RequestPermission会直接打开录音
        
        RecordApp.UniWebViewActivate(this); //App环境下必须先切换成当前页面WebView
        RecordApp.RequestPermission(()=>{
            console.log("已获得录音权限，可以开始录音了");
        },(msg,isUserNotAllow)=>{
            if(isUserNotAllow){//用户拒绝了录音权限
                //这里你应当编写代码进行引导用户给录音权限，不同平台分别进行编写
            }
            console.error("请求录音权限失败："+msg);
        });
    }
    
    //开始录音
    ,recStart(){
        //Android App如果要后台录音，需要启用后台录音保活服务（iOS不需要），需使用配套原生插件、或使用第三方保活插件
        //RecordApp.UniNativeUtsPluginCallAsync("androidNotifyService",{ title:"正在录音" ,content:"正在录音中，请勿关闭App运行" }).then(()=>{...}).catch((e)=>{...})
        
        //录音配置信息
        var set={
            type:"mp3",sampleRate:16000,bitRate:16 //mp3格式，指定采样率hz、比特率kbps，其他参数使用默认配置；注意：是数字的参数必须提供数字，不要用字符串；需要使用的type类型，需提前把格式支持文件加载进来，比如使用wav格式需要提前加载wav.js编码引擎
            /*,audioTrackSet:{ //可选，如果需要同时播放声音（比如语音通话），需要打开回声消除（打开后声音可能会从听筒播放，部分环境下（如小程序、App原生插件）可调用接口切换成扬声器外放）
                //注意：H5、App+renderjs中需要在请求录音权限前进行相同配置RecordApp.RequestPermission_H5OpenSet后此配置才会生效
                echoCancellation:true,noiseSuppression:true,autoGainControl:true} */
            ,onProcess:(buffers,powerLevel,duration,sampleRate,newBufferIdx,asyncEnd)=>{
                //全平台通用：可实时上传（发送）数据，配合Recorder.SampleData方法，将buffers中的新数据连续的转换成pcm上传，或使用mock方法将新数据连续的转码成其他格式上传，可以参考Recorder文档里面的：Demo片段列表 -> 实时转码并上传-通用版；基于本功能可以做到：实时转发数据、实时保存数据、实时语音识别（ASR）等
                
                //注意：App里面是在renderjs中进行实际的音频格式编码操作，此处的buffers数据是renderjs实时转发过来的，修改此处的buffers数据不会改变renderjs中buffers，所以不会改变生成的音频文件，可在onProcess_renderjs中进行修改操作就没有此问题了；如需清理buffers内存，此处和onProcess_renderjs中均需要进行清理，H5、小程序中无此限制
                //注意：如果你要用只支持在浏览器中使用的Recorder扩展插件，App里面请在renderjs中引入此扩展插件，然后在onProcess_renderjs中调用这个插件；H5可直接在这里进行调用，小程序不支持这类插件；如果调用插件的逻辑比较复杂，建议封装成js文件，这样逻辑层、renderjs中直接import，不需要重复编写
                
                //H5、小程序等可视化图形绘制，直接运行在逻辑层；App里面需要在onProcess_renderjs中进行这些操作
                // #ifdef H5 || MP-WEIXIN
                if(this.waveView) this.waveView.input(buffers[buffers.length-1],powerLevel,sampleRate);
                // #endif
                
                /*实时释放清理内存，用于支持长时间录音；在指定了有效的type时，编码器内部可能还会有其他缓冲，必须同时提供takeoffEncodeChunk才能清理内存，否则type需要提供unknown格式来阻止编码器内部缓冲，App的onProcess_renderjs中需要进行相同操作
                if(this.clearBufferIdx>newBufferIdx){ this.clearBufferIdx=0 } //重新录音了就重置
                for(var i=this.clearBufferIdx||0;i<newBufferIdx;i++) buffers[i]=null;
                this.clearBufferIdx=newBufferIdx; */
            }
            ,onProcess_renderjs:`function(buffers,powerLevel,duration,sampleRate,newBufferIdx,asyncEnd){
                //App中在这里修改buffers会改变生成的音频文件，但注意：buffers会先转发到逻辑层onProcess后才会调用本方法，因此在逻辑层的onProcess中需要重新修改一遍
                //本方法可以返回true，renderjs中的onProcess将开启异步模式，处理完后调用asyncEnd结束异步，注意：这里异步修改的buffers一样的不会在逻辑层的onProcess中生效
                //App中是在renderjs中进行的可视化图形绘制，因此需要写在这里，this是renderjs模块的this（也可以用This变量）；如果代码比较复杂，请直接在renderjs的methods里面放个方法xxxFunc，这里直接使用this.xxxFunc(args)进行调用
                if(this.waveView) this.waveView.input(buffers[buffers.length-1],powerLevel,sampleRate);
                
                /*和onProcess中一样进行释放清理内存，用于支持长时间录音
                if(this.clearBufferIdx>newBufferIdx){ this.clearBufferIdx=0 } //重新录音了就重置
                for(var i=this.clearBufferIdx||0;i<newBufferIdx;i++) buffers[i]=null;
                this.clearBufferIdx=newBufferIdx; */
            }`
            ,onProcessBefore_renderjs:`function(buffers,powerLevel,duration,sampleRate,newBufferIdx){
                //App中本方法会在逻辑层onProcess之前调用，因此修改的buffers会转发给逻辑层onProcess，本方法没有asyncEnd参数不支持异步处理
                //一般无需提供本方法只用onProcess_renderjs就行，renderjs的onProcess内部调用过程：onProcessBefore_renderjs -> 转发给逻辑层onProcess -> onProcess_renderjs
            }`

            ,takeoffEncodeChunk:true?null:(chunkBytes)=>{
                //全平台通用：实时接收到编码器编码出来的音频片段数据，chunkBytes是Uint8Array二进制数据，可以实时上传（发送）出去
                //App中如果未配置RecordApp.UniWithoutAppRenderjs时，建议提供此回调，因为录音结束后会将整个录音文件从renderjs传回逻辑层，由于uni-app的逻辑层和renderjs层数据交互性能实在太拉跨了，大点的文件传输会比较慢，提供此回调后可避免Stop时产生超大数据回传
                
                //App中使用原生插件时，可方便的将数据实时保存到同一文件，第一帧时append:false新建文件，后面的append:true追加到文件
                //RecordApp.UniNativeUtsPluginCallAsync("writeFile",{path:"xxx.mp3",append:回调次数!=1, dataBase64:RecordApp.UniBtoa(chunkBytes.buffer)}).then(...).catch(...)
            }
            ,takeoffEncodeChunk_renderjs:true?null:`function(chunkBytes){
                //App中这里可以做一些仅在renderjs中才生效的事情，不提供也行，this是renderjs模块的this（也可以用This变量）
            }`
            
            ,start_renderjs:`function(){
                //App中可以放一个函数，在Start成功时renderjs中会先调用这里的代码，this是renderjs模块的this（也可以用This变量）
                //放一些仅在renderjs中才生效的事情，比如初始化，不提供也行
            }`
            ,stop_renderjs:`function(arrayBuffer,duration,mime){
                //App中可以放一个函数，在Stop成功时renderjs中会先调用这里的代码，this是renderjs模块的this（也可以用This变量）
                //放一些仅在renderjs中才生效的事情，不提供也行
            }`
        };
        
        RecordApp.UniWebViewActivate(this); //App环境下必须先切换成当前页面WebView
        RecordApp.Start(set,()=>{
            console.log("已开始录音");
            //【稳如老狗WDT】可选的，监控是否在正常录音有onProcess回调，如果长时间没有回调就代表录音不正常
            //var wdt=this.watchDogTimer=setInterval ... 请参考示例Demo的main_recTest.vue中的watchDogTimer实现
            
            //创建音频可视化图形绘制，App环境下是在renderjs中绘制，H5、小程序等是在逻辑层中绘制，因此需要提供两段相同的代码
            //view里面放一个canvas，canvas需要指定宽高（下面style里指定了300*100）
            //<canvas type="2d" class="recwave-WaveView" style="width:300px;height:100px"></canvas>
            RecordApp.UniFindCanvas(this,[".recwave-WaveView"],`
                this.waveView=Recorder.WaveView({compatibleCanvas:canvas1, width:300, height:100});
            `,(canvas1)=>{
                this.waveView=Recorder.WaveView({compatibleCanvas:canvas1, width:300, height:100});
            });
        },(msg)=>{
            console.error("开始录音失败："+msg);
        });
    }
    
    //暂停录音
    ,recPause(){
        if(RecordApp.GetCurrentRecOrNull()){
            RecordApp.Pause();
            console.log("已暂停");
        }
    }
    //继续录音
    ,recResume(){
        if(RecordApp.GetCurrentRecOrNull()){
            RecordApp.Resume();
            console.log("继续录音中...");
        }
    }
    
    //停止录音
    ,recStop(){
        //RecordApp.UniNativeUtsPluginCallAsync("androidNotifyService",{ close:true }) //关闭Android App后台录音保活服务
        
        RecordApp.Stop((arrayBuffer,duration,mime)=>{
            //全平台通用：arrayBuffer是音频文件二进制数据，可以保存成文件或者发送给服务器
            //App中如果在Start参数中提供了stop_renderjs，renderjs中的函数会比这个函数先执行
            
            //注意：当Start时提供了takeoffEncodeChunk后，你需要自行实时保存录音文件数据，因此Stop时返回的arrayBuffer的长度将为0字节
            
            //如果是H5环境，也可以直接构造成Blob/File文件对象，和Recorder使用一致
            // #ifdef H5
                var blob=new Blob([arrayBuffer],{type:mime});
                console.log(blob, (window.URL||webkitURL).createObjectURL(blob));
                var file=new File([arrayBuffer],"recorder.mp3");
                //uni.uploadFile({file:file, ...}) //参考demo中的test_upload_saveFile.vue
            // #endif
            
            //如果是App、小程序环境，可以直接保存到本地文件，然后调用相关网络接口上传
            // #ifdef APP || MP-WEIXIN
                RecordApp.UniSaveLocalFile("recorder.mp3",arrayBuffer,(savePath)=>{
                    console.log(savePath); //app保存的文件夹为`plus.io.PUBLIC_DOWNLOADS`，小程序为 `wx.env.USER_DATA_PATH` 路径
                    //uni.uploadFile({filePath:savePath, ...}) //参考demo中的test_upload_saveFile.vue
                },(errMsg)=>{ console.error(errMsg) });
            // #endif
        },(msg)=>{
            console.error("结束录音失败："+msg);
        });
    }
    
}
}
```






*⠀*

*⠀*

*⠀*

*⠀*

# 部分原理和需要注意的细节
## 编译成H5时录音和权限
编译成H5时，录音功能由Recorder H5提供，无需额外处理录音权限。


*⠀*

## 编译成微信小程序时录音和权限
编译成微信小程序时，录音功能由小程序的`RecorderManager`提供，屏蔽了微信原有的底层细节（无录音时长限制）。

小程序录音需要用户授予录音权限，调用`RecordApp.RequestPermission`的时候会检查是否能正常录音，如果用户拒绝了录音权限，会进入错误回调，回调里面你应当编写代码检查`wx.getSetting`中的`scope.record`录音权限，然后引导用户进行授权（可调用`wx.openSetting`打开设置页面，方便用户给权限）。

更多细节请参考 [miniProgram-wx](https://gitee.com/xiangyuecn/Recorder/tree/master/app-support-sample/miniProgram-wx) 测试项目文档。


*⠀*

## 编译成App时录音和权限
编译成App录音时，分两种情况：
1. 默认未配置`RecordApp.UniNativeUtsPlugin`（未使用原生录音插件和uts插件）时，会在renderjs中使用Recorder H5进行录音，录音数据会实时回传到逻辑层。
2. 配置了`RecordApp.UniNativeUtsPlugin`使用原生录音插件或uts插件时，会直接调用原生插件进行录音；录音数据默认会传递到renderjs中进行音频编码处理（WebWorker加速），然后再实时回传到逻辑层，如果配置了`RecordApp.UniWithoutAppRenderjs=true`时，音频编码处理将会在逻辑层中直接处理。

当App是在renderjs中使用H5进行录音时（未使用原生录音插件和uts插件），iOS上只支持14.3以上版本，**且iOS上每次进入页面后第一次请求录音权限时、或长时间无操作再请求录音权限时WebView均会弹出录音权限对话框**，不同旧iOS版本（低于iOS17）下H5录音可能存在的问题在App中同样会存在；使用配套的[原生录音插件](https://ext.dcloud.net.cn/plugin?name=Recorder-NativePlugin)或uts插件时无以上问题和版本限制（uts插件开发中暂不可用），Android也无以上问题。

当音频编码是在renderjs中进行处理时，录音结束后会将整个录音文件传回逻辑层，由于uni-app的逻辑层和renderjs层大点的文件传输会比较慢，**建议Start时使用takeoffEncodeChunk实时获取音频文件数据可避免Stop时产生超大数据回传**；配置了`RecordApp.UniWithoutAppRenderjs=true`后，因为音频编码直接是在逻辑层中进行，将不存在传输性能损耗，但会影响逻辑层的性能（正常情况轻微不明显），需要配套使用原生录音插件才可以进行此项配置。

在调用`RecordApp.RequestPermission`的时候，`Recorder-UniCore`组件会自动处理好App的系统录音权限，只需要在uni-app项目的 `manifest.json` 中配置好Android和iOS的录音权限声明。
```
//Android需要勾选的权限，第二个也必须勾选
<uses-permission android:name="android.permission.RECORD_AUDIO"/>
<uses-permission android:name="android.permission.MODIFY_AUDIO_SETTINGS"/>
【注意】Android如果需要在后台录音，需要启用后台录音保活服务，Android 9开始，锁屏或进入后台一段时间后App可能会被禁止访问麦克风导致录音静音、无法录音（renderjs中H5录音、原生插件录音均受影响），请调用配套原生插件的`androidNotifyService`接口，或使用第三方保活插件

//iOS需要声明的权限
NSMicrophoneUsageDescription
【注意】iOS需要在 `App常用其它设置`->`后台运行能力`中提供`audio`配置，不然App切到后台后立马会停止录音
```


*⠀*

## 语音通话、回声消除、声音外放
在App、H5中，使用[BufferStreamPlayer](https://gitee.com/xiangyuecn/Recorder/tree/master/src/extensions/buffer_stream.player.js)可以实时播放语音；其中App中需要在renderjs中加载BufferStreamPlayer，在逻辑层中调用`RecordApp.UniWebViewVueCall`等方法将逻辑层中接收到的实时语音数据发送到renderjs中播放；播放声音的同时进行录音，声音可能会被录进去产生回声，因此一般需要打开回声消除。

微信小程序请参考 [miniProgram-wx](https://gitee.com/xiangyuecn/Recorder/tree/master/app-support-sample/miniProgram-wx) 文档里面的同名章节，使用WebAudioContext播放。

配置audioTrackSet可尝试打开回声消除，或者切换听筒播放或外放，打开回声消除时，一般会转为听筒播放显著降低回声。
``` js
//打开回声消除
RecordApp.Start({
    ... 更多配置参数请参考RecordApp文档
    //此配置App、H5、小程序均可打开回声消除；注意：H5、App+renderjs中需要在请求录音权限前进行相同配置RecordApp.RequestPermission_H5OpenSet后此配置才会生效
    ,audioTrackSet:{echoCancellation:true,noiseSuppression:true,autoGainControl:true}
    
    //Android指定麦克风源（App搭配原生插件、小程序可用），0 DEFAULT 默认音频源，1 MIC 主麦克风，5 CAMCORDER 相机方向的麦，6 VOICE_RECOGNITION 语音识别，7 VOICE_COMMUNICATION 语音通信(带回声消除)
    ,android_audioSource:7 //提供此配置时优先级比audioTrackSet更高，默认值为0
    
    //iOS的AVAudioSession setCategory的withOptions参数值（App搭配原生插件可用），取值请参考配套原生插件文档中的iosSetDefault_categoryOptions
    //,ios_categoryOptions:0x1|0x4 //默认值为5(0x1|0x4)
});

//App搭配原生插件时尝试切换听筒播放或外放
await RecordApp.UniNativeUtsPluginCallAsync("setSpeakerOff",{off:true或false});
//小程序尝试切换
wx.setInnerAudioOption({ speakerOn:false或true })
//H5不支持切换
```




*⠀*

*⠀*

*⠀*

# 详细文档、RecordApp方法、属性文档
请先阅读 [demo_UniApp文档](https://gitee.com/xiangyuecn/Recorder/tree/master/app-support-sample/demo_UniApp)，含Demo项目；更高级使用还需深入阅读 [Recorder文档](https://gitee.com/xiangyuecn/Recorder)、[RecordApp文档](https://gitee.com/xiangyuecn/Recorder/tree/master/app-support-sample) （均为完整的一个README.md文档），Recorder文档中包含了更丰富的示例代码：基础录音、实时处理、格式转码、音频分析、音频混音、音频生成 等等，大部分能在uniapp中直接使用。




*⠀*

*⠀*

*⠀*

# 本组件的授权许可限制
**本组件内的app-uni-support.js文件在uni-app中编译到App平台时仅供测试用（App平台包括：Android App、iOS App），不可用于正式发布或商用，正式发布或商用需先到DCloud插件市场购买[此带授权的插件](https://ext.dcloud.net.cn/plugin?name=Recorder-NativePlugin-Android)（费用为¥199元，赠送Android版原生插件），即可获得授权许可**；编译到其他平台时无此授权限制，比如：H5、小程序，均为免费授权。

在App中，如果未获得授权许可，将会在App打开后第一次调用`RecordApp.RequestPermission`请求录音权限时，弹出“未获得商用授权时，App上仅供测试”提示框。

在DCloud插件市场购买了[带授权的插件](https://ext.dcloud.net.cn/plugin?name=Recorder-NativePlugin-Android)获得了授权后，请在调用`RecordApp.RequestPermission`请求录音权限前，赋值`RecordApp.UniAppUseLicense="我已获得UniAppID=***的商用授权"`（星号为你项目的uni-app应用标识），就不会弹提示框了；或者直接使用配套的[原生录音插件](https://ext.dcloud.net.cn/plugin?name=Recorder-NativePlugin)，设置`RecordApp.UniNativeUtsPlugin`参数后，也不会弹提示框；其他情况请联系作者咨询，更多细节请参考[本组件的GitHub文档](https://gitee.com/xiangyuecn/Recorder/tree/master/app-support-sample/demo_UniApp)。

获取授权、需要技术支持、或有不清楚的地方可以联系我们，客服联系方式：QQ 1251654593 ，或者直接联系作者QQ 753610399 （回复可能没有客服及时）。

插件开发维护不易，感谢支持~


*⠀*

*⠀*


