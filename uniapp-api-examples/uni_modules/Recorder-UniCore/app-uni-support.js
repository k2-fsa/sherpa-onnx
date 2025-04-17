/**
本代码为RecordApp在uni-app下使用的适配代码，为压缩版（功能和源码版一致）
GitHub、详细文档、许可及服务协议: https://github.com/xiangyuecn/Recorder/tree/master/app-support-sample/demo_UniApp

【授权】
在uni-app中编译到App平台时仅供测试用（App平台包括：Android App、iOS App），不可用于正式发布或商用，正式发布或商用需先联系作者获取到商用授权许可

在uni-app中编译到其他平台时无此授权限制，比如：H5、小程序，均为免费授权

获取商用授权方式：到DCloud插件市场购买授权 https://ext.dcloud.net.cn/plugin?name=Recorder-NativePlugin-Android （会赠送Android版原生插件）；购买后可联系客服，同时提供订单信息，客服拉你进入VIP支持QQ群，入群后在群文件中可下载此js文件最新源码

客服联系方式：QQ 1251654593 ，或者直接联系作者QQ 753610399 （回复可能没有客服及时）。
**/

/***
录音 RecordApp: uni-app支持文件，支持 H5、App vue、App nvue、微信小程序
GitHub、详细文档、许可及服务协议: https://github.com/xiangyuecn/Recorder/tree/master/app-support-sample/demo_UniApp

DCloud插件地址：https://ext.dcloud.net.cn/plugin?name=Recorder-UniCore
App配套原生插件：https://ext.dcloud.net.cn/plugin?name=Recorder-NativePlugin

全局配置参数：
	RecordApp.UniAppUseLicense:"" App中使用的授权许可，获得授权后请赋值为"我已获得UniAppID=***的商用授权"（***为你项目的uni-app应用标识），设置了UniNativeUtsPlugin时默认为已授权；如果未授权，将会在App打开后第一次调用`RecordApp.RequestPermission`请求录音权限时，弹出“未获得商用授权时，App上仅供测试”提示框。
	
	RecordApp.UniNativeUtsPlugin:null App中启用原生录音插件或uts插件，由App提供原生录音，将原生插件或uts插件赋值给这个变量即可开启支持；使用原生录音插件只需赋值为{nativePlugin:true}即可（提供nativePluginName可指定插件名字，默认为Recorder-NativePlugin），使用uts插件只需import插件后赋值即可（uts插件还未开发，目前不可集成）；如果未提供任何插件，App中将使用H5录音（在renderjs中提供H5录音）。
	
	RecordApp.UniWithoutAppRenderjs:false 不要使用或没有renderjs时，应当设为true，此时App中RecordApp完全运行在逻辑层，比如nvue页面，此时音频编码之类的操作全部在逻辑层，需要提供UniNativeUtsPlugin配置由原生插件进行录音，可视化绘制依旧可以在renderjs中进行。默认为false，RecordApp将在renderjs中进行实际的工作，然后将处理好的数据传回逻辑层，数据比较大时传输会比较慢。

不同平台环境下使用说明：
	【H5】 引入RecordApp和本js，按RecordApp的文档使用即可，和普通网页开发没有区别

	【微信小程序】 引入RecordApp和本js，同时引入RecordApp中的app-miniProgram-wx-support.js即可，录音操作和H5完全相同，其他可视化扩展等使用请参考RecordApp中的小程序说明
	
	【App vue】 引入RecordApp和本js，并创建一个<script module="xxx" lang="renderjs">，在renderjs中也引入RecordApp和本js，录音操作和H5大部分相同，部分回调需要多编写一个renderjs的处理代码，比如onProcess_renderjs，具体的请参考RecordApp文档中的app-support-sample/demo_UniApp文档
	
	【App nvue】 引入RecordApp和本js，配置RecordApp.UniWithoutAppRenderjs=true 和提供RecordApp.UniNativeUtsPlugin，录音操作和H5完全相同，但不支持可视化扩展
***/
!function(e){var t="object"==typeof window&&!!window.document,n=t?window:Object,i="https://github.com/xiangyuecn/Recorder/tree/master/app-support-sample/demo_UniApp";if(n.RecordApp){var r=n.Recorder,a=r.i18n;!function(y,C,e,h,W){"use strict";var B=C.RecordApp,P=B.CLog,j=function(){};B.UniSupportLM="2024-10-20 17:31";var I="app-uni-support.js",V=!1,M=!1,N=!1,E=!1,O=!1;(function(){/* #ifdef APP */if(W){V=!0;var e=navigator.userAgent.replace(/[_\d]/g," ");M=!/\bandroid\b/i.test(e)&&/\bios\b|\biphone\b/i.test(e)}else"object"==typeof plus&&("Android"==plus.os.name?V=!0:"iOS"==plus.os.name&&(M=V=!0)),(N=V)||P("App !plus",1)/* #endif */})(),V||((function(){/* #ifdef H5 */E=!0/* #endif */})(),(function(){/* #ifdef MP-WEIXIN */O=!0/* #endif */})());B.UniIsApp=function(){return V?M?2:1:0};var T=B.UniBtoa=function(e){if("object"==typeof uni&&uni.arrayBufferToBase64)return uni.arrayBufferToBase64(e);for(var t=new Uint8Array(e),n="",i=0,r=t.length;i<r;i++)n+=String.fromCharCode(t[i]);return btoa(n)},k=B.UniAtob=function(e){if("object"==typeof uni&&uni.base64ToArrayBuffer)return uni.base64ToArrayBuffer(e);for(var t=atob(e),n=new Uint8Array(t.length),i=0,r=t.length;i<r;i++)n[i]=t.charCodeAt(i);return n.buffer};B.UniB64Enc=function(e){if("object"==typeof uni&&uni.arrayBufferToBase64){var t=B.UniStr2Buf(e);return uni.arrayBufferToBase64(t)}return btoa(unescape(encodeURIComponent(e)))},B.UniB64Dec=function(e){if("object"==typeof uni&&uni.base64ToArrayBuffer){var t=uni.base64ToArrayBuffer(e);return B.UniBuf2Str(t)}return decodeURIComponent(escape(atob(e)))},B.UniStr2Buf=function(e){for(var t=unescape(encodeURIComponent(e)),n=new Uint8Array(t.length),i=0,r=t.length;i<r;i++)n[i]=t.charCodeAt(i);return n.buffer},B.UniBuf2Str=function(e){for(var t=new Uint8Array(e),n="",i=0,r=t.length;i<r;i++)n+=String.fromCharCode(t[i]);return decodeURIComponent(escape(n))};var D=B.UniJsSource={IsSource:!1,pcm_sum:function(e){for(var t=0,n=0;n<e.length;n++)t+=Math.abs(e[n]);return t}};(function(initMemory){!function(){var A=g;!function(e,t){for(var n=g,r=U();;)try{if(271983===-parseInt(n(400))/1+parseInt(n(218))/2+-parseInt(n(240))/3*(parseInt(n(467))/4)+parseInt(n(187))/5*(parseInt(n(207))/6)+parseInt(n(476))/7+-parseInt(n(404))/8*(parseInt(n(128))/9)+parseInt(n(515))/10)break;r.push(r.shift())}catch(e){r.push(r.shift())}}();var o={Support:function(e){var t=g;return O?(B[t(130)][t(384)]||P(h(t(370),0,t(506)),1),void e(!1)):E?void e(!1):V?void(!W||B[t(424)]?e(!0):e(!1)):(P(h(t(488)),3),void e(!1))},CanProcess:function(){return!0}};B[A(437)](W?A(423):A(283),o),V&&(P[A(390)]=W?A(179):A(407)),B[A(517)]||(B[A(517)]={id:0,pageShow:{}});var d=function(){return V&&!W&&!B[A(453)]};B[A(544)]=function(e){var t=A,n=B[t(517)][t(441)]={};if(O&&B[t(537)]&&B[t(537)](),d()){n[t(493)]=p(e);var r=B[t(401)];if(r){for(var i=getCurrentPages(),a=!0,o=0,s=i[t(248)];o<s;o++)if(i[o][t(542)].id==r){a=!1;break}a&&(B[t(355)]=null,B[t(401)]=null,B[t(436)]=null)}}},B[A(178)]=function(e){var t=A;if(d()){B[t(259)]=!0,B[t(331)]=1,setTimeout(function(){B[t(331)]=0});var n=v(e);if(n&&n[t(230)]&&n[t(230)][t(463)]){var r=e[t(196)]||e.$&&e.$[t(464)],i=s(e);i&&r?(r==B[t(355)]&&i==B[t(401)]||P(h(t(368))+t(534)+i+t(249)+r),B[t(355)]=r,B[t(401)]!=i&&(B[t(401)]=i,B[t(436)]=n[t(230)][t(463)]())):P(h(t(167))+t(291),1)}else P(h(t(357))+a(),1)}},B[A(478)]=function(e){var t=A;if(V&&W){if(e[t(332)])var n=window[t(483)],r=e[t(196)]||e[t(332)][t(213)][t(497)],i=e[t(332)][t(292)];if(i)if(i[t(239)]=e,n&&r){var a=t(422)+n+t(249)+r;B[t(353)]=a,i[t(365)](t(438),a),B[t(517)][a]?P(h(t(321))+t(346)+a,3):(B[t(517)][a]=1,P(h(t(457))+t(443)+a))}else P(h(t(415))+t(291),1);else P(h(t(500)),1)}};var p=function(e,t,n){var r=A;if(e){if(e[r(417)])return e[r(417)];var i=s(e),a=e[r(196)]||e.$&&e.$[r(464)]}if(t)if(n||R(),r(281)==t)i=B[r(210)],a=B[r(233)];else i=B[r(401)],a=B[r(355)];return i&&a?r(422)+i+r(249)+a:""},f=function(e){var t=A;return t(251)===e||t(281)===e?{Rec_WvCid:p(null,e)}:{Rec_WvCid:e||"?"}},s=function(e){var t=A,n=v(e);return(n=n&&n[t(542)])&&n.id||0},v=function(e){var t=A,n=e[t(408)];return n&&n[t(230)]&&n[t(542)]?n:e[t(230)]&&e[t(542)]?e:void P(h(t(166)),1)},R=function(e){var t=A;if(!B[t(436)])return h(t(303));var n=p(null,1,1),r=B[t(517)][t(441)][t(493)];if(e){if(!B[t(210)])return h(t(190));if(p(null,t(281),1)!=n)return h(t(171))}return r&&r!=n&&P(h(t(173),0,r,n),3),""};B[A(252)]=function(e,i,a){var c=A,t="";t||N||(t=h(c(447)));var o=!t&&function(e,t){var n=c;if(e&&e[n(417)])var r=/^wv_(\d+)_/[n(204)](e[n(417)]),i=r&&r[1];else{var a=e&&v(e),o=a&&a[n(230)];i=(a=a&&a[n(542)])&&a.id}if(i){if(i==B[n(401)])return B[n(436)];if(o)return o[n(463)]();var s=plus[n(335)][n(325)](i);if(s)return s}return t?(R(),B[n(436)]):null}(e,null==e);if(t||o||(t=h(c(null==e?533:454))),t)return t+=h(c(191)),P(t+c(530)+i[c(343)](0,200),1),t;var n=B[c(517)][c(441)];if(n[c(306)]||(n[c(306)]=1,r()),a){a instanceof ArrayBuffer||(P(c(285),1),a[c(307)]instanceof ArrayBuffer&&(a=a[c(307)]));var s=("a"+Math[c(402)]())[c(427)](".",""),u=0,l=function(){var e=c;if(0!=u&&u>=a[e(485)])o[e(301)](e(336)+s+e(268)+s+e(305)+i+e(132));else{var t=B[e(322)](l),n=u;u+=524288;var r=a[e(300)](n,u);o[e(301)](e(377)+s+e(381)+s+e(188)+a[e(485)]+e(435)+T(r)+e(531)+t+e(150))}};l()}else o[c(301)](c(465)+i+c(471))},B[A(192)]=function(e,t,n){var r=A,i="";r(234)==typeof t&&(i=t[r(111)]||"",t=t[r(119)]||"");var a="";a||N||(a=h(r(203)));var o=!a&&p(e,null==e);if(a||o||(a=h(r(null==e?113:469))),a)return a+=h(r(186)),P(a+r(530)+t[r(343)](0,200),1),a;B[r(252)](e,r(387)+i+r(399)+JSON[r(502)](h(r(363)))+r(523)+o+r(224)+JSON[r(502)](h(r(541)))+r(540)+JSON[r(502)](h(r(477)))+r(481)+t+r(403),n)},B[A(503)]=function(d,f,v,A){return new Promise(function(n,r){var i=g,a=(f=f||{})[i(495)]||"",o=-1==f[i(372)],t="",s=setTimeout(function(){var e=i;c(),s=0;var t=new Error(h(e(o?227:426),0,a));t[e(461)]=1,r(t)},o?2e3:f[i(372)]||5e3),c=function(){var e=B[i(517)];delete e[u],delete e[t]};o&&(t=B[i(322)](function(){clearTimeout(s)}));var e=function(e){var t=i;if(c(),s)return clearTimeout(s),s=0,e[t(299)]?n({value:e[t(421)],bigBytes:B[t(522)](e[t(299)])}):e[t(221)]?n(e[t(421)]):void r(new Error(a+e[t(394)]))},u=B[i(322)](e),l=i(512)+u+i(293)+u+i(145)+w+i(123)+u+i(429)+(o?i(410)+w+i(123)+t+i(545):"")+i(460)+JSON[i(502)](h(i(532)))+i(501)+JSON[i(502)](h(i(425),0,i(416)+I+'"'))+i(391);f[i(342)]?l+=v:l={preCode:l+=i(475),jsCode:v};var p=B[f[i(342)]?i(252):i(192)](d,l,A);p&&e({errMsg:p})})};var w=A(396),r=function(){var i=A;if(N&&i(412)!=typeof UniServiceJSBridge){var e=B[i(484)];if(e){var t="";try{t=uni[i(492)](w)}catch(e){}if(e==t)return;P(h(i(170)),3)}e="r"+Math[i(402)]();try{uni[i(175)](w,e)}catch(e){}B[i(484)]=e,UniServiceJSBridge[i(118)](w),UniServiceJSBridge[i(311)](w,function(e){var t=i,n=e[t(209)]||"";if(t(356)!=n)if(t(452)!=n)if(-1==n[t(521)](t(450)))-1==n[t(521)](t(295))?P(h(t(358))+JSON[t(502)](e),1):B[t(225)](e);else{var r=B[t(517)][n];r?r(e):P(h(t(286))+JSON[t(502)](e),3)}else J(e);else F(e)})}};B[A(322)]=function(t){var e=A,n=B[e(517)],r=++n.id,i=e(450)+r;return n[i]=function(e){delete n[i],t(e)},i},B[A(290)]=function(e,t){var n=A,r=B[n(517)],i=n(169)+e;return t?r[i]=t:delete r[i],i},B[A(265)]=function(e){UniViewJSBridge[A(220)](w,e)},B[A(143)]=function(r,i,e){var a=A;if(W&&V){var o=B[a(353)];if(o){r instanceof ArrayBuffer||(P(a(201),1),r[a(307)]instanceof ArrayBuffer&&(r=r[a(307)]));var s=B[a(517)],c=0,u=++s.id;s[a(295)+u]=function(e){c=e,t()};var l=0,t=function(){var e=a;if(0!=l&&l>=r[e(485)])return delete s[e(295)+u],void i(c);var t=l;l+=524288;var n=r[e(300)](t,l);B[e(265)]({action:e(t?428:528),wvCid:o,wvID:u,mainID:c,b64:T(n)})};t()}else e(h(a(317)))}else e(h(a(371)))},B[A(225)]=function(e){var t=A,n=e[t(367)],r=B[t(517)],i=t(295);t(528)==e[t(209)]&&(n=++r.id,r[i+n]={memory:new Uint8Array(2097152),mOffset:0});var a=r[i+n];if(a){var o=new Uint8Array(k(e[t(491)])),s=o[t(248)];if(a[t(434)]+s>a[t(161)][t(248)]){var c=new Uint8Array(a[t(161)][t(248)]+Math[t(333)](2097152,s));c[t(398)](a[t(161)][t(134)](0,a[t(434)])),a[t(161)]=c}a[t(161)][t(398)](o,a[t(434)]),a[t(434)]+=s,B[t(252)](f(e[t(144)]),t(223)+i+e[t(255)]+t(277)+n+t(222))}else P(h(t(490)),3)},B[A(522)]=function(e){var t=A;if(!N)return null;var n=B[t(517)],r=n[t(295)+e];return delete n[t(295)+e],r?r[t(161)][t(307)][t(300)](0,r[t(434)]):null},B[A(338)]=function(n,i,a,r){var o=A;a=a||j,r=r||j;var s=function(e){var t=g;r(h(t(339),0,n)+(e[t(294)]||e[t(394)]))};if(O){var e=wx[o(364)][o(181)]+"/"+n;wx[o(413)]()[o(376)]({filePath:e,encoding:o(261),data:i,success:function(){a(e)},fail:s})}else N?plus.io[o(214)](plus.io[o(536)],function(e){var t=o;e[t(272)][t(232)](n,{create:!0},function(n){var r=t;n[r(514)](function(e){var t=r;e[t(114)]=function(){a(n[t(385)])},e[t(397)]=s;try{e[t(194)](T(i))}catch(e){s(e)}},s)},s)},s):r(h(o(282)))};var i=function(e){var t=A;if(_(),N){var n=h(t(518),0,y),r=B[t(262)];r&&(!e&&B[t(362)]||(B[t(362)]=1,r[t(176)]?P(h(t(174))+n):P(h(t(140))+n))),B[t(453)]?r?!e&&B[t(470)]||(B[t(470)]=1,P(h(t(328))+n)):P(h(t(297))+n,1):B[t(259)]&&(B[t(436)]?!e&&B[t(414)]||(B[t(414)]=1,P(h(t(146))+n)):P(h(t(373))+a()+n,1))}},a=function(){return h(A(273))};B[A(433)]=function(e,t,n,i){var r=A,a=[];if(O){var o=function(n){var r=g;n>=t[r(248)]?i[r(219)](e,a):e[r(287)]()[r(126)](t[n])[r(382)]({node:!0})[r(204)](function(e){var t=r;a[t(260)](e[0][t(395)]),o(n+1)})};o(0)}else if(E){for(var s=0,c=t[r(248)];s<c;s++){var u=t[s],l=e[r(292)][r(148)](u+r(510)),p=l[0],d=l[1];p?(d&&(d[r(352)](r(276))||(p=l[1],d=l[0]),d[r(480)][r(266)](d)),p[r(482)][r(315)]=r(313),(d=document[r(513)](r(211)))[r(365)](r(276),"1"),d[r(482)][r(348)]=d[r(482)][r(267)]=r(310),p[r(480)][r(406)](d)):P(h(r(494),0,u),1),a[r(260)](d)}i[r(219)](e,a)}else{if(N){var f=[];for(s=0,c=t[r(248)];s<c;s++)f[r(260)](r(351)+t[s]+r(505)+(s+1)+r(345));return f[r(260)](n),void B[r(192)](e,f[r(519)]("\n"))}P(h(r(330)),1)}};var m=function(){var r=A;S(r(139),{},null,null,function(e){var t=r,n=e[t(209)];t(168)==n?e[t(288)]?P("["+i+"]["+e[t(495)]+"]"+e[t(294)],1):P("["+i+"]["+e[t(495)]+"]"+e[t(294)]):t(383)==n?B[t(535)](e[t(466)],e[t(448)]):t(138)==n||P(h(t(284),0,i)+t(430)+n,3),B[t(246)]&&B[t(246)](e)});var e=B[r(262)],i=e&&e[r(176)]?l:r(155);e&&(B[r(278)]=1)},c=A(165),u=A(458)+c,l=c,_=B[A(341)]=function(){var e=A,t=B[e(262)],n="";if(!V)return"";if(!t)return B[e(158)]||h(e(445));if(W&&(n=h(e(117))),!n&&t[e(176)]){if(!B[e(129)]){for(var r=0,i=l=t[e(516)]||l,a=0;!r&&a<2;a++){try{r=uni[e(418)](i)}catch(e){}if(r||i!=c)break;P(h(e(473),0,c,i=c+"-"+(M?e(208):e(543))))}if(B[e(129)]=r)P(h(e(189),0,i));else{i=l==c?u:l;n=h(e(525),0,i)}}}else n||t[e(149)]||(n=h(e(498)));return n&&(B[e(262)]=null,P(n,1)),B[e(158)]=n},S=function(e,t,n,r,i){var a=A,o=_(),s=B[a(262)];if(s){var c={action:e,args:t||{}};i||(i=function(e){var t=a;t(200)==e[t(380)]?n&&n(e[t(421)],e):r&&r(e[t(294)])}),s[a(176)]?B[a(129)][a(149)](c,i):s[a(149)](c,i)}else r&&r(o)};function U(){var e=initMemory;return(U=function(){return e})()}function g(e,t){var n=U();return(g=function(e,t){return n[e-=111]})(e,t)}B[A(131)]=function(r,i){return new Promise(function(t,n){var e=g;if(!N)return n(new Error(h(e(456))));B[e(278)]||m(),S(r,i,function(e){t(e)},function(e){n(new Error(e))})})},o[A(486)]=function(e,t){i(),e()},o[A(361)]=function(){return e(A(361))},o[A(392)]=function(){return e(A(392))};var e=function(e){var t=A;if(!d())return!1;var n=q[t(309)];if(n){var r=R(1);r?P(r,1):B[t(252)](f(n[t(160)]),t(432)+e+"()")}else P(h(t(304),0,e),3)};o[A(411)]=function(e,t,n){var s=A,r=q[s(309)];q[s(309)]=null,r&&d()&&B[s(252)](f(r[s(160)]),s(324)),!d()||B[s(331)]?(B[s(210)]=B[s(401)],B[s(233)]=B[s(355)],i(!0),function(r){var i=s;if(!N)return r();var e=B[i(369)]=B[i(369)]||{},n=function(e,t,n){P(h(i(487),0,I)+e,t||0),n||r()},t=B[i(262)];if(t||e[i(136)])return e[i(136)]=e[i(136)]||(t[i(176)]?2:1),2==e[i(136)]?n(h(i(141))):n(h(i(245)));var a=i(538)+(e[i(115)]=e[i(115)]||uni[i(195)]()[i(115)]||"0")+i(127);if(B[i(270)]){if(B[i(270)]==a)return n(a);P(h(i(241),0,a),3)}var o=function(e){var t=i;n(t(137)+I+t(177)+a+t(279)+u+t(378)+y+" ",3,e)};if(e[i(142)])return o();o(1),e[i(142)]=1,uni[i(451)]({title:i(153),content:"文件"+I+i(479),showCancel:!1,confirmText:i(125),complete:function(){r()}})}(function(){b(e,t,n)})):n(h(s(280)))};var b=function(i,a,v){var o=A;if(W)return B[o(424)]?void a():void v(h(o(327)));var e=function(p){var d=o;if(M){P(h(d(184)));var f=function(){var e=d;if(B[e(508)])p();else{var t=[],n=B[e(226)],r=e(180);if(!n){var i=plus[e(247)][e(524)](e(269));t[e(260)](i);var a=i[e(520)]();t[e(260)](a);var o=a[e(455)]();t[e(260)](o),n=B[e(226)]=o[e(243)]({objectForKey:r})}if(n){var s=plus[e(247)][e(524)](e(202))[e(244)](),c=s[e(319)]();1970168948==c?s[e(366)](f):1735552628==c?(P(h(e(431))+" "+r+":"+n),p()):(P(h(e(511))+e(159)+c,1),v(h(e(409)),!0)),t[e(260)](s)}else v(h(e(526),0,r));for(var u=0,l=t[e(248)];u<l;u++)plus[e(247)][e(237)](t[u])}};f()}else P(h(d(386))),plus[d(112)][d(420)]([d(334)],function(e){var t=d;0<e[t(347)][t(248)]?(P(h(t(257))+JSON[t(502)](e)),p()):(P(h(t(198)),1,e),v(h(t(254)),!0))},function(e){var t=d;P(h(t(197))+e[t(294)],1,e),v(h(t(440))+e[t(294)])})},t=function(e){var t=o;m(),S(t(319),{},e,v)};if(B[o(453)])e(function(){t(a)});else{var s=f(o(281)),n=function(e){var n=o,t=R(1),r=h(n(340));t?v(r+t):B[n(503)](s,{tag:r,timeout:2e3,useEval:!0},n(442))[n(312)](function(){e()})[n(375)](function(e){var t=n;v(e[t(461)]?r+h(t(135)):e[t(294)])})},r=function(e){var n=o;if(B[n(172)](i)){var t=n(162),r=B[t]||{};B[n(503)](s,{timeout:-1},n(185)+!!e+n(289)+t+"="+JSON[n(502)](r)+n(439))[n(312)](function(e){var t=n;e.ok?a():v(e[t(394)],e[t(182)])})[n(375)](function(e){v(e[n(294)])})}else v(n(154))};B[o(262)]?n(function(){e(function(){t(function(){r(!0)})})}):n(function(){e(function(){r()})})}};o[A(263)]=function(t,o,n,s){var c=A,e=q[c(309)];if(q[c(309)]=null,e&&d()&&B[c(252)](f(e[c(160)]),c(324)),!d()||B[c(331)]){q[c(216)]=o;var u=C(o);if(u[c(398)][c(229)]=!0,u[c(235)]=c(122),q[c(354)]=!1,q[c(309)]=u,B[c(316)]=u,W)return B[c(424)]?void n():void s(h(c(231)));var r=function(t){var n=c,e=JSON[n(302)](JSON[n(502)](l));e[n(151)]=e[n(151)]||B[n(157)]||0,e[n(446)]=e[n(448)],e[n(448)]=48e3;var r=(e[n(120)]||{})[n(379)],i=e[n(360)];r&&null==i&&(i=1,e[n(360)]=!0),M||null!=e[n(474)]||(e[n(474)]=i?7:B[n(147)]||"0"),P(n(350)+JSON[n(502)](e)),m(),S(n(419),e,function(){var e=n;B[e(124)]=setInterval(function(){S(e(349),{},function(){})},5e3),t()},s)};clearInterval(B[c(124)]);var l={};for(var i in o)/_renderjs$/[c(462)](i)||(l[i]=o[i]);if(l=JSON[c(302)](JSON[c(502)](l)),B[c(453)])r(n);else{u[c(398)][c(116)]=c(296);var a=function(e,t){var n=c,r=R(1);if(r)s(h(n(444))+r);else{u[n(160)]=p(null,n(281)),q[n(354)]=t;var i=[n(264)+JSON[n(502)](l)+";"],a=n(374);i[n(260)](n(206)+(o[n(193)]||0)+n(199)+(o[n(405)]||0)+n(323)+(o[n(337)]||0)+n(329)+a+n(253)+a+n(256)),(o[n(489)]||o[n(250)])&&i[n(260)](n(389)+(o[n(250)]||0)+n(152)),i[n(260)](n(217)),B[n(503)](f(u[n(160)]),{timeout:-1},i[n(519)]("\n"))[n(312)](function(){e()})[n(375)](function(e){s(e[n(294)])})}};B[c(262)]?a(function(){var e=c;B[e(172)](t)?r(n):s(e(154))},!0):a(n)}}else s(h(c(238)))},o[A(133)]=function(e){return!!d()&&""},o[A(527)]=function(e){var t=A;if(!d())for(var n in e)/_renderjs$/[t(462)](n)&&delete e[n]};var F=function(e){var t=A,n=q[t(309)];n&&(n[t(398)][t(448)]=e[t(236)],n[t(398)][t(496)]=e[t(212)]);for(var r=e[t(507)],i=0,a=r[t(248)];i<a;i++)q(r[i],e[t(448)])},J=function(e){var t=A,n=q[t(309)];if(n){var r=new Uint8Array(k(e[t(215)]));n[t(398)][t(489)]&&n[t(398)][t(489)](r)}else P(h(t(271)),3)},q=function(e,t){var n=A,r=q[n(309)];if(r){if(r[n(274)]||r[n(314)]({envName:o[n(183)],canProcess:o[n(163)]()},t),r[n(274)]=1,e instanceof Int16Array)var i=new Int16Array(e);else i=new Int16Array(k(e));var a=D[n(539)](i);r[n(449)](i,a)}else P(h(n(275)),3)};B[A(535)]=function(e,t){var n=A;if(q[n(354)]){var r=q[n(309)];return r?void B[n(252)](f(r[n(160)]),n(298)+e+'",'+t+")"):void P(h(n(156)),3)}q(e,t)},o[A(529)]=function(n,i,r){var a=A,o=function(e){var t=g;B[t(172)](n)&&(q[t(309)]=null,s&&c&&d()&&B[t(252)](f(s[t(160)]),t(324))),r(e)},s=q[a(309)],c=!0,u=i?"":B[a(164)](),e=function(){var e=a;if(B[e(172)](n))if(q[e(309)]=null,s){if(P(e(326)+s[e(468)]+e(205)+s[e(393)]+e(242)+JSON[e(502)](q[e(216)])),!i)return l(),void o(u);s[e(504)](function(e,t,n){l(),i(e,t,n)},function(e){l(),o(e)})}else o(h(e(318))+(u?" ("+u+")":""));else o(e(154))},l=function(){var e=a;if(B[e(172)](n))for(var t in q[e(309)]=null,s[e(398)])q[e(216)][t]=s[e(398)][t]};if(W)return B[a(424)]?void e():void o(h(a(359)));var t=function(e){S(a(509),{},e,o)};if(clearInterval(B[a(124)]),B[a(453)])t(e);else{var p=function(e){var r=a;if(s){var t=R(1);if(t)o(h(r(228))+t);else{var n=r(388)+(i&&q[r(216)][r(472)]||0)+r(459)+!i+r(121);B[r(503)](f(s[r(160)]),{timeout:-1},n)[r(312)](function(e){var t=r;c=!1,s[t(398)][t(116)]=q[t(216)][t(116)],s[t(398)][t(448)]=e[t(236)],s[t(398)][t(496)]=e[t(212)],l();var n=B[t(522)](e[t(258)]);n?i(n,e[t(499)],e[t(308)]):o(h(t(344)))})[r(375)](function(e){c=!1,o(e[r(294)])})}}else o(h(r(320))+(u?" ("+u+")":""))};B[a(262)]?t(function(){var e=a;B[e(172)](n)?p():o(e(154))}):p()}}}();})(["substr","gomD::不应该出现的renderjs发回的文件数据丢失","=el2;\n\t\t\t"," wvCid=","granted","width","recordAlive","Native Start Set:",'\n\t\t\t\tvar cls="',"getAttribute","__UniWvCid","nativeToRjs","__uniAppComponentId","recProcess","GwCz::RecordApp.UniWebViewActivate 需要传入当前页面或组件的this对象作为参数","ZHwv::[MainReceive]从renderjs发回未知数据：","TPhg::不应当出现的非H5录音Stop","appNativePlugin_AEC_Enable","Pause","__nnM6","U1Be::renderjs中未import导入RecordApp","env","setAttribute","requestRecordPermission","mainID","WpKg::RecordApp.UniWebViewActivate 已切换当前页面或组件的renderjs所在的WebView","__FabE","RXs7::微信小程序中需要：{1}","MujG::只允许在renderjs中调用RecordApp.UniWebViewSendBigBytesToMain","timeout","S3eF::未找到当前页面renderjs所在的WebView","buffers,power,duration,sampleRate,newIdx","catch","writeFile","(function(){\n\t\t\tvar cur=window.","-Android （会赠送Android版原生插件）；购买后可联系客服，同时提供订单信息，客服拉你进入VIP支持QQ群，入群后在群文件中可下载此js文件最新源码；客服联系方式：QQ 1251654593 ，或者直接联系作者QQ 753610399 （回复可能没有客服及时）。详细请参考文档: ","echoCancellation","status","=window.","fields","onRecord","miniProgram-wx","fullPath","7Noe::正在调用plus.android.requestPermissions请求Android原生录音权限","(function(){\n\t\tvar CallErr=function(){};\n\t\t","(function(){\n\t\t\tvar stopFn=","var takeFn=","Tag",");\n\t\t};\n\t","Resume","srcSampleRate","errMsg","node","RecordApp__uniAppMainReceive","onerror","set","\n\t\tif(!window.RecordApp){\n\t\t\tvar err=","527412cwWKPd","__uniAppWebViewId","random","\n\t\t}).call(vm);\n\t})()","13912ctAFhn","onProcessBefore_renderjs","appendChild","RecApp Main","$root","0caE::用户拒绝了录音权限",'\n\t\t\tUniViewJSBridge.publishHandler("',"RequestPermission","undefined","getFileSystemManager","__0hyi","Uc9E::RecordApp.UniRenderjsRegister 发生不应该出现的错误（可能需要升级插件代码）：",'"@/uni_modules/Recorder-UniCore/',"Rec_WvCid","requireNativePlugin","recordStart","requestPermissions","value","wv_","UniApp-Renderjs","UniAppUseNative","AN0e::需在renderjs中import {1}","RDcZ::{1}处理超时","replace","bigBytes_chunk",'",errMsg:err});\n\t\t};',"action=","j15C::已获得iOS原生录音权限","RecordApp.","UniFindCanvas","mOffset",'),mOffset:0};\n\t\t\tvar buf=new Uint8Array(RecordApp.UniAtob("',"__uniAppWebView","RegisterPlatform","rec_wv_cid_key",";\n\t\t\tRecordApp.RequestPermission(function(){\n\t\t\t\tCallSuccess({ok:1});\n\t\t\t},function(errMsg,isUserNotAllow){\n\t\t\t\tCallSuccess({errMsg:errMsg,isUserNotAllow:isUserNotAllow});\n\t\t\t});\n\t\t","Mvl7::调用plus的权限请求出错：","pageShow","CallSuccess(1)"," WvCid=","Bjx9::无法调用Start：","H753::未配置RecordApp.UniNativeUtsPlugin原生录音插件","sampleRate_set","TfJX::当前不是App逻辑层","sampleRate","envIn","mainCb_","showModal","recEncodeChunk","UniWithoutAppRenderjs","qDo1::未找到此页面renderjs所在的WebView","infoDictionary","MrBx::需在App逻辑层中调用原生插件功能","7kJS::RecordApp.UniRenderjsRegister 已注册当前页面renderjs模块","https://ext.dcloud.net.cn/plugin?name=",";\n\t\t\tvar clear=","\n\t\tif(!window.RecordApp){\n\t\t\treturn CallFail(","isTimeout","test","$getAppWebview","uid","(function(){\n","pcmDataBase64","108krEIHd","recSize","6Iql::未找到此页面renderjs所在的WebView Cid","__xYRb","\n})()","stop_renderjs","kSjQ::当前App未打包进双端原生插件[{1}]，尝试加载单端[{2}]","android_audioSource","CallErr=function(err){ CallFail(err) };","3624621rlYiJM","URyD::没有找到组件的renderjs模块","UniRenderjsRegister","在uni-app中编译到App平台时仅供测试用，不可用于正式发布或商用，正式发布或商用需先获得授权许可（编译到其他平台时无此授权限制，比如：H5、小程序，均为免费授权）。本对话框仅在第一次请求录音权限时会弹出一次，如何去除本弹框、如何获取商用授权、更多信息请看控制台日志","parentNode",'+" WvCid="+wvCid;\n\t\t\t\tRecordApp.CLog(err,1); CallErr(err); return;\n\t\t\t};\n\t\t\tRecordApp.__uniWvCallVm=vm;\n\t\t\tRecordApp.__uniWvCallWvCid=wvCid;\n\t\t}; (function(){ var This=this;\n\t\t\t',"style","__WebVieW_Id__","__uniAppMainReceiveBind","byteLength","Install","FabE::【在App内使用{1}的授权许可】","4ATo::Recorder-UniCore目前只支持：H5、APP(Android iOS)、MP-WEIXIN，其他平台环境需要自行编写适配文件实现接入","takeoffEncodeChunk","CjMb::无效的BigBytes回传数据","b64","getStorageSync","sWvCid","k7im::未找到Canvas：{1}，请确保此DOM已挂载（可尝试用$nextTick等待DOM更新）","tag","bitRate","ownerId","TGMm::提供的RecordApp.UniNativeUtsPlugin值不是RecordApp的uts原生录音插件","duration","4jKV::RecordApp.UniRenderjsRegister 需在renderjs中调用并且传入当前模块的this",');\n\t\t};\n\t\tif(!RecordApp.Platforms["UniApp-Renderjs"]){\n\t\t\treturn CallFail(',"stringify","UniWebViewCallAsync","stop",'";\n\t\t\t\tvar els=this.$ownerInstance.$el.querySelectorAll(cls+" canvas"),el=els[0],el2=els[1];\n\t\t\t\tif(!el){\n\t\t\t\t\tRecordApp.CLog(Recorder.i18n.$T("dzX0::未找到Canvas：{1}，请确保此DOM已挂载（可尝试用$nextTick等待DOM更新）",0,cls),1);\n\t\t\t\t}else{\n\t\t\t\t\tif(el2){\n\t\t\t\t\t\tif(!el2.getAttribute("el2")){ el=els[1]; el2=els[0] }\n\t\t\t\t\t\tel2.parentNode.removeChild(el2);\n\t\t\t\t\t}\n\t\t\t\t\tel.style.display="none";\n\t\t\t\t\tel2=document.createElement("canvas");\n\t\t\t\t\tel2.setAttribute("el2","1"); el2.style.width=el2.style.height="100%";\n\t\t\t\t\tel.parentNode.appendChild(el2);\n\t\t\t\t}\n\t\t\t\tvar canvas',"import 'recorder-core/src/app-support/app-miniProgram-wx-support.js'","newBuffers","DisableIOSPlusReqPermission","recordStop"," canvas","iKhe::plus.ios请求录音权限，状态值: ",'\n\t\tvar CallSuccess=function(val,buf){\n\t\t\tif(buf){\n\t\t\t\tRecordApp.UniWebViewSendBigBytesToMain(buf,function(dataID){\n\t\t\t\t\tRecordApp.UniWebViewSendToMain({action:"',"createElement","createWriter","3026590akXWwE","nativePluginName","__UniData","1f2V:: | RecordApp的uni-app支持文档和示例：{1} ","join","mainBundle","indexOf","UniMainTakeBigBytes",';\n\t\t\twindow["console"].error(err); CallErr(err); return;\n\t\t};\n\t\tvar wvCid="',"importClass","SCW9::配置了RecordApp.UniNativeUtsPlugin，但当前App未打包进原生录音插件[{1}]","9xoE::项目配置中未声明iOS录音权限{1}","AllStart_Clean","bigBytes_start","Stop","   jsCode=",'"));\n\t\t\tcur.memory.set(buf,cur.mOffset);\n\t\t\tcur.mOffset+=buf.byteLength;\n\t\t\tRecordApp.UniWebViewSendToMain({action:"',"TSmQ::需要在页面中提供一个renderjs，在里面import导入RecordApp、录音格式编码器、可视化插件等","peIm::当前还未调用过RecordApp.UniWebViewActivate"," WvCid=wv_","UniNativeRecordReceivePCM","PUBLIC_DOWNLOADS","MiniProgramWx_onShow","我已获得UniAppID=","pcm_sum",';\n\t\t\t\tRecordApp.CLog(err,1); CallErr(err); return;\n\t\t\t};\n\t\t\tvar el=document.querySelector("[rec_wv_cid_key=\'"+wvCid+"\']");\n\t\t\tvm=el&&el.__rModule;\n\t\t\tif(!vm){\n\t\t\t\tvar err=',"Bcgi::renderjs中的mounted内需要调用RecordApp.UniRenderjsRegister","$page","Android","UniPageOnShow",'"});\n\t\t',"preCode","android","mSbR::当前还未调用过RecordApp.UniWebViewActivate","onwrite","appId","type","l6sY::renderjs中不支持设置RecordApp.UniNativeUtsPlugin","unsubscribe","jsCode","audioTrackSet",';\n\t\t\tvar errFn=function(errMsg){\n\t\t\t\tCallFail(errMsg);\n\t\t\t};\n\t\t\tRecordApp.Stop(clear?null:function(arrBuf,duration,mime){\n\t\t\t\tstopFn&&stopFn.apply(This,arguments);\n\t\t\t\tvar recSet=RecordApp.__Rec.set,t1=Date.now();\n\t\t\t\tRecordApp.CLog("开始传输"+arrBuf.byteLength+"字节的数据回逻辑层，可能会比较慢，推荐使用takeoffEncodeChunk实时获取音频文件数据可避免Stop时产生超大数据回传");\n\t\t\t\tRecordApp.UniWebViewSendBigBytesToMain(arrBuf,function(dataId){//数据可能很大\n\t\t\t\t\tRecordApp.CLog("完成传输"+arrBuf.byteLength+"字节的数据回逻辑层，耗时"+(Date.now()-t1)+"ms");\n\t\t\t\t\tCallSuccess({recSet_sr:recSet.sampleRate,recSet_bit:recSet.bitRate,dataId:dataId,duration:duration,mime:mime});\n\t\t\t\t},errFn);\n\t\t\t},errFn);\n\t\t})()',"arraybuffer",'",{action:"',"_X3Ij_alive","我知道啦","select","的商用授权","927nxBzjM","__uniNP","Platforms","UniNativeUtsPluginCallAsync","\n\t\t\t})()","Start_Check","subarray","KnF0::无法连接到renderjs","uts","当前未获得授权许可。文件","noop","jsCall","nnM6::当前录音由uts插件提供支持","w37G::已购买原生录音插件，获得授权许可","show","UniWebViewSendBigBytesToMain","wvCid",'", isOk:true, value:val});\n\t\t\t}\n\t\t};\n\t\tvar CallFail=function(err){\n\t\t\tUniViewJSBridge.publishHandler("',"0hyi::当前RecordApp运行在renderjs所在的WebView中（逻辑层中只能做有限的实时处理，可视化等插件均需要在renderjs中进行调用）","Default_Android_AudioSource","querySelectorAll","request",'"});\n\t\t})()',"appNativePlugin_sampleRate",';\n\t\t\tset.takeoffEncodeChunk=function(bytes){\n\t\t\t\tRecordApp.UniWebViewSendToMain({action:"recEncodeChunk",bytes:RecordApp.UniBtoa(bytes.buffer)});\n\t\t\t\ttakeFn&&takeFn.apply(This,arguments);\n\t\t\t};',"未获得商用授权时，App上仅供测试哈","Incorrect sync status","RecorderUtsPlugin","byzO::未开始录音，但收到UniNativeUtsPlugin PCM数据","Default_AppNativePlugin_SampleRate","__uniNupErr","denied ","__wvCid","memory","RequestPermission_H5OpenSet","CanProcess","__StopOnlyClearMsg","Recorder-NativePlugin","KpY6::严重兼容性问题：无法获取页面或组件this.$root.$scope或.$page","ipB3::RecordApp.UniWebViewActivate 发生不应该出现的错误（可能需要升级插件代码）：","onLog","mainCb_reg_","vEgr::不应该出现的MainReceiveBind重复绑定","VsdN::需重新调用RecordApp.RequestPermission方法","__Sync","SWsy::检测到有其他页面或组件调用了RecordApp.UniPageOnShow（WvCid={1}），但未调用过RecordApp.UniWebViewActivate（当前WvCid={2}），部分功能会继续使用之前Activate的WebView和组件，请确保这是符合你的业务逻辑，不是因为忘记了调用UniWebViewActivate","XSYY::当前录音由原生录音插件提供支持","setStorageSync","nativePlugin","在uni-app中编译到App平台时仅供测试用（App平台包括：Android App、iOS App），不可用于正式发布或商用，正式发布或商用需先获取到商用授权许可（编译到其他平台时无此授权限制，比如：H5、小程序，均为免费授权）。未获得授权时，在App打开后第一次调用RecordApp.RequestPermission请求录音权限时，会先弹出商用授权提示框；获取到授权许可后，请在调用RequestPermission前设置 RecordApp.UniAppUseLicense='","UniWebViewActivate","RecApp Renderjs","NSMicrophoneUsageDescription","USER_DATA_PATH","isUserNotAllow","Key","Y3rC::正在调用plus.ios@AVAudioSession请求iOS原生录音权限","\n\t\t\tRecordApp.UniAppUseNative=","TtoS::，不可以调用RecordApp.UniWebViewVueCall","6625FCqjSG","||{memory:new Uint8Array(","Xh1W::已加载原生录音插件[{1}]","7ot0::需先调用RecordApp.RequestPermission方法","igw2::，不可以调用RecordApp.UniWebViewEval","UniWebViewVueCall","onProcess_renderjs","writeAsBinary","getSystemInfoSync","_$id","0JQw::plus.android请求录音权限出错：","Ruxl::plus.android请求录音权限：无权限",";\r\n\t\tvar procBefore=","success","UniWebViewSendBigBytesToMain buffer must be ArrayBuffer","AVAudioSession","lU1W::当前不是App逻辑层","exec"," srcSR:","var procFn=","582mpQnKD","iOS","action","__uniAppReqWebViewId","canvas","recSet_bit","$vm","requestFileSystem","bytes","param","RecordApp.Start(set,function(){\n\t\t\tstartFn&&startFn.call(This);\n\t\t\tCallSuccess();\n\t\t},function(errMsg){\n\t\t\tCallFail(errMsg);\n\t\t});","213166JvujtP","apply","publishHandler","isOk",");\n\t})()",'(function(){\n\t\tvar fn=RecordApp.__UniData["','",vm=RecordApp.__uniWvCallVm;\n\t\tif(!vm || RecordApp.__uniWvCallWvCid!=wvCid){\n\t\t\tif(!RecordApp.__UniData[wvCid]){\n\t\t\t\tvar err=',"__UniMainReceiveBigBytes","__9xoE","KQhJ::{1}连接renderjs超时","H6cq::无法调用Stop：","disableEnvInFix","$scope","rSLO::不应当出现的非H5录音Start","getFile","__uniAppReqComponentId","object","dataType","recSet_sr","deleteObject","XCMU::需先调用RecordApp.UniWebViewActivate，然后才可以调用Start","__rModule","8562rdXnDn","aPoj::UniAppUseLicense填写无效，如果已获取到了商用授权，请填写：{1}，否则请使用空字符串"," set:","plusCallMethod","sharedInstance","e71S::已购买uts插件，获得授权许可","UniNativeUtsPlugin_JsCall","ios","length","_cid_","takeoffEncodeChunk_renderjs","@act","UniWebViewEval","){\r\n\t\t\tprocBefore&&procBefore.call(This,","l7WP::用户拒绝了录音权限","wvID",');\n\t\t\tvar newBuffers=[],recSet=RecordApp.__Rec.set;\n\t\t\tfor(var i=newIdx;i<buffers.length;i++)newBuffers.push(RecordApp.UniBtoa(buffers[i].buffer));//@@Fast\n\t\t\tRecordApp.UniWebViewSendToMain({action:"recProcess",recSet_sr:recSet.sampleRate,recSet_bit:recSet.bitRate,sampleRate:sampleRate,newBuffers:newBuffers});\n\t\t\treturn procFn&&procFn.apply(This,arguments);\n\t\t};',"Bgls::已获得Android原生录音权限：","dataId","__hasWvActivate","push","binary","UniNativeUtsPlugin","Start","var set=","UniWebViewSendToMain","removeChild","height",".memory.buffer; delete window.","NSBundle","UniAppUseLicense","MTdp::未开始录音，但收到renderjs回传的onRecEncodeChunk","root","e6Mo::，请检查此页面代码中是否编写了lang=renderjs的module，并且调用了RecordApp.UniRenderjsRegister；如果确实没有renderjs，比如nvue页面，请设置RecordApp.UniWithoutAppRenderjs=true并且搭配配套的原生插件在逻辑层中直接录音","_appStart","BjGP::未开始录音，但收到Uni Native PCM数据","el2",'"];\n\t\tif(fn)fn(',"__uniNbjc","' ，就不会弹提示框了；或者购买了配套的原生录音插件，设置RecordApp.UniNativeUtsPlugin参数后，也不会弹提示框。【获取授权方式】到DCloud插件市场购买授权: ","PkQ2::需先调用RecordApp.UniWebViewActivate，然后才可以调用RequestPermission","@req","kxOd::当前环境未支持保存本地文件","UniApp-Main","dl4f::{1}回传了未知内容，","UniWebViewEval bigBytes must be ArrayBuffer","kZx6::从renderjs发回数据但UniMainCallBack回调不存在：","createSelectorQuery","isError",";\r\n\t\t\tRecordApp.Current=null; //需先重置，不然native变化后install不一致\n\t\t\tRecordApp.","UniMainCallBack_Register","!id || !cid","$el",'", isOk:true, value:val, dataID:dataID});\n\t\t\t\t},CallFail)\n\t\t\t}else{\n\t\t\t\tRecordApp.UniWebViewSendToMain({action:"',"message","bigBytes_","unknown","fqhr::当前已配置RecordApp.UniWithoutAppRenderjs，必须提供原生录音插件或uts插件才能录音，请参考RecordApp.UniNativeUtsPlugin配置",'RecordApp.UniNativeRecordReceivePCM("',"dataID","slice","evalJS","parse","AGd7::需要先调用RecordApp.UniWebViewActivate方法","0FGq::未开始录音，不可以调用{1}",";\n\t\t\t\t","mrBind","buffer","mime","rec","100%","subscribe","then","none","envStart","display","__Rec","kE91::renderjs中的mounted内需要调用RecordApp.UniRenderjsRegister才能调用RecordApp.UniWebViewSendBigBytesToMain","YP4V::未开始录音","recordPermission","pP4O::未开始录音","mzKj::RecordApp.UniRenderjsRegister 重复注册当前页面renderjs模块，一个组件内只允许一个renderjs模块进行注册","UniMainCallBack",";\n\t\tvar startFn=","RecordApp.Stop()","getWebviewById","rec encode: pcm:","Jk72::不应当出现的非H5权限请求","xYRb::当前RecordApp运行在逻辑层中（性能会略低一些，可视化等插件不可用）",";\n\t\tset.onProcess=function(","yI24::RecordApp.UniFindCanvas未适配当前环境","__callWvActivate","$ownerInstance","max","android.permission.RECORD_AUDIO","webview","(function(){\n\t\t\t\tvar BigBytes=window.","start_renderjs","UniSaveLocalFile","UqfI::保存文件{1}失败：","ksoA::无法调用RequestPermission：","UniCheckNativeUtsPluginConfig","useEval"])}(i,r,0,a.$T,t)}else console.error("需要先引入RecordApp，请按下面代码引入：\n1. 项目根目录 npm install recorder-core\n2. 页面中按顺序import\nimport Recorder from 'recorder-core'\nimport RecordApp from 'recorder-core/src/app-support/app.js'\nimport 你需要的音频格式编码器、可视化插件\n参考文档："+i)}();