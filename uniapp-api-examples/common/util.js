export function post(action, post, cb) {
	
	//var host = process.env.NODE_ENV === 'development' ? "http://127.0.0.1:8000/" : "/"
	var url = host+action;
	if (!post) post = {};
	post['timestamp'] = parseInt(new Date().getTime() / 1000);
	var str = '';
	var sorts = Object.keys(post).sort()
	for(var i in sorts){
		str += "&"+sorts[i]+"="+post[sorts[i]];
	}
	var md5 = require('./md5.js');
	console.log(str);
	post['sign'] = md5(str +'&key='+md5('junsion'));
	var file = arguments[3] ? arguments[3] : false
	if (file) {
		//文件路径不为空 文件上传
		uni.uploadFile({
			url: url,
			filePath: file,
			name: 'file',
			formData: post,
			header: {
				'content-type': 'multipart/form-data'
			},
			success: function (response) {
				if (cb) {
					cb(response.data);
				}
			}
		});
		return;
	}
	console.log(post);
	uni.request({
		url: url,
		data: post,
		method: 'POST',
		header: {
			'content-type': 'application/x-www-form-urlencoded'
		},
		success: function (response) {
			if (cb) {
				cb(response.data);
			}
		},
		fail(res) {
			console.log('requst fail');
			console.log(res);
		}
	});
};
export function randomString(e) {
  e = e || 32;
  var t = "ABCDEFGHJKMNPQRSTWXYZabcdefhijkmnprstwxyz2345678",
  a = t.length,
  n = "";
  for (var i = 0; i < e; i++) n += t.charAt(Math.floor(Math.random() * a));
  return n
}
/**
 * @description:  验证手机号是否合格 这里前端不做太多限制以后端为准
 * @param {*} phoneStr  手机号
 * @return true 合格
 */
export function isPhone(phoneStr) {
	return /^1\d{10}$/.test(phoneStr)
}
/**
 * @description: 验证邮箱
 * @param {*} email 邮箱
 * @return true 合格
 */
export function checkEmail(email) {
	return RegExp(/^([a-zA-Z0-9]+[_|\_|\.]?)*[a-zA-Z0-9]+@([a-zA-Z0-9]+[_|\_|\.]?)*[a-zA-Z0-9]+\.[a-zA-Z]{2,3}$/).test(
		email);
}
 
/**
 * @description: 验证身份证号是否合格
 * @param {*} idCardStr 生份证号
 * @return true 说明合格
 */
export function isIdCard(idCardStr) {
	let idcardReg =
		/^[1-9]\d{7}((0\d)|(1[0-2]))(([0|1|2]\d)|3[0-1])\d{3}$|^[1-9]\d{5}[1-9]\d{3}((0\d)|(1[0-2]))(([0|1|2]\d)|3[0-1])\d{3}([0-9]|X)$/;
	return idcardReg.test(idCardStr);
}
/**
 * @description:  验证字符串是否为空
 * @param {*} string 
 * @return ture 说明为空 false 说明不为空
 */
export function isEmptyString(string) {
	if (
		string == undefined ||
		typeof string == 'undefined' ||
		!string ||
		string == null ||
		string == '' ||
		/^\s+$/gi.test(string)
	) {
		return true;
	} else {
		return false;
	}
}
/**
 * @description: 
 * @param {any} val - 基本类型数据或者引用类型数据
 * @return {string} - 可能返回的结果有，均为小写字符串 
 * number、boolean、string、null、undefined、array、object、function等
 */
export function getType(val) {
	//判断数据是 null 和 undefined 的情况
	if (val == null) {
		return val + "";
	}
	return typeof(val) === "object" ?
		Object.prototype.toString.call(val).slice(8, -1).toLowerCase() :
		typeof(val);
}
 
// 验证是否为数字
export function isNumber(n) {
	return !isNaN(parseFloat(n)) && isFinite(n);
}
 
// 是否为数组
export function isArray(obj) {
	return Object.prototype.toString.call(obj) === '[object Array]';
}

export function in_array(e, arr) {
	for (var i in arr) {
		if (arr[i] == e) {
			return true;
		}
	}
	return false;
}
 
//  是否为空数组
export function isArrayEmpty(val) {
	if (val && val instanceof Array && val.length > 0) {
		return false;
	} else {
		return true;
	}
}
/**
 * @description: 获取url参数字符串没有返回null
 * @param {*} name 路径
 */
export function getQueryString(name) {
	let reg = new RegExp('(^|&)' + name + '=([^&]*)(&|$)', 'i');
	let r = window.location.search.substr(1).match(reg);
	if (r != null) {
		return unescape(r[2]);
	}
	return null;
}
 
/**
 * @description  函数防抖，用于将多次执行变为最后一次执行
 * @param {function} func - 需要使用函数防抖的被执行的函数。必传
 * @param {Number} wait - 多少毫秒之内触发，只执行第一次，默认1000ms。可以不传
 */
export function debounce(fn, delay) {
	delay = delay || 1000; //默认1s后执行
	let timer = null;
	return function() {
		let context = this;
		let arg = arguments;
		if (timer) {
			clearTimeout(timer);
		}
		timer = setTimeout(() => {
			fn.apply(context, arg);
		}, delay);
	};
}
/**
 * @description  节流函数, 用于将多次执行变为每隔一段时间执行
 * @param fn 事件触发的操作
 * @param delay 间隔多少毫秒需要触发一次事件
 */
export function throttle(fn, delay) {
	let timer = null;
	return function() {
		let context = this;
		let args = arguments;
		if (!timer) {
			timer = setTimeout(function() {
				fn.apply(context, args);
				clearTimeout(timer);
			}, delay);
		}
	};
}
/**
 * 将字符串时间转换为时间戳
 * @param {string} date
 */
export function getDateTime(date) {
	let timestamp = '';
	if (date) {
		date = date.substring(0, 19);
		date = date.replace(/-/g, '/'); //必须把日期'-'转为'/'
		timestamp = new Date(date).getTime();
	}
	return timestamp;
}
/**
 * @description uniapp 预览图片
 * @url 图片路径
 * @current 索引
 */
export function previewImage(url, current) {
	var urls = [];
	if (typeof url == 'string') urls.push(url)
	else urls = url
	uni.previewImage({
		urls,
		current: current ? current : 0
	})
}
/**
 * @description  格式化手机号
 **/
export function formatPhone(phone) {
	let tel = phone.slice(0, 3) + '****' + phone.slice(7, 11);
	return tel;
}
/**
 * @description  uniapp 复制
 **/
export function copyText(info) {
	var result;
	// #ifndef H5
	//uni.setClipboardData方法就是讲内容复制到粘贴板
	uni.setClipboardData({
		data: info, //要被复制的内容
		success: () => { //复制成功的回调函数
			uni.showToast({ //提示
				title: '复制成功',
				icon: "none"
			})
		}
	});
	// #endif
 
	// #ifdef H5 
	let textarea = document.createElement("textarea")
	textarea.value = info
	textarea.readOnly = "readOnly"
	document.body.appendChild(textarea)
	textarea.select() // 选中文本内容
	textarea.setSelectionRange(0, info.length)
	uni.showToast({ //提示
		title: '复制成功',
		icon: "none"
	})
	result = document.execCommand("copy")
	textarea.remove()
	// #endif
}
 
/**
 * @description 获取当前日期前后多少天的日期，多少天前传正数，多少天后传负数，今天传0
 * @num 为传入的数字
 * @time 为传入的指定日期，如果time不传，则默认为当前时间
 **/
export function getBeforeDate(num, time) {
	let n = num;
	let d = '';
	if (time) {
		d = new Date(time);
	} else {
		d = new Date();
	}
	let year = d.getFullYear();
	let mon = d.getMonth() + 1;
	let day = d.getDate();
	if (day <= n) {
		if (mon > 1) {
			mon = mon - 1;
		} else {
			year = year - 1;
			mon = 12;
		}
	}
	d.setDate(d.getDate() - n);
	year = d.getFullYear();
	mon = d.getMonth() + 1;
	day = d.getDate();
	let s = year + "-" + (mon < 10 ? ('0' + mon) : mon) + "-" + (day < 10 ? ('0' + day) : day);
	return s;
}
/**
 * @description 获取年-月-日
 * @data {Any} 时间戳
 */
export function getDates(data) {
	let timeObj = {};
	data = new Date(data);
	let y = data.getFullYear();
	let m =
		data.getMonth() + 1 < 10 ?
		'0' + (data.getMonth() + 1) :
		data.getMonth() + 1;
	let d = data.getDate() < 10 ? '0' + data.getDate() : data.getDate();
	let w = data.getDay();
	switch (w) {
		case 1:
			w = '星期一';
			break;
		case 2:
			w = '星期二';
			break;
		case 3:
			w = '星期三';
			break;
		case 4:
			w = '星期四';
			break;
		case 5:
			w = '星期五';
			break;
		case 6:
			w = '星期六';
			break;
		case 7:
			w = '星期日';
			break;
	}
	let h = data.getHours() < 10 ? '0' + data.getHours() : data.getHours();
	let mi = data.getMinutes() < 10 ? '0' + data.getMinutes() : data.getMinutes();
	let s = data.getSeconds() < 10 ? '0' + data.getSeconds() : data.getSeconds();
	// 年月日 星期几 时分秒
	timeObj = {
		year: y + '',
		month: m + '',
		day: d + '',
		week: w + '',
		hour: h + '',
		minute: mi + '',
		second: s + ''
	};
	return timeObj;
}
 
 
/**
 *  @description 页面跳转
 */
export function urlTo(e) {
	uni.navigateTo({
		url: e
	})
}
 
/**
 * @description 页面跳转
 */
export function urltabTo(e) {
	uni.switchTab({
		url: e
	})
}
 
/**
 * @description 消息提示框 
 * @isback 为true时返回上级页面
 */
export function toast(msg = '', isback) {
	uni.showToast({
		title: msg,
		duration: 2000,
		icon: 'none'
	});
	if (isback) {
		setTimeout(function() {
			uni.navigateBack()
		}, 1000)
	}
}
 
/**
 * @description 弹出提示信息结束后执行方法
 */
export function showMsg(msg, duration = 2000, callback) {
	uni.showToast({
		title: msg,
		icon: 'none',
		duration: duration,
		success: function() {
			setTimeout(callback, duration);
		}
	})
}
