package com.k2fsa.sherpa.onnx

import org.java_websocket.client.WebSocketClient
import org.java_websocket.handshake.ServerHandshake
import java.net.URI

class MyWebsocketClient(serverUri: URI?) : WebSocketClient(serverUri) {

    override fun onOpen(handshakedata: ServerHandshake) {
        clientCallback?.onOpen(handshakedata)

    }
    override fun onMessage(message: String) {
        clientCallback?.onMessage(message)
    }

    override fun onClose(code: Int, reason: String, remote: Boolean) {
        clientCallback?.onClose(code,reason,remote)
    }

    override fun onError(ex: Exception) {
        clientCallback?.onError(ex)
    }

    private var clientCallback: WebsocketClientCallback? = null

    fun setClientCallback(clientCallback: WebsocketClientCallback?) {
        this.clientCallback = clientCallback
    }

    interface WebsocketClientCallback {
        fun onOpen(handshakedata: ServerHandshake?)
        fun onMessage(message: String?)
        fun onClose(code: Int, reason: String?, remote: Boolean?)
        fun onError(ex: Exception?)
    }


}