/*
 * // Copyright 2022-2023 by zhaoming
 */
// connection data act as a bridge between different threads pools

package websocketsrv;

import com.k2fsa.sherpa.onnx.OnlineStream;
import java.time.LocalDateTime;
// ClientData  help exchange data between WebsocketSrv and StreamHandler and DecoderHandler
import java.util.concurrent.*;
import org.java_websocket.WebSocket;

public class ConnectionData {

  private WebSocket webSocket; // the websocket for this client

  private OnlineStream stream; // connection stream

  private float[] samples; // binary data rec from the client

  private String msg; // for text

  private int type; // 1 binary, 2 text

  private boolean eof = false; // connection data is done

  private LocalDateTime lastHandleTime; // used for time out

  public ConnectionData(
      WebSocket webSocket, OnlineStream stream, float[] samples, String msg, int type) {
    this.webSocket = webSocket;
    this.samples = samples;
    this.stream = stream;
    this.msg = msg;
    this.type = type;
    this.lastHandleTime = LocalDateTime.now();
  }

  public LocalDateTime getLastHandleTime() {
    return this.lastHandleTime;
  }

  public void setLastHandleTime(LocalDateTime now) {
    this.lastHandleTime = now;
  }

  public String getMsg() {
    return this.msg;
  }

  public boolean getEof() {
    return this.eof;
  }

  public void setEof(boolean eof) {
    this.eof = eof;
  }

  public int getType() {
    return this.type;
  }

  public WebSocket getWebSocket() {
    return this.webSocket;
  }

  public float[] getSamples() {
    return this.samples;
  }

  public OnlineStream getStream() {
    return this.stream;
  }
}
