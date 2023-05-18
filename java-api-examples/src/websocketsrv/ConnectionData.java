/*
 * // Copyright 2022-2023 by zhaoming
 */
// connection data act as a bridge between different threads pools

package websocketsrv;

import com.k2fsa.sherpa.onnx.OnlineStream;
import java.time.LocalDateTime;
import java.util.LinkedList;
import java.util.Queue;
import java.util.concurrent.*;
import org.java_websocket.WebSocket;

public class ConnectionData {

  private WebSocket webSocket; // the websocket for this connection data

  private OnlineStream stream; // connection stream

  private Queue<float[]> queueSamples =
      new LinkedList<float[]>(); // binary data rec from the client

  private boolean eof = false; // connection data is done

  private LocalDateTime lastHandleTime; // used for time out in ms

  public ConnectionData(WebSocket webSocket, OnlineStream stream) {
    this.webSocket = webSocket;

    this.stream = stream;
  }

  public void addSamplesToData(float[] samples) {
    this.queueSamples.add(samples);
  }

  public LocalDateTime getLastHandleTime() {
    return this.lastHandleTime;
  }

  public void setLastHandleTime(LocalDateTime now) {
    this.lastHandleTime = now;
  }

  public boolean getEof() {
    return this.eof;
  }

  public void setEof(boolean eof) {
    this.eof = eof;
  }

  public WebSocket getWebSocket() {
    return this.webSocket;
  }

  public Queue<float[]> getQueueSamples() {
    return this.queueSamples;
  }

  public OnlineStream getStream() {
    return this.stream;
  }
}
