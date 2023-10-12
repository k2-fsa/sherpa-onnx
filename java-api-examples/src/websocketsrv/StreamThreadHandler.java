/*
 * // Copyright 2022-2023 by zhaoming
 */
// java StreamThreadHandler
package websocketsrv;

import com.k2fsa.sherpa.onnx.OnlineStream;
import java.nio.*;
import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.LinkedBlockingQueue;
import org.java_websocket.WebSocket;
// thread for processing stream

public class StreamThreadHandler extends Thread {
  //  Queue between io network io thread pool and stream thread pool, use websocket as the key
  private LinkedBlockingQueue<WebSocket> streamQueue;
  //  Queue waiting for deocdeing, use websocket as the key
  private LinkedBlockingQueue<WebSocket> decoderQueue;
  // mapping between websocket connection and connection data
  private ConcurrentHashMap<WebSocket, ConnectionData> connMap;

  public StreamThreadHandler(
      LinkedBlockingQueue<WebSocket> streamQueue,
      LinkedBlockingQueue<WebSocket> decoderQueue,
      ConcurrentHashMap<WebSocket, ConnectionData> connMap) {
    this.streamQueue = streamQueue;
    this.decoderQueue = decoderQueue;
    this.connMap = connMap;
  }

  public void run() {
    while (true) {
      try {
        // fetch one websocket from queue
        WebSocket conn = (WebSocket) this.streamQueue.take();
        // get the connection data according to websocket
        ConnectionData connData = connMap.get(conn);
        OnlineStream stream = connData.getStream();

        // handle received binary data
        if (!connData.getQueueSamples().isEmpty()) {
          // loop to put all received binary data to stream
          while (!connData.getQueueSamples().isEmpty()) {

            float[] samples = connData.getQueueSamples().poll();

            stream.acceptWaveform(samples);
          }
          //  if data is finished
          if (connData.getEof() == true) {

            stream.inputFinished();
          }
          // add this websocket to decoder Queue if not in the Queue
          if (!decoderQueue.contains(conn)) {

            decoderQueue.put(conn);
          }
        }

      } catch (Exception e) {
        e.printStackTrace();
      }
    }
  }
}
