/*
 * // Copyright 2022-2023 by zhaoming
 */
// java DecoderThreadHandler
package websocketsrv;

import com.k2fsa.sherpa.onnx.OnlineRecognizer;
import com.k2fsa.sherpa.onnx.OnlineStream;
import java.nio.*;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.util.*;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.LinkedBlockingQueue;
import org.java_websocket.WebSocket;
import org.java_websocket.drafts.Draft;
import org.java_websocket.framing.Framedata;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DecoderThreadHandler extends Thread {
  private static final Logger logger = LoggerFactory.getLogger(DecoderThreadHandler.class);
  // Websocket Queue that waiting for decoding
  private LinkedBlockingQueue<WebSocket> decoderQueue;
  // the mapping between websocket and connection data
  private ConcurrentHashMap<WebSocket, ConnectionData> connMap;

  private OnlineRecognizer rcgOjb = null; // recgnizer object

  // connection data list for this thread to decode in parallel
  private List<ConnectionData> connDataList = new ArrayList<ConnectionData>();

  private int parallelDecoderNum = 10; // parallel decoding number
  private int deocderTimeIdle = 10; // idle time(ms) when no job
  private int deocderTimeOut = 3000; // if it is timeout(ms), the connection data will be removed

  public DecoderThreadHandler(
      LinkedBlockingQueue<WebSocket> decoderQueue,
      ConcurrentHashMap<WebSocket, ConnectionData> connMap,
      OnlineRecognizer rcgOjb,
      int deocderTimeIdle,
      int parallelDecoderNum,
      int deocderTimeOut) {
    this.decoderQueue = decoderQueue;
    this.connMap = connMap;
    this.rcgOjb = rcgOjb;
    this.deocderTimeIdle = deocderTimeIdle;
    this.parallelDecoderNum = parallelDecoderNum;
    this.deocderTimeOut = deocderTimeOut;
  }

  public void run() {
    while (true) {
      try {
        // time(ms) idle  if there is no job

        Thread.sleep(deocderTimeIdle);
        // clear data list for this threads
        connDataList.clear();
        if (rcgOjb == null) continue;

        // loop for total decoder Queue
        while (!decoderQueue.isEmpty()) {

          // get websocket
          WebSocket conn = decoderQueue.take();
          // get connection data according to websocket
          ConnectionData connData = connMap.get(conn);

          // if the websocket closed, continue
          if (connData == null) continue;
          // get the stream
          OnlineStream stream = connData.getStream();

          // put to decoder list if 1) stream is ready; 2) and
          // size not > parallelDecoderNum
          if ((rcgOjb.isReady(stream) && connDataList.size() < parallelDecoderNum)) {

            // add to this thread's decoder list
            connDataList.add(connData);
            // change the handled time for this connection data
            connData.setLastHandleTime(LocalDateTime.now());
          }
          // break when decoder list size >= parallelDecoderNum
          if (connDataList.size() >= parallelDecoderNum) {
            break;
          }
        }

        // if decoder data list for this thread >0
        if (connDataList.size() > 0) {

          // create a stream array for parallel decoding
          OnlineStream[] arr = new OnlineStream[connDataList.size()];
          for (int i = 0; i < connDataList.size(); i++) {

            arr[i] = connDataList.get(i).getStream();
          }

          // parallel decoding
          rcgOjb.decodeStreams(arr);
        }

        // get result for each connection
        for (ConnectionData connData : connDataList) {

          OnlineStream stream = connData.getStream();
          WebSocket webSocket = connData.getWebSocket();

          String txtResult = rcgOjb.getResult(stream);

          // decode text in utf-8
          byte[] utf8Data = txtResult.getBytes(StandardCharsets.UTF_8);

          boolean isEof = (connData.getEof() == true && !rcgOjb.isReady(stream));
          // result
          if (utf8Data.length > 0) {

            String jsonResult =
                "{\"text\":\"" + txtResult + "\",\"eof\":" + String.valueOf(isEof) + "\"}";

            if (webSocket.isOpen()) {
              // create a TEXT Frame for send back json result
              Draft draft = webSocket.getDraft();
              List<Framedata> frames = null;
              frames = draft.createFrames(jsonResult, false);
              // send to client
              webSocket.sendFrame(frames);
            }
          }
        }
        // loop for each connection data in this thread
        for (ConnectionData connData : connDataList) {
          OnlineStream stream = connData.getStream();
          WebSocket webSocket = connData.getWebSocket();
          // if the stream is still ready, put it to decoder Queue again for next decoding
          if (rcgOjb.isReady(stream)) {
            decoderQueue.put(webSocket);
          }
          // the duration between last handled time and now
          java.time.Duration duration =
              java.time.Duration.between(connData.getLastHandleTime(), LocalDateTime.now());
          // close the websocket if 1) data is done  and  stream not ready; 2) or data is time out;
          // 3) or
          // connection is closed
          if ((connData.getEof() == true
                  && !rcgOjb.isReady(stream)
                  && connData.getQueueSamples().isEmpty())
              || duration.toMillis() > deocderTimeOut
              || !connData.getWebSocket().isOpen()) {

            logger.info("close websocket!!!");

            // delay close web socket as data may still in processing
            Timer timer = new Timer();
            timer.schedule(
                new TimerTask() {
                  public void run() {

                    webSocket.close();
                  }
                },
                5000); // 5 seconds
          }
        }

      } catch (Exception e) {
        e.printStackTrace();
      }
    }
  }
}
