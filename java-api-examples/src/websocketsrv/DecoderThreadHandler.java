/*
 * // Copyright 2022-2023 by zhaoming
 */
// java websocketServer
package websocketsrv;

import com.k2fsa.sherpa.onnx.OnlineRecognizer;
import com.k2fsa.sherpa.onnx.OnlineStream;
import java.nio.*;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
// thread use to process data from client.
import java.util.*;
import java.util.List;
import java.util.concurrent.*;
import org.java_websocket.WebSocket;
import org.java_websocket.drafts.Draft;
import org.java_websocket.framing.Framedata;

public class DecoderThreadHandler extends Thread {
  // total data list that wait for decoding
  private CopyOnWriteArrayList<ConnectionData> decoderList;

  private OnlineRecognizer rcgOjb = null; // recgnizer object

  // data list for this thread in parallel decoding
  private List<ConnectionData> connDataList = new ArrayList<ConnectionData>();

  private int parallelDecoderNum = 10; // parallel decoding number
  private int deocderTimeIdle = 10; // idle time when no job
  private int deocderTimeOut = 3000; // timeout data will be remove
  CopyOnWriteArrayList<OnlineStream>
      decodeActiveList; // active list that synchronized for decoder threads

  public DecoderThreadHandler(
      CopyOnWriteArrayList<ConnectionData> decoderList,
      CopyOnWriteArrayList<OnlineStream> decodeActiveList,
      OnlineRecognizer rcgOjb,
      int deocderTimeIdle,
      int parallelDecoderNum,
      int deocderTimeOut) {
    this.decoderList = decoderList;
    this.rcgOjb = rcgOjb;
    this.decodeActiveList = decodeActiveList;
    this.deocderTimeIdle = deocderTimeIdle;
    this.parallelDecoderNum = parallelDecoderNum;
    this.deocderTimeOut = deocderTimeOut;
  }

  public void run() {
    while (true) {
      try {
        // time idle  if there is no job

        Thread.sleep(deocderTimeIdle);
        // clear list for this threads
        connDataList.clear();
        if (rcgOjb == null) continue;

        // loop total decoder list
        for (int i = 0; i < decoderList.size(); i++) {
          ConnectionData connData = decoderList.get(i);
          OnlineStream stream = connData.getStream();
          // synchronized decoderList
          synchronized (decoderList) {
            // put to decoder list if 1.stream not in other threads and 2.stream is ready and 3.
            // size not > parallelDecoderNum
            if ((!decodeActiveList.contains(stream)
                && rcgOjb.isReady(stream)
                && connDataList.size() < parallelDecoderNum)) {
              // add to active list
              decodeActiveList.add(stream);
              // add to this thread list
              connDataList.add(connData);
              // change the handled time for this data
              connData.setLastHandleTime(LocalDateTime.now());
            }

            java.time.Duration duration =
                java.time.Duration.between(connData.getLastHandleTime(), LocalDateTime.now());
            // remove data if 1. data is done  and  stream not ready or 2  data is time out or
            // 3.connect is closed
            if ((connData.getEof() == true && !rcgOjb.isReady(stream))
                || duration.toMillis() > deocderTimeOut
                || !connData.getWebSocket().isOpen()) {
              decoderList.remove(connData);
              WebSocket webSocket = connData.getWebSocket();
              // delay close socket as data may still in processing
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
        }

        // if data list for this thread >0
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

          String txtResult;

          txtResult = rcgOjb.getResult(stream);

          // decode text in utf-8
          byte[] utf8Data = txtResult.getBytes(StandardCharsets.UTF_8);

          // result
          if (utf8Data.length > 0) {
            System.out.println("txtResult:" + new String(utf8Data));
            String jsonResult = "{\"text\":\"" + txtResult + "\"}";

            // create a TEXT Frame for send back json result
            Draft draft = webSocket.getDraft();
            List<Framedata> frames = null;
            frames = draft.createFrames(jsonResult, false);
            // send to client
            webSocket.sendFrame(frames);
          }
          // stream done in this loop, release for next time
          decodeActiveList.remove(stream);
        }

      } catch (Exception e) {
        e.printStackTrace();
      }
    }
  }
}
