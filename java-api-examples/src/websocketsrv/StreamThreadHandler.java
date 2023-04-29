/*
 * // Copyright 2022-2023 by zhaoming
 */
// java StreamThreadHandler
package websocketsrv;

import com.k2fsa.sherpa.onnx.OnlineStream;
import java.nio.*;
import java.util.*;
import java.util.concurrent.*;

// thread for processing stream
public class StreamThreadHandler extends Thread {
  // Queue for all connection data
  private ConcurrentLinkedQueue<ConnectionData> clientQueue;
  // List that sent to decoder thread for decoding
  private CopyOnWriteArrayList<ConnectionData> decoderList;
  // time(ms) idle for threads
  private int timeIdle = 10;

  public StreamThreadHandler(
      ConcurrentLinkedQueue<ConnectionData> clientQueue,
      CopyOnWriteArrayList<ConnectionData> decoderList,
      int timeIdle) {
    this.clientQueue = clientQueue;
    this.decoderList = decoderList;
    this.timeIdle = timeIdle;
  }

  public void run() {
    while (true) {
      try {
        // fetch one data from queue
        ConnectionData dataObj = (ConnectionData) this.clientQueue.poll();

        // time idle if there is no job
        if (dataObj == null) {
          Thread.sleep(timeIdle);
          continue;
        }

        OnlineStream stream = dataObj.getStream();
        // handle  stream sequentially
        synchronized (stream) {

          // type==2 means TEXT, recevived msg "Done" from client means finished
          if (dataObj.getType() == 2 && dataObj.getMsg().equals("Done")) {

            stream.inputFinished();
            dataObj.setEof(true); // set end for this connection
          }
          float[] samples = dataObj.getSamples();
          //  type==1 means binary,recevice binary data from client
          if (dataObj.getType() == 1 && samples != null) {

            // feed data for asr stream
            stream.acceptWaveform(samples);

            // add this data to decoder list, and wait for decoding
            decoderList.add(dataObj);
          }
        }

      } catch (Exception e) {
        e.printStackTrace();
      }
    }
  }
}
