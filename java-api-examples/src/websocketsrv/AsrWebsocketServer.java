/*
 * // Copyright 2022-2023 by zhaoming
 */
// java websocketServer
// usage: AsrWebsocketServer soPath modelCfgPath
package websocketsrv;

import com.k2fsa.sherpa.onnx.OnlineRecognizer;
import com.k2fsa.sherpa.onnx.OnlineStream;
import java.io.*;
import java.io.IOException;
import java.net.InetSocketAddress;
import java.net.UnknownHostException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.*;
import java.util.Collections;
import java.util.concurrent.*;
import java.util.concurrent.LinkedBlockingQueue;
import org.java_websocket.WebSocket;
import org.java_websocket.drafts.Draft;
import org.java_websocket.drafts.Draft_6455;
import org.java_websocket.handshake.ClientHandshake;
import org.java_websocket.server.WebSocketServer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * AsrWebSocketServer has three threads pools, one pool for network io, one pool for asr stream and
 * one pool for asr decoder.
 */
public class AsrWebsocketServer extends WebSocketServer {
  private static final Logger logger = LoggerFactory.getLogger(AsrWebsocketServer.class);
  //  Queue between io network io thread pool and stream thread pool, use websocket as the key
  private LinkedBlockingQueue<WebSocket> streamQueue = new LinkedBlockingQueue<WebSocket>();
  //  Queue waiting for deocdeing, use websocket as the key
  private LinkedBlockingQueue<WebSocket> decoderQueue = new LinkedBlockingQueue<WebSocket>();

  // recogizer object
  private OnlineRecognizer rcgOjb = null;

  // mapping between websocket connection and connection data
  private ConcurrentHashMap<WebSocket, ConnectionData> connectionMap =
      new ConcurrentHashMap<WebSocket, ConnectionData>();

  public AsrWebsocketServer(int port, int numThread) throws UnknownHostException {
    // server port and num of threads for  network io
    super(new InetSocketAddress(port), numThread);
  }

  public AsrWebsocketServer(InetSocketAddress address) {
    super(address);
  }

  public AsrWebsocketServer(int port, Draft_6455 draft) {
    super(new InetSocketAddress(port), Collections.<Draft>singletonList(draft));
  }

  @Override
  public void onOpen(WebSocket conn, ClientHandshake handshake) {}

  @Override
  public void onClose(WebSocket conn, int code, String reason, boolean remote) {
    connectionMap.remove(conn);
    logger.info(
        conn
            + " remove one connection!, now connection number="
            + String.valueOf(connectionMap.size()));
  }

  @Override
  public void onMessage(WebSocket conn, String message) {
    // this is text message
    try {
      // if rec "Done" msg from client
      if (message.equals("Done")) {
        ConnectionData connData = creatOrGetConnectionData(conn);
        connData.setEof(true);
        if (!streamQueueFind(conn)) {
          streamQueue.put(conn);
        }
      }

    } catch (Exception e) {
      e.printStackTrace();
    }
  }

  private ConnectionData creatOrGetConnectionData(WebSocket conn) {
    // create a new connection data if not in connection map or return the existed one

    ConnectionData connData = null;
    try {
      if (!connectionMap.containsKey(conn)) {
        OnlineStream stream = rcgOjb.createStream();
        connData = new ConnectionData(conn, stream);
        connectionMap.put(conn, connData);
      } else {
        connData = connectionMap.get(conn);
      }

      logger.info(
          conn.getRemoteSocketAddress().getAddress().getHostAddress()
              + " open one connection,, now connection number="
              + String.valueOf(connectionMap.size()));

    } catch (Exception e) {
      System.err.println(e);
      e.printStackTrace();
    }
    return connData;
  }

  @Override
  public void onMessage(WebSocket conn, ByteBuffer blob) {
    try {

      // for handle binary data
      blob.order(ByteOrder.LITTLE_ENDIAN); // set little endian

      // set to float
      FloatBuffer floatbuf = blob.asFloatBuffer();

      if (floatbuf.capacity() > 0) {
        // allocate memory for float data
        float[] arr = new float[floatbuf.capacity()];

        floatbuf.get(arr);
        ConnectionData connData = creatOrGetConnectionData(conn);
        // put websocket  to stream queue with binary type==1
        connData.addSamplesToData(arr);

        if (!streamQueueFind(conn)) {
          streamQueue.put(conn);
        }
      }
    } catch (Exception e) {
      e.printStackTrace();
    }
  }

 

  public boolean streamQueueFind(WebSocket conn) {
    return streamQueue.contains(conn);
  }

  public void initModelWithCfg(Map<String, String> cfgMap, String cfgPath) {
    try {

      rcgOjb = new OnlineRecognizer(cfgPath);
      // size of stream thread pool
      int streamThreadNum = Integer.valueOf(cfgMap.get("stream_thread_num"));
      // size of decoder thread pool
      int decoderThreadNum = Integer.valueOf(cfgMap.get("decoder_thread_num"));

      // time(ms) idle for decoder thread when no job
      int decoderTimeIdle = Integer.valueOf(cfgMap.get("decoder_time_idle"));
      // size of streams for parallel decoding
      int parallelDecoderNum = Integer.valueOf(cfgMap.get("parallel_decoder_num"));
      // time(ms) out for connection data
      int deocderTimeOut = Integer.valueOf(cfgMap.get("deocder_time_out"));

      // create stream threads
      for (int i = 0; i < streamThreadNum; i++) {
        new StreamThreadHandler(streamQueue, decoderQueue, connectionMap).start();
      }
      // create decoder threads
      for (int i = 0; i < decoderThreadNum; i++) {
        new DecoderThreadHandler(
                decoderQueue,
                connectionMap,
                rcgOjb,
                decoderTimeIdle,
                parallelDecoderNum,
                deocderTimeOut)
            .start();
      }
    } catch (Exception e) {
      System.err.println(e);
      e.printStackTrace();
    }
  }

  public static Map<String, String> readProperties(String CfgPath) {
    // read and parse config file
    Properties props = new Properties();
    Map<String, String> proMap = new HashMap<String, String>();
    try {

      File file = new File(CfgPath);
      if (!file.exists()) {
        logger.info(String.valueOf(CfgPath) + " cfg file not exists!");
        System.exit(0);
      }
      InputStream in = new BufferedInputStream(new FileInputStream(CfgPath));
      props.load(in);
      Enumeration en = props.propertyNames();
      while (en.hasMoreElements()) {
        String key = (String) en.nextElement();
        String Property = props.getProperty(key);
        proMap.put(key, Property);
      }

    } catch (Exception e) {
      e.printStackTrace();
    }
    return proMap;
  }

  public static void main(String[] args) throws InterruptedException, IOException {
    if (args.length != 2) {
      logger.info("usage: AsrWebsocketServer soPath modelCfgPath");

      return;
    }

    String soPath = args[0];
    String cfgPath = args[1];
 
    OnlineRecognizer.setSoPath(soPath);
    logger.info("readProperties");
    Map<String, String> cfgMap = AsrWebsocketServer.readProperties(cfgPath);
    int port = Integer.valueOf(cfgMap.get("port"));

    int connectionThreadNum = Integer.valueOf(cfgMap.get("connection_thread_num"));
    AsrWebsocketServer s = new AsrWebsocketServer(port, connectionThreadNum);
    logger.info("initModelWithCfg");
    s.initModelWithCfg(cfgMap, cfgPath);
    logger.info("Server started on port: " + s.getPort());
    s.start();
  }

  @Override
  public void onError(WebSocket conn, Exception ex) {
    ex.printStackTrace();
    if (conn != null) {
      // some errors like port binding failed may not be assignable to a specific websocket
    }
  }

  @Override
  public void onStart() {
    logger.info("Server started!");
    setConnectionLostTimeout(0);
    setConnectionLostTimeout(100);
  }
}
