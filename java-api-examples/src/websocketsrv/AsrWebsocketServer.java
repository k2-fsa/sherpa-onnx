/*
 * // Copyright 2022-2023 by zhaoming
 */
// java websocketServer
package websocketsrv;

import com.k2fsa.sherpa.onnx.OnlineRecognizer;
import com.k2fsa.sherpa.onnx.OnlineStream;
import java.io.*;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.InetSocketAddress;
import java.net.UnknownHostException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.*;
import java.util.Collections;
import java.util.concurrent.*;
import org.java_websocket.WebSocket;
import org.java_websocket.drafts.Draft;
import org.java_websocket.drafts.Draft_6455;
import org.java_websocket.handshake.ClientHandshake;
import org.java_websocket.server.WebSocketServer;

/**
 * AsrWebSocketServer with three threads pools, one pool for recevice data, one pool for asr stream
 * and one pool for asr decoder
 */
public class AsrWebsocketServer extends WebSocketServer {
  // data Queue  consumed by stream thread
  static ConcurrentLinkedQueue<ConnectionData> clientQueue =
      new ConcurrentLinkedQueue<ConnectionData>();
  // total data list wait for deocdeing
  static CopyOnWriteArrayList<ConnectionData> decoderList =
      new CopyOnWriteArrayList<ConnectionData>();
  // in active list means handling by other thread
  static CopyOnWriteArrayList<OnlineStream> decoderActiveList =
      new CopyOnWriteArrayList<OnlineStream>();
  // recogizer object
  private OnlineRecognizer rcgOjb = null;

  // each connect can only has one stream, this map help to do it
  static ConcurrentHashMap<WebSocket, OnlineStream> connectionMap =
      new ConcurrentHashMap<WebSocket, OnlineStream>();

  public AsrWebsocketServer(int port, int numThread) throws UnknownHostException {
    // server port and num of threads for network
    super(new InetSocketAddress(port), numThread);
  }

  public AsrWebsocketServer(InetSocketAddress address) {
    super(address);
  }

  public AsrWebsocketServer(int port, Draft_6455 draft) {
    super(new InetSocketAddress(port), Collections.<Draft>singletonList(draft));
  }

  @Override
  public void onOpen(WebSocket conn, ClientHandshake handshake) {

    System.out.println(
        conn.getRemoteSocketAddress().getAddress().getHostAddress()
            + " open one connection,, now connection number="
            + String.valueOf(connectionMap.size()));
  }

  @Override
  public void onClose(WebSocket conn, int code, String reason, boolean remote) {
    connectionMap.remove(conn);
    System.out.println(
        conn
            + " remove one connection!, now connection number="
            + String.valueOf(connectionMap.size()));
  }

  @Override
  public void onMessage(WebSocket conn, String message) {
    // create a new stream if the first time
    OnlineStream stream = creatOrGetStream(conn);
    // put this data to total data queue, its a text message type==2
    clientQueue.add(new ConnectionData(conn, stream, null, message, 2));
  }

  private OnlineStream creatOrGetStream(WebSocket conn) {
    // create a new stream if not contained or return the existed one
    OnlineStream stream = null;
    try {
      if (!connectionMap.containsKey(conn)) {
        stream = rcgOjb.createStream();
        connectionMap.put(conn, stream);
      } else {
        stream = connectionMap.get(conn);
      }

    } catch (Exception e) {
      System.err.println(e);
      e.printStackTrace();
    }
    return stream;
  }

  @Override
  public void onMessage(WebSocket conn, ByteBuffer blob) {
    // for handle binary data
    blob.order(ByteOrder.LITTLE_ENDIAN); // set little endian

    // set to float
    FloatBuffer floatbuf = blob.asFloatBuffer();

    if (floatbuf.capacity() > 0) {
      // allocate memory for float data
      float[] arr = new float[floatbuf.capacity()];

      floatbuf.get(arr);
      OnlineStream stream = creatOrGetStream(conn);
      // put data connect data to queue with binary type==1
      ConnectionData connObj = new ConnectionData(conn, stream, arr, null, 1);
      clientQueue.add(connObj);
    }
  }

  public void initModelWithCfg(Map<String, String> cfgMap, String cfgPath) {
    try {
      // you should set setCfgPath() before new the recognizer
      rcgOjb = new OnlineRecognizer(cfgPath);

      int streamThreadNum = Integer.valueOf(cfgMap.get("stream_thread_num"));
      int decoderThreadNum = Integer.valueOf(cfgMap.get("decoder_thread_num"));
      int streamTimeIdle = Integer.valueOf(cfgMap.get("stream_time_idle"));
      int decoderTimeIdle = Integer.valueOf(cfgMap.get("decoder_time_idle"));
      int parallelDecoderNum = Integer.valueOf(cfgMap.get("parallel_decoder_num"));
      int deocderTimeOut = Integer.valueOf(cfgMap.get("deocder_time_out"));

      // create stream threads
      for (int i = 0; i < streamThreadNum; i++) {
        new StreamThreadHandler(clientQueue, decoderList, streamTimeIdle).start();
      }
      // create decoder threads
      for (int i = 0; i < decoderThreadNum; i++) {
        new DecoderThreadHandler(
                decoderList,
                decoderActiveList,
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
        System.out.println(String.valueOf(CfgPath) + " cfg file not exists!");
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
      System.out.println("usage: AsrWebsocketSrv soPath modelCfgPath");

      return;
    }

    String soPath = args[0]; // appDir + "/../build/lib/libsherpa-onnx-jni.so";
    String cfgPath = args[1]; // appDir + "/modelconfig.cfg";

    OnlineRecognizer.setSoPath(soPath);

    Map<String, String> cfgMap = AsrWebsocketServer.readProperties(cfgPath);
    int port = Integer.valueOf(cfgMap.get("port"));

    int connectionThreadNum = Integer.valueOf(cfgMap.get("connection_thread_num"));
    AsrWebsocketServer s = new AsrWebsocketServer(port, connectionThreadNum);
    s.initModelWithCfg(cfgMap, cfgPath);
    System.out.println("Server started on port: " + s.getPort());
    s.start();

    // String in = sysin.readLine();
    BufferedReader sysin = new BufferedReader(new InputStreamReader(System.in));
    // String in = sysin.readLine();
    while (true) {
      String in = sysin.readLine();
      // s.broadcast(in);
      if (in.equals("exit")) {
        // s.stop(1000);
        break;
      }
    }
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
    System.out.println("Server started!");
    setConnectionLostTimeout(0);
    setConnectionLostTimeout(100);
  }
}
