/*
 * // Copyright 2022-2023 by zhaomingwork
 */
// java AsrWebsocketClient
// usage: AsrWebsocketClient soPath srvIp srvPort wavPath numThreads
package websocketsrv;

import com.k2fsa.sherpa.onnx.OnlineRecognizer;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.*;
import java.util.Map;
import org.java_websocket.client.WebSocketClient;
import org.java_websocket.drafts.Draft;
import org.java_websocket.handshake.ServerHandshake;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** This example demonstrates how to connect to websocket server. */
public class AsrWebsocketClient extends WebSocketClient {
  private static final Logger logger = LoggerFactory.getLogger(AsrWebsocketClient.class);

  public AsrWebsocketClient(URI serverUri, Draft draft) {
    super(serverUri, draft);
  }

  public AsrWebsocketClient(URI serverURI) {
    super(serverURI);
  }

  public AsrWebsocketClient(URI serverUri, Map<String, String> httpHeaders) {
    super(serverUri, httpHeaders);
  }

  @Override
  public void onOpen(ServerHandshake handshakedata) {

    float[] floats = OnlineRecognizer.readWavFile(AsrWebsocketClient.wavPath);
    ByteBuffer buffer =
        ByteBuffer.allocate(4 * floats.length)
            .order(ByteOrder.LITTLE_ENDIAN); // float is sizeof 4. allocate enough buffer

    for (float f : floats) {
      buffer.putFloat(f);
    }
    buffer.rewind();
    buffer.flip();
    buffer.order(ByteOrder.LITTLE_ENDIAN);

    send(buffer.array()); // send buf to server
    send("Done"); // send 'Done' means finished
  }

  @Override
  public void onMessage(String message) {

    logger.info("received: " + message);
  }

  @Override
  public void onClose(int code, String reason, boolean remote) {

    logger.info(
        "Connection closed by "
            + (remote ? "remote peer" : "us")
            + " Code: "
            + code
            + " Reason: "
            + reason);
  }

  @Override
  public void onError(Exception ex) {
    ex.printStackTrace();
    // if the error is fatal then onClose will be called additionally
  }

  public static OnlineRecognizer rcgobj;
  public static String wavPath;

  public static void main(String[] args) throws URISyntaxException {

    if (args.length != 5) {
      System.out.println("usage: AsrWebsocketClient soPath srvIp srvPort wavPath numThreads");
      return;
    }

    String soPath = args[0];
    String srvIp = args[1];
    String srvPort = args[2];
    String wavPath = args[3];
    int numThreads = Integer.parseInt(args[4]);
    System.out.println("serIp=" + srvIp + ",srvPort=" + srvPort + ",wavPath=" + wavPath);

    class ClientThread implements Runnable {

      String soPath;
      String srvIp;
      String srvPort;
      String wavPath;

      ClientThread(String soPath, String srvIp, String srvPort, String wavPath) {
        this.soPath = soPath;
        this.srvIp = srvIp;
        this.srvPort = srvPort;
        this.wavPath = wavPath;
      }

      public void run() {
        try {

          OnlineRecognizer.setSoPath(soPath);

          AsrWebsocketClient.wavPath = wavPath;

          String wsAddress = "ws://" + srvIp + ":" + srvPort;
          AsrWebsocketClient c = new AsrWebsocketClient(new URI(wsAddress));

          c.connect();
        } catch (Exception e) {
          e.printStackTrace();
        }
      }
    }
    for (int i = 0; i < numThreads; i++) {
      System.out.println("Thread1 is running...");
      Thread t = new Thread(new ClientThread(soPath, srvIp, srvPort, wavPath));
      t.start();
    }
  }
}
