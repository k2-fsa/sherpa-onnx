/*
 * // Copyright 2022-2023 by zhaoming
 */
// java AsrWebsocketClient
package websocketsrv;

import com.k2fsa.sherpa.onnx.OnlineRecognizer;
import java.net.URI;
import java.net.URISyntaxException;
import java.nio.*;
import java.util.Map;
import org.java_websocket.client.WebSocketClient;
import org.java_websocket.drafts.Draft;
import org.java_websocket.handshake.ServerHandshake;

/** This example demonstrates how to connect to websocket server. */
public class AsrWebsocketClient extends WebSocketClient {

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
            .order(ByteOrder.LITTLE_ENDIAN); // allocate byte buffer

    for (float f : floats) {
      buffer.putFloat(f);
    }
    buffer.rewind();
    buffer.flip();
    buffer.order(ByteOrder.LITTLE_ENDIAN);

    send(buffer.array()); // send buf to server
    send("Done"); // send 'Done' means finished

    // if you plan to refuse connection based on ip or httpfields overload:
    // onWebsocketHandshakeReceivedAsClient
  }

  @Override
  public void onMessage(String message) {
    System.out.println("received: " + message);
  }

  @Override
  public void onClose(int code, String reason, boolean remote) {

    System.out.println(
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

    if (args.length != 4) {
      System.out.println("usage: AsrWebsocketClient soPath srvIp srvPort wavPath");
      return;
    }

    String soPath = args[0]; // appDir + "/../build/lib/libsherpa-onnx-jni.so";
    String srvIp = args[1];
    String srvPort = args[2];
    String wavPath = args[3];
    System.out.println("serIp=" + srvIp + ",srvPort=" + srvPort + ",wavPath=" + wavPath);
    OnlineRecognizer.setSoPath(soPath);

    AsrWebsocketClient.wavPath =
        wavPath; // "/sherpa/forgithub/sherpa-onnx/java-api-examples/test.wav";

    String wsAddress = "ws://" + srvIp + ":" + srvPort;
    AsrWebsocketClient c =
        new AsrWebsocketClient(
            new URI(
                wsAddress)); // more about drafts here:
                             // http://github.com/TooTallNate/Java-WebSocket/wiki/Drafts
    c.connect();
  }
}
