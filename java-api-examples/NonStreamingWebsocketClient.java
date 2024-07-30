// Refer to
// https://stackoverflow.com/questions/55380813/require-assistance-with-simple-pure-java-11-websocket-client-example
//
//
// This is a WebSocketClient client for ../python-api-examples/non_streaming_server.py
//
// Please see ./run-non-streaming-websocket-client.sh
import com.k2fsa.sherpa.onnx.*;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.WebSocket;
import java.nio.*;
import java.util.concurrent.CompletionStage;
import java.util.concurrent.CountDownLatch;

public class NonStreamingWebsocketClient {
  public static void main(String[] args) throws Exception {
    CountDownLatch latch = new CountDownLatch(1);

    WebSocket ws =
        HttpClient.newHttpClient()
            .newWebSocketBuilder()
            .buildAsync(URI.create("ws://localhost:6006"), new WebSocketClient(latch))
            .join();

    // Please use a 16-bit, single channel wav for testing.
    // the sample rate does not need to be 16kHz
    String waveFilename = "./zh.wav";
    WaveReader reader = new WaveReader(waveFilename);
    int sampleRate = reader.getSampleRate();
    int numSamples = reader.getSamples().length;

    // Here is the format of the message
    // byte 0-3 in little endian: sampleRate
    // byte 4-7 in little endian: number of bytes for samples
    // remaining bytes: samples. Each sample is a float32
    ByteBuffer buffer = ByteBuffer.allocate(8 + 4 * numSamples).order(ByteOrder.LITTLE_ENDIAN);
    buffer.putInt(sampleRate);
    buffer.putInt(numSamples * 4); // each sample has 4 bytes

    for (float s : reader.getSamples()) {
      buffer.putFloat(s);
    }

    buffer.rewind();
    buffer.flip();
    buffer.order(ByteOrder.LITTLE_ENDIAN);

    ws.sendBinary(ByteBuffer.wrap(buffer.array()), true).join();

    // Send Done to the server to indicate that we don't have new wave files to decode
    ws.sendText("Done", true).join();

    latch.await();
  }

  private static class WebSocketClient implements WebSocket.Listener {
    private final CountDownLatch latch;

    public WebSocketClient(CountDownLatch latch) {
      this.latch = latch;
    }

    @Override
    public CompletionStage<?> onText(WebSocket webSocket, CharSequence data, boolean last) {
      System.out.println("Result is " + data);
      latch.countDown();
      return WebSocket.Listener.super.onText(webSocket, data, last);
    }
  }
}
