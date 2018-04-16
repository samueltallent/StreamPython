import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.net.InetAddress;
import java.net.Socket;
import java.net.UnknownHostException;

public class Client {
    public static void main(String[] args) {
        String hostAddress = "localhost";
        InetAddress addr = null;
        int tcpPort = 4444;// hardcoded -- must match the server's tcp port

        try {
            addr = InetAddress.getByName(hostAddress);
            Socket tcp = new Socket(addr, tcpPort);
            PrintWriter out = new PrintWriter(tcp.getOutputStream(),true);
            BufferedReader in = new BufferedReader(new InputStreamReader(tcp.getInputStream()));
        } catch (UnknownHostException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

    }
}
