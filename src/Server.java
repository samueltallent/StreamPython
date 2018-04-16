import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.ServerSocket;
import java.net.Socket;

public class Server {
    public static void main(String[] args) throws IOException {

        ServerSocket serverSocket = null;
        try {
            serverSocket = new ServerSocket(4444);
        } catch (IOException e) {
            System.err.println("Could not listen on port: 4444.");
            System.exit(1);
        }

        Socket clientSocket = null;
        try {
            clientSocket = serverSocket.accept();
            System.out.println("Accepting: " + clientSocket);
            Process p = Runtime.getRuntime().exec(new String[]{"python3", "lane-detector.py"});
            BufferedReader in1 = new BufferedReader(
                    new InputStreamReader(
                            p.getInputStream()));
            String inputLine1 = null;
            while ((inputLine1 = in1.readLine()) != null) {
                System.out.println("Input: " + inputLine1);
            }
        } catch (IOException e) {
            System.err.println("Accept failed.");
            System.exit(1);
        }
    }
}
