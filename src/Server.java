import java.io.*;
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
                System.out.println("Input: " + inputLine1.substring(2, inputLine1.length() - 1)); //inputLine1.substring(2, inputLine1.length() - 1)) => to client to be decoded and displayed
                PrintWriter cOut = new PrintWriter(clientSocket.getOutputStream());
                cOut.println(inputLine1.substring(2, inputLine1.length() - 1));
                /* This section demonstrates how to decode the image from the above string, this code should be moved to the android client.
                BufferedImage img = null;
                byte[] imageByte;
                imageByte = Base64.getDecoder().decode(inputLine1.substring(2, inputLine1.length() - 1));
                img = ImageIO.read(new ByteArrayInputStream(imageByte));
                 */
            }
        } catch (IOException e) {
            System.err.println("Accept failed.");
            System.exit(1);
        }
    }
}
