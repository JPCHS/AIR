package io.github.et.tools;

import java.io.*;
import java.net.*;
import java.nio.charset.StandardCharsets;

public class Client {
    private static Socket socket;
    private static OutputStream outputStream;
    private static InputStream inputStream;
    private static PrintWriter writer;
    private static BufferedReader reader;


    public static int getReply(String path) throws Exception {
        try {
            BufferedReader portReader = new BufferedReader(new FileReader("./port.txt"));
            int port = Integer.parseInt(portReader.readLine().trim());
            portReader.close();
            socket = new Socket("127.0.0.1", port);
            outputStream = socket.getOutputStream();
            writer = new PrintWriter(new OutputStreamWriter(outputStream, StandardCharsets.UTF_8), true);
            inputStream = socket.getInputStream();
            reader = new BufferedReader(new InputStreamReader(inputStream, StandardCharsets.UTF_8));
        } catch (IOException e) {
            e.printStackTrace();
            throw new RuntimeException(e);
        }
        writer.println(path);
        if (path.equals("Exit")) {
            System.exit(0);
        }
        String response = reader.readLine();
        if (response == null) {
            throw new NullPointerException("Received null response from server.");
        }

        switch (response) {
            case "0":
                return 0;
            case "1":
                return 1;
            case "2":
                return 2;
            default:
                throw new Exception("Unknown response from server.");
        }
    }
}
