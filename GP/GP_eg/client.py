import socket
import numpy as np
import struct

class Client:
    def __init__(self, PORT):
        # Set PORT
        self.PORT = PORT
        # Socket creation
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Connection
        self.sock.connect(('localhost', PORT))

        self.data1 = np.float64(0)
        self.data2 = np.float64(0)

    def get_data(self):
        self.data1 = np.float64(input("Input data1: "))
        self.data2 = np.float64(input("Input data2: "))

    def combine_byte_array(self):
        byte_array1 = self.data2byte_array(self.data1)
        byte_array2 = self.data2byte_array(self.data2)
        return byte_array1 + byte_array2

    def send_all_data(self):
        self.sock.send(self.combine_byte_array())

    def data2byte_array(self, data):
        byte_array = bytearray(struct.pack('d',data))
        return byte_array

    def byte2output_data(self):
        data_recv = self.sock.recv(8)
        data_recv = np.float64(struct.unpack('d',data_recv))
        print(data_recv)


    def run_client(self):
        while True:
            self.get_data()
            self.send_all_data()
            self.byte2output_data()
        self.sock.close()


if __name__ == "__main__":
    client = Client(PORT=1029)
    client.run_client()