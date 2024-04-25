import os
import socket
import numpy as np
import struct
import time

from typing import Callable, Tuple
import numpy as np
import tensorflow as tf
import gpflow
import time


class SGPserver:
    def __init__(self, saved_dir, PORT = 4000):

        # Get the saved directory
        self.saved_dir = saved_dir
        print(self.saved_dir)


        # GPU memory limit setup
        self.gpu_setup()

        # Load the saved model
        self.loaded_model = self.load_saved_model()
        # Sets PORT
        self.PORT = PORT

        self.conn, self.addr = self.socket_handler()

        self.tau_motor = np.zeros((3,1))
        self.omega = np.zeros((3,1))
        self.tau_fric = np.zeros((3,1))
        self.tau_fric_var = np.zeros((3,1))

        self.X_vstack = np.zeros((2,3))

    def gpu_setup(self):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            # Restrict tf to only allocate 1 GB of memory on the first GPU
            try:
                tf.config.set_logical_device_configuration(
                    gpus[0],
                    [tf.config.LogicalDeviceConfiguration(memory_limit = 1024)])
                logical_gpus = tf.config.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                print(e)

    def load_saved_model(self):
        saved_model = tf.saved_model.load(self.saved_dir)
        return saved_model

    def socket_handler(self):
        # Socket creation
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        print("Socket created")

        # Socket bind
        sock.bind(('localhost', self.PORT))
        print("Socket bind complete")

        # Socket listen
        sock.listen()
        print("Socket listen")

        # Socket accept
        conn, addr = sock.accept()
        print("Connected by", addr)

        return conn, addr

    def SGPR_run(self):
        while True:
            input_data = self.conn.recv(100)
            if input_data:
                self.get_input(input_data)
                self.get_tau_fric()
                self.send_tau_fric()
        self.conn.close()


    def get_input(self, input_data):
        """
        Gets input data from the client and convert it to numpy float64 format.
        :param input_data: byte array of tau_motor and omega
        :return: tau_motor, omega
        """
        for i in range(3):
            self.tau_motor[i] = np.float64(struct.unpack('d', input_data[8*i:8*(i+1)]))
        
        offset = 24
        for i in range(3):
            self.omega[i] = np.float64(struct.unpack('d', input_data[8*i + offset:8*(i+1)+offset]))
        

    def get_tau_fric(self):
        """
        From the saved SGPR model, get the mean and variance of the tau_fric, respectively.
        :param tau_motor:
        :param oemga:
        :return: tau_fric_mean, tau_fric_var
        """

        X_vstack = np.vstack((self.tau_motor.flatten(), self.omega.flatten())).T

        print('-'*30)
        print('Input matrix')
        print(X_vstack)
        print('-'*30)
        for i in range(3):
            self.tau_fric[i], self.tau_fric_var[i] = \
            self.loaded_model.compiled_predict_f(X_vstack[i,:].reshape(1,2))

    def send_tau_fric(self):
        """
        Converts into byte array and sends it to the client
        :param tau_fric_mean:
        :param tau_fric_var:
        :return:
        """

        print('-'*30)
        print('Friction')
        print(self.tau_fric)
        print('-'*30)

        for i in range(3):
            if i == 0:
                byte_array = bytearray(struct.pack('d',self.tau_fric[i]))
                continue
            byte_array =  byte_array + bytearray(struct.pack('d',self.tau_fric[i]))
        self.conn.sendall(byte_array)


if __name__ == "__main__":
    current_dir = os.getcwd()
    saved_dir = current_dir + "/saved_model"
    SGP_server = SGPserver(saved_dir)
    SGP_server.SGPR_run()
