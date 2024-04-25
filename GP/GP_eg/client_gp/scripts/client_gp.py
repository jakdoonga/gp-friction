#! /usr/bin/env python
import rospy
import numpy as np
import socket
import struct
from msg_pkg.msg import actual
from msg_pkg.msg import tau_fric

class ROS_SGP_Client:
    def __init__(self, PORT):
        # Set PORT
        self.PORT = PORT
        # Socket creation
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        # Connection
        self.sock.connect(('localhost', PORT))

        self.tau_motor = np.zeros((3,1))
        self.omega = np.zeros((3,1))
        self.tau_fric = np.zeros((3,1))

        self.subscriber = rospy.Subscriber("/actual", actual,self.callback,queue_size=1)
        self.publisher = rospy.Publisher("/tau_fric",tau_fric,queue_size=1)

        self.tau_fric_msg = tau_fric()

    def combine_byte_array(self):
        for i in range(3):
            if i == 0:
                byte_array = self.pack_byte_array(self.tau_motor[i])
                continue
            byte_array = byte_array + self.pack_byte_array(self.tau_motor[i])
        
        for i in range(3):
            byte_array = byte_array + self.pack_byte_array(self.omega[i])
        return byte_array

    def send_all_data(self):
        self.sock.send(self.combine_byte_array())

    def pack_byte_array(self, data):
        byte_array = bytearray(struct.pack('d',data))
        return byte_array

    def byte2output_data(self):
        data_recv = self.sock.recv(100)
        print(len(data_recv))
        for i in range(3):
            self.tau_fric[i] = np.float64(struct.unpack('d',data_recv[8*i:8*(i+1)]))

    def callback(self, msg):
        for i in range(3):
            self.tau_motor[i] = msg.act_LIFT_torque[i]
            self.omega[i] = msg.act_LIFT_vel[i]

        self.send_all_data()
        self.byte2output_data()
        self.publish_tau_fric()
    
    def connection_close(self):
        self.sock.close()

    def publish_tau_fric(self):
        now = rospy.get_rostime()
        self.tau_fric_msg.stamp.secs = now.secs
        self.tau_fric_msg.stamp.nsecs = now.nsecs
        for i in range(3):
            self.tau_fric_msg.tau_fric[i] = self.tau_fric[i]
        self.publisher.publish(self.tau_fric_msg)

if __name__=='__main__':
    rospy.init_node("SGP_client",anonymous=True)
    ros_SGP_client = ROS_SGP_Client(4000)
    r = rospy.Rate(50)
    while not rospy.is_shutdown():
        r.sleep()

    ros_SGP_client.connection_close()
