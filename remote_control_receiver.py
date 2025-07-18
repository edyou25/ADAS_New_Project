#!/usr/bin/env python
import glob
import os
import sys
import cv2
import random
import time
import numpy as np
import math
import socket
import struct



def run_simulation():
    try:

        # intialize socket
        socket_address = '0.0.0.0'
        socket_port = 12345
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        server_address = (socket_address, socket_port)  # 设置接收方的IP地址和端口号
        sock.bind(server_address)


        
        while True:
            time_start = time.time()
            data, address = sock.recvfrom(4096)
            message = struct.unpack('fff', data)
            # ====================================================================================
            time_end = time.time()
            print('message:', message,  ' time cost:', round(time_end-time_start,3) )

    finally:
        # close socket
        sock.close()


def main():
    try:
        run_simulation()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
