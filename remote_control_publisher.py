#!/usr/bin/env python
import glob
import os
import sys
import json
import math
import random
import socket
import datetime
import time

from remote_control_message import Messge

def game_loop():

    # ==============================================================================================
    # intiallize socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)  
    server_ip = '192.168.5.4'  # 替换为接收端的实际 IP 地址
    server_port = 12345
    sock.connect((server_ip, server_port))
    # ==============================================================================================

    try:

        idx = 0
        while idx < 30:
            idx += 1
            time_start = time.time()

            # ==============================================================================================
            # socket publish
            message = Messge()
            packed_message  = message.pack()
            sock.sendall(packed_message)
            time.sleep(1.0)
            # ==============================================================================================

            



            current_time = datetime.datetime.now()
            time_str = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")
            time_end = time.time()
            print("message:",message, '   time cost:', round(time_end-time_start,3))   

        


    finally:
        # close socket
        sock.close()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================
def main():
    try:
        game_loop()

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
