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

class Messge:
    def __init__(self):
        self.steer = 0.0
        self.brake = 0.0
        self.throttle = 0.0


    def pack(self):
        return struct.pack('fff', self.steer, self.brake, self.throttle)