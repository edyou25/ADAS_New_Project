#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Script that render multiple sensors in the same pygame window

By default, it renders four cameras, one LiDAR and one Semantic LiDAR.
It can easily be configure for any different number of sensors. 
To do that, check lines 290-308.
"""
import glob
import os
import sys
import cv2
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import argparse
import random
import time
import numpy as np
from queue import Queue
import math
import socket
try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

def init_camera(world, sensor_type, transform, attached, image_with, image_hight, fov):
    if sensor_type == 'RGBCamera':
        camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(image_with))
        camera_bp.set_attribute('image_size_y', str(image_hight))
        camera_bp.set_attribute('fov', str(fov))
        camera_rgb = world.spawn_actor(camera_bp, transform, attach_to=attached)
        return camera_rgb
    elif sensor_type == 'SemanticCamera':
        camera_bp = world.get_blueprint_library().find('sensor.camera.semantic_segmentation')
        camera_bp.set_attribute('image_size_x', str(image_with))
        camera_bp.set_attribute('image_size_y', str(image_hight))
        camera_bp.set_attribute('fov', str(fov))
        camera_sematic = world.spawn_actor(camera_bp, transform, attach_to=attached)
        return camera_sematic
    elif sensor_type == 'OpticalFlowCamera':
        camera_bp = world.get_blueprint_library().find('sensor.camera.optical_flow')
        camera_bp.set_attribute('image_size_x', str(image_with))
        camera_bp.set_attribute('image_size_y', str(image_hight))
        camera_bp.set_attribute('fov', str(fov))
        camera_optical = world.spawn_actor(camera_bp, transform, attach_to=attached)
        return camera_optical
    else:
        return None

def process_rgb_image(image, image_rgb_queue, queue_len):
    image.convert(carla.ColorConverter.Raw)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    image_rgb_queue.append(array)
    if len(image_rgb_queue) > queue_len:
        image_rgb_queue.pop(0)

def process_semantic_image(image, image_sematic_queue, queue_len):
    # image.convert(carla.ColorConverter.Raw)
    image.convert(carla.ColorConverter.CityScapesPalette)
    array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    array = np.reshape(array, (image.height, image.width, 4))
    array = array[:, :, :3]
    array = array[:, :, ::-1]
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    image_sematic_queue.append(array)
    if len(image_sematic_queue) > queue_len:
        image_sematic_queue.pop(0)

def get_prediction(cur_tf, vel, steer_cur, horizon, dt):
    trajectory = []
    L = 4.0
    x0, y0, yaw0 = cur_tf.location.x, cur_tf.location.y, math.radians(cur_tf.rotation.yaw)
    # trajectory.append([x0, y0, yaw0, vel])
    trajectory.append([x0, y0, yaw0, vel])
    steer = steer_cur
    for idx in range(horizon):
        x1 = x0 + vel*math.cos(yaw0)*dt
        y1 = y0 + vel*math.sin(yaw0)*dt
        if idx == 4:
            steer = 0.5 * steer_cur
        if idx >= 6:
            steer = -2.0 * steer_cur
        # print('steer: ', steer)
        yaw1 = yaw0 + vel / L * math.tan(steer)*dt
        x0, y0, yaw0 = x1, y1, yaw1
        trajectory.append([x1, y1, yaw1, vel])
    return trajectory

def draw_trajectory(trajectory, ego_tf, world):
    idx = 3
    horizon = len(trajectory)
    while idx < horizon-1:
        x0 = trajectory[idx]
        x1 = trajectory[idx+1]
        begin = carla.Location(x=x0[0],y=x0[1],z=ego_tf.location.z+0.5)
        end = carla.Location(x=x1[0], y=x1[1], z=ego_tf.location.z+0.5)
        world.debug.draw_arrow(begin, end, thickness=0.035, arrow_size=0.7, color=carla.Color(0,60, 60),
                                    life_time=0.05)
        idx += 2

def draw_behavior(image, width, hight, behavior):
    color = (0, 255, 255, 1)  # 荧光黄色 (BGR格式)，透明度设置为 100
    thickness = 20  # 线宽
    if behavior == 'lane_keep' or behavior == 'end_action':
        start_point = (width // 2, hight // 5 * 4)
        end_point   = (width // 2, hight // 5 * 3)
        cv2.arrowedLine(image, start_point, end_point, color, thickness, tipLength=0.4)
        cv2.putText(image, 'lane keeping', (width//2-70, hight//5*4+50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 1)
    elif behavior == 'left_change' or behavior == 'lchange':
        start_point = (width // 5*2, hight // 5 * 4)
        end_point   = (width // 5*1, hight // 5 * 4)
        cv2.arrowedLine(image, start_point, end_point, color, thickness, tipLength=0.4)
        cv2.putText(image, 'left change', (width//2-70, hight//5*4+50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 1)
    elif behavior == 'right_change' or behavior == 'rchange':
        start_point = (width // 5*3, hight // 5 * 4)
        end_point   = (width // 5*4, hight // 5 * 4)
        cv2.arrowedLine(image, start_point, end_point, color, thickness, tipLength=0.4)
        cv2.putText(image, 'right change', (width//2-70, hight//5*4+50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 1)
    elif behavior == 'left_turn' or behavior == 'lturn':
        start_point = (width // 5*2, hight // 5 * 3)
        end_point   = (width // 5*1, hight // 5 * 3)
        cv2.arrowedLine(image, start_point, end_point, color, thickness, tipLength=0.4)
        start_point = (width // 5*2, hight // 5 * 4)
        end_point   = (width // 5*2, hight // 5 * 3)
        cv2.line(image, start_point, end_point, color, thickness)
        cv2.putText(image, 'left turn', (width//2-70, hight//5*4+50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 1)
    elif behavior == 'right_turn' or behavior == 'rturn':
        start_point = (width // 5*3, hight // 5 * 3)
        end_point   = (width // 5*4, hight // 5 * 3)
        cv2.arrowedLine(image, start_point, end_point, color, thickness, tipLength=0.4)
        start_point = (width // 5*3, hight // 5 * 4)
        end_point   = (width // 5*3, hight // 5 * 3)
        cv2.line(image, start_point, end_point, color, thickness)
        cv2.putText(image, 'right turn', (width//2-70, hight//5*4+50), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 0, 255), 1)
    return image

def run_simulation(cfg, client):
    """This function performed one test run using the args parameters
    and connecting to the carla client passed.
    """
    actor_list = []

    try:
        # Getting the world and
        world = client.get_world()
        original_settings = world.get_settings()
        vehicle = None

        if cfg['sync_mode'] == True:
            print('It is sync mode')
            traffic_manager = client.get_trafficmanager(8000)
            settings = world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            settings.sync_modehronous_mode = True
            settings.fixed_delta_seconds = 0.010
            world.apply_settings(settings)
        # Instanciating the vehicle
        if cfg['use_socket']==True:
            # intialize socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            server_address = (cfg['socket_address'], cfg['socket_port'])  # 设置接收方的IP地址和端口号
            sock.bind(server_address)
            data, address = sock.recvfrom(4096)
            ego_id = int(data.decode())
            print('Ego ID: ', ego_id)
            # close socket
            sock.close()
            vehicle = world.get_actor(ego_id)
        else: # creat vehicle
            bp = world.get_blueprint_library().filter('charger_2020')[0]
            vehicle = world.spawn_actor(bp, random.choice(world.get_map().get_spawn_points()))
            actor_list.append(vehicle)
            vehicle.set_autopilot(True)

        # SensorManager: RGBCamera, SemanticCamera
        IM_WIDTH = cfg['im_width']
        IM_HEIIGHT = cfg['im_hight']
        QUEUE_LEN = cfg['queue_length']
        # RGB camera
        camera_rgb = init_camera(world, 'RGBCamera', carla.Transform(carla.Location(x=0, z=2.4), 
                                 carla.Rotation(yaw=0)), vehicle, IM_WIDTH, IM_HEIIGHT, 90 )
        actor_list.append(camera_rgb)
        image_rgb_queue = []
        camera_rgb.listen(lambda image: process_rgb_image(image, image_rgb_queue, QUEUE_LEN))
        # Semantic Camera
        camera_sematic = init_camera(world, 'SemanticCamera', carla.Transform(carla.Location(x=10,y=0,z=30.0), 
                                     carla.Rotation(pitch=-90, yaw=0, roll=0)), vehicle, IM_WIDTH, IM_HEIIGHT, 70 )
        actor_list.append(camera_sematic)
        image_sematic_queue = []
        camera_sematic.listen(lambda image: process_semantic_image(image, image_sematic_queue, QUEUE_LEN))

        wheel_angle_queue = []

        # Initialize camera
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, IM_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IM_HEIIGHT)

        lat_acc_queue = []

        #Simulation loop
        while True:
            # Carla Tick
            if cfg['sync_mode']  == True:
                world.tick()     
            time_start = time.time()
            # =======================================================================
            if cap.isOpened():
                success, driver_img = cap.read()  # Capture frame-by-frame
            # =======================================================================
            ego_tf = vehicle.get_transform()
            ego_velocity_3d = vehicle.get_velocity()
            ego_velocity_ms = math.sqrt(ego_velocity_3d.x**2 + ego_velocity_3d.y**2 + ego_velocity_3d.z**2)
            ego_velocity_kmh = round(ego_velocity_ms * 3.6, 1)
            wheel_angle = vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.Front_Wheel)
            wheel_angle_queue.append(wheel_angle)
            if len(wheel_angle_queue)>5:
                wheel_angle_queue.pop(0)


            if cfg['use_socket']==False:
                # top view spectator
                spectator = world.get_spectator()
                ego_yaw = math.radians(vehicle.get_transform().rotation.yaw)
                spectator.set_transform(carla.Transform(ego_tf.location + 
                        carla.Location(x=-10*math.cos(ego_yaw),y=-10*math.sin(ego_yaw),z=7.5),
                        carla.Rotation(pitch=-30,yaw=math.degrees(ego_yaw))))
                # disable traffic light
                traffic_light = vehicle.get_traffic_light()
                if traffic_light != None:
                    traffic_light.set_state(carla.TrafficLightState.Green)
                    traffic_light.set_green_time(100.0)
                    traffic_light.set_red_time(0.0)
                    traffic_light.set_yellow_time(0.0)
                # draw ego prediction trajectory
                steer = vehicle.get_control().steer
                horizon, dt = cfg["ego_traj_horizon"], cfg["ego_traj_dt"]
                trajectory = get_prediction(ego_tf, ego_velocity_ms, steer-0.0, horizon, dt)
                draw_trajectory(trajectory, ego_tf, world)
            

            ego_w = vehicle.get_angular_velocity().z
            lat_acc = round(ego_velocity_ms * ego_w, 3)
            lat_acc_queue.append(lat_acc)
            if len(lat_acc_queue)>20:
                lat_acc_queue.pop(0)
            avg_lat_acc = 0
            if len(lat_acc_queue) != 0:
                total = sum(lat_acc_queue)  # 求和
                avg_lat_acc = total / len(lat_acc_queue)  # 求平均值
            # =================================== do something  =================================
            if len(image_rgb_queue)>1 and len(image_sematic_queue)>1 :
                image_rgb = image_rgb_queue[-1]
                image_sematic = image_sematic_queue[-1]

                wheel_angle = sum(wheel_angle_queue)/ len(wheel_angle_queue)
                if wheel_angle<4.0 and  wheel_angle>-4.0:
                    height, width = image_rgb.shape[:2]
                    draw_behavior(image_rgb, width, height, 'lane_keep')
                elif wheel_angle>=4.0 and  wheel_angle<7.0:
                    height, width = image_rgb.shape[:2]
                    draw_behavior(image_rgb, width, height, 'right_change')
                elif wheel_angle<=-4.0 and  wheel_angle>=-7.0:
                    height, width = image_rgb.shape[:2]
                    draw_behavior(image_rgb, width, height, 'left_change')
                elif wheel_angle>7.0:
                    height, width = image_rgb.shape[:2]
                    draw_behavior(image_rgb, width, height, 'right_turn')
                elif wheel_angle<-7.0:
                    height, width = image_rgb.shape[:2]
                    draw_behavior(image_rgb, width, height, 'left_turn')

                speed_inform =    "Speed:    " + str(ego_velocity_kmh) + " km/h"
                wheel_angle_inform = "Wheel angle: " + str(round(wheel_angle,2))
                lat_acc_info =    "Lat_acc:  " + str(lat_acc)
                w_info =          "w_z:      " + str(round( ego_w,3)) 
                behavior_inform = "Behavior: lane keeping"
                if avg_lat_acc > 5.0:
                    behavior_inform = "Behavior: right change"
                elif avg_lat_acc < -5.0:
                    behavior_inform = "Behavior: left change"
                cv2.putText(image_rgb, speed_inform, (30, 35), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(image_rgb, wheel_angle_inform, (30, 60), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(image_rgb, lat_acc_info, (30, 85), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1)
                cv2.putText(image_rgb, behavior_inform, (30, 110), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(image_rgb, w_info, (30, 135), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1)
                # imgs = np.vstack([image_rgb, image_sematic, driver_img])
                imgs = np.vstack([image_rgb, driver_img])
                cv2.imshow("output",imgs)

                # cv2.imshow("driver",driver_img)
                # print(driver_img.shape,  imgs.shape, image_rgb.shape)
                # print(cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT)
                cv2.waitKey(5)
                # TODO ****************************************************************
                # image_rgb_queue 为需要的历史图片，长度由 第255行的 "queue_length" 决定



            # ====================================================================================
            time_end = time.time()
            print('speed:', ego_velocity_kmh,  ' time cost:', round(time_end-time_start,3), 
                  ' len: ', len(image_rgb_queue), len(image_sematic_queue))

    finally:
        for actor in actor_list:
            actor.destroy()
        cap.release()
        cv2.destroyAllWindows()
        print('destroy all actor')
        if cfg['use_socket']==True:
            world.apply_settings(original_settings)   

def main():
    try:
        host = '127.0.0.1'
        port = 2000
        client = carla.Client(host, port)
        client.set_timeout(10.0)
        cfg = {
            'socket_address': '127.0.0.1',
            'socket_port': 12345,
            'ego_traj_horizon': 8,
            'ego_traj_dt': 0.2,
            'sync_mode': False,
            'use_socket': True,
            
            'im_width': 640,
            'im_hight': 480,
            'queue_length': 20,
            }

        run_simulation(cfg, client)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
