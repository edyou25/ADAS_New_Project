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
import matplotlib.pyplot as plt
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
    elif sensor_type == 'LiDAR':
        lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_bp.set_attribute('range', '100')
        lidar_bp.set_attribute('dropoff_general_rate', lidar_bp.get_attribute('dropoff_general_rate').recommended_values[0])
        lidar_bp.set_attribute('dropoff_intensity_limit', lidar_bp.get_attribute('dropoff_intensity_limit').recommended_values[0])
        lidar_bp.set_attribute('dropoff_zero_intensity', lidar_bp.get_attribute('dropoff_zero_intensity').recommended_values[0])
        lidar = world.spawn_actor(lidar_bp, transform, attach_to=attached)
        return lidar
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

def process_lidar_image(image, width, length, lidar_queue, queue_len):
    disp_size = [length, width ]
    lidar_range = 2.0*float('100')

    points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
    points = np.reshape(points, (int(points.shape[0] / 4), 4))
    lidar_data = np.array(points[:, :2])
    lidar_data *= min(disp_size) / lidar_range
    lidar_data += (0.5 * disp_size[0], 0.5 * disp_size[1])
    lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
    lidar_data = lidar_data.astype(np.int32)
    lidar_data = np.reshape(lidar_data, (-1, 2))
    lidar_img_size = (disp_size[0], disp_size[1], 3)
    lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)

    lidar_img[tuple(lidar_data.T)] = (255, 255, 255)

    lidar_queue.append(lidar_img)
    if len(lidar_queue) > queue_len:
        lidar_queue.pop(0)


def get_prediction(cur_tf, vel, steer_cur, horizon, dt):
    trajectory = []
    L = 4.0
    x0, y0, yaw0 = cur_tf.location.x, cur_tf.location.y, math.radians(cur_tf.rotation.yaw)
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


def run_simulation(cfg, client):
    actor_list = []
    try:
        world = client.get_world()
        original_settings = world.get_settings()
        vehicle = None
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
        ego_id = vehicle.id

        # SensorManager: RGBCamera, SemanticCamera
        IM_WIDTH = cfg['im_width']
        IM_HEIIGHT = cfg['im_hight']
        QUEUE_LEN = cfg['queue_length']
        # RGB camera
        # front
        camera_front = init_camera(world, 'RGBCamera', carla.Transform(carla.Location(x=0, z=2.4), 
                                 carla.Rotation(yaw=0)), vehicle, IM_WIDTH, IM_HEIIGHT, 90 )
        actor_list.append(camera_front)
        image_front_queue = []
        camera_front.listen(lambda image: process_rgb_image(image, image_front_queue, QUEUE_LEN))
        # front left
        camera_front_left = init_camera(world, 'RGBCamera', carla.Transform(carla.Location(x=0, z=2.4), 
                                 carla.Rotation(yaw=-60)), vehicle, IM_WIDTH, IM_HEIIGHT, 90 )
        actor_list.append(camera_front_left)
        image_front_left_queue = []
        camera_front_left.listen(lambda image: process_rgb_image(image, image_front_left_queue, QUEUE_LEN))
        # front fight
        camera_front_right = init_camera(world, 'RGBCamera', carla.Transform(carla.Location(x=0, z=2.4), 
                                 carla.Rotation(yaw=60)), vehicle, IM_WIDTH, IM_HEIIGHT, 90 )
        actor_list.append(camera_front_right)
        image_front_right_queue = []
        camera_front_right.listen(lambda image: process_rgb_image(image, image_front_right_queue, QUEUE_LEN))
        # back
        camera_rgb_back = init_camera(world, 'RGBCamera', carla.Transform(carla.Location(x=0, z=2.4), 
                                 carla.Rotation(yaw=-180)), vehicle, IM_WIDTH, IM_HEIIGHT, 90 )
        actor_list.append(camera_rgb_back)
        image_back_queue = []
        camera_rgb_back.listen(lambda image: process_rgb_image(image, image_back_queue, QUEUE_LEN))
        # back left
        camera_back_left = init_camera(world, 'RGBCamera', carla.Transform(carla.Location(x=0, z=2.4), 
                                 carla.Rotation(yaw=-120)), vehicle, IM_WIDTH, IM_HEIIGHT, 90 )
        actor_list.append(camera_back_left)
        image_back_left_queue = []
        camera_back_left.listen(lambda image: process_rgb_image(image, image_back_left_queue, QUEUE_LEN))
        # back fight
        camera_back_right = init_camera(world, 'RGBCamera', carla.Transform(carla.Location(x=0, z=2.4), 
                                 carla.Rotation(yaw=120)), vehicle, IM_WIDTH, IM_HEIIGHT, 90 )
        actor_list.append(camera_back_right)
        image_back_right_queue = []
        camera_back_right.listen(lambda image: process_rgb_image(image, image_back_right_queue, QUEUE_LEN))

        # Semantic Camera
        camera_sematic = init_camera(world, 'SemanticCamera', carla.Transform(carla.Location(x=0,y=0,z=20.4), 
                                     carla.Rotation(pitch=-90, yaw=0, roll=0)), vehicle, IM_WIDTH, IM_HEIIGHT*2, 70 )
        actor_list.append(camera_sematic)
        image_sematic_queue = []
        camera_sematic.listen(lambda image: process_semantic_image(image, image_sematic_queue, QUEUE_LEN))
        # LiDAR
        lidar = init_camera(world, 'LiDAR', carla.Transform(carla.Location(x=0,y=0,z=2.4), 
                                     carla.Rotation(pitch=0, yaw=0, roll=0)), vehicle, IM_WIDTH, IM_HEIIGHT, 70 )
        actor_list.append(lidar)
        lidar_queue = []
        lidar.listen(lambda image: process_lidar_image(image, IM_WIDTH*2, IM_HEIIGHT*2, lidar_queue, QUEUE_LEN))

        #Simulation loop
        plt.figure(figsize=(10,5), dpi=100)
        horizon, dt, distance_threshold = cfg["ego_traj_horizon"], cfg["ego_traj_dt"],cfg["distance_threshold"] 
        while True:   
            time_start = time.time()

            ego_tf = vehicle.get_transform()
            ego_velocity_3d = vehicle.get_velocity()
            ego_velocity_ms = math.sqrt(ego_velocity_3d.x**2 + ego_velocity_3d.y**2 + ego_velocity_3d.z**2)
            steer = vehicle.get_control().steer
            steer = max(steer, -0.02)
            steer = min(steer,  0.02)
            is_reverse = vehicle.get_control().reverse
            if is_reverse == True:
                ego_velocity_ms = -ego_velocity_ms
            ego_traj = get_prediction(ego_tf, ego_velocity_ms, steer, horizon, dt)
            # draw_trajectory(ego_traj, ego_tf, world)
            vehicles = client.get_world().get_actors().filter('vehicle.*')
            surrounding_vehicle = []
            all_trajectorys = []
            all_trajectorys.append(ego_traj)
            for veh_temp in vehicles:
                if veh_temp.id != ego_id:
                    other_tf = veh_temp.get_transform()
                    distance = math.sqrt((ego_tf.location.x - other_tf.location.x)**2 + 
                                        (ego_tf.location.y - other_tf.location.y)**2)
                    buffer = random.uniform(0.0, 10.0)
                    if(distance) <  distance_threshold + buffer:
                        surrounding_vehicle.append(veh_temp)
                        # other_steer = veh_temp.get_control().steer
                        # other_vel = veh_temp.get_velocity()
                        # other_vel_ms = math.sqrt(other_vel.x**2 + other_vel.y**2 + other_vel.z**2)
                        other_steer = random.uniform(-0.03, 0.03)
                        other_vel_ms = random.uniform(6.5, 9.5)
                        other_traj = get_prediction(other_tf, other_vel_ms, other_steer, horizon, dt)
                        all_trajectorys.append(other_traj)
                        # print('id: ',veh_temp.id, '  vel: ', round(other_vel_ms,3), 'steer', round(other_steer,3) )   
            # Get walkers
            walkers = client.get_world().get_actors().filter('walker.pedestrian.*')  
            # print(len(walkers))
            surrounding_walker = []
            walker_trajectorys = []
            for walker_temp in walkers:
                if walker_temp.id != ego_id:
                    other_tf = walker_temp.get_transform()
                    distance = math.sqrt((ego_tf.location.x - other_tf.location.x)**2 + 
                                        (ego_tf.location.y - other_tf.location.y)**2)
                    buffer = random.uniform(0.0, 5.0)
                    if(distance) <  distance_threshold + buffer:
                        # First trajectory
                        surrounding_walker.append(walker_temp)
                        other_steer = random.uniform(-0.1, 0.15)
                        other_vel_ms = random.uniform(3.5, 5.5)
                        other_traj = get_prediction(other_tf, other_vel_ms, other_steer, horizon, dt)
                        walker_trajectorys.append(other_traj)
                        # Second trajectory
                        other_steer = random.uniform(-0.15, 0.1)
                        other_vel_ms = random.uniform(3.5, 6.5)
                        other_traj = get_prediction(other_tf, other_vel_ms, other_steer, horizon, dt)
                        walker_trajectorys.append(other_traj)
                        # Third trajectory
                        other_steer = random.uniform(-0.55, 0.55)
                        other_vel_ms = random.uniform(3.5, 6.5)
                        other_traj = get_prediction(other_tf, other_vel_ms, other_steer, horizon, dt)
                        walker_trajectorys.append(other_traj)
                        # print('id: ',walker_temp.id, '  vel: ', round(other_vel_ms,3), 'steer', round(other_steer,3) ) 
            # ============================== part1 ==============================
            ego_box_w = vehicle.bounding_box.get_world_vertices(ego_tf)
            ego_box_w_x = [ego_box_w[0].x, ego_box_w[2].x, ego_box_w[6].x, ego_box_w[4].x, ego_box_w[0].x]
            ego_box_w_y = [-ego_box_w[0].y, -ego_box_w[2].y, -ego_box_w[6].y, -ego_box_w[4].y, -ego_box_w[0].y]
            p1 = plt.subplot(1, 2, 1)
            p1.cla()
            ego_box_b_x = [(x - ego_tf.location.x) for x in  ego_box_w_x ]
            ego_box_b_y = [(y + ego_tf.location.y) for y in  ego_box_w_y ]
            p1.plot(ego_box_b_x, ego_box_b_y)
            # ego trajectory
            x, y = [], []
            index = 0
            for temp in all_trajectorys[0]:
                x.append(temp[0] - ego_tf.location.x)
                y.append(-(temp[1] - ego_tf.location.y))
                alpha = 0.1
                alpha = max(alpha, 0.1)
                alpha = min(alpha, 1.0)
                plt.scatter(temp[0] - ego_tf.location.x,-(temp[1] - ego_tf.location.y),
                            s = 50, c ='r', alpha=alpha, marker='p' )
                index += 1
            p1.plot(x, y)
            # mot trajectory
            for trajectory in walker_trajectorys:
                x, y = [], []
                for temp in trajectory:
                    x.append(temp[0] - ego_tf.location.x)
                    y.append(-(temp[1] - ego_tf.location.y))
                size = random.randint(1, 5)
                p1.plot(x, y, marker='o', markersize=size, alpha=0.3, linewidth=2)
            plt.axis('equal')
            plt.xlim(-60, 60)
            plt.ylim(-60, 60) 
            plt.xticks([])
            plt.yticks([])
            # ============================== part2 ==============================
            p2 = plt.subplot(1, 2, 2)
            p2.cla()
            ego_box_b_x = [(x - ego_tf.location.x) for x in  ego_box_w_x ]
            ego_box_b_y = [(y + ego_tf.location.y) for y in  ego_box_w_y ]
            p2.plot(ego_box_b_x, ego_box_b_y)
            # ego trajectory
            x, y = [], []
            index = 0
            for temp in all_trajectorys[0]:
                x.append(temp[0] - ego_tf.location.x)
                y.append(-(temp[1] - ego_tf.location.y))
                alpha = 0.1
                alpha = max(alpha, 0.1)
                alpha = min(alpha, 1.0)
                plt.scatter(temp[0] - ego_tf.location.x,-(temp[1] - ego_tf.location.y),
                            s = 50, c ='r', alpha=alpha )
                index += 1
            p2.plot(x, y)
            # mot trajectory
            for trajectory in all_trajectorys[1:]:
                x, y = [], []
                for temp in trajectory:
                    x.append(temp[0] - ego_tf.location.x)
                    y.append(-(temp[1] - ego_tf.location.y))
                size = random.randint(6, 8)
                p2.plot(x, y, marker='D', markersize=size, alpha=0.1, linewidth=2)
                
            plt.axis('equal')
            plt.xlim(-60, 60)
            plt.ylim(-60, 60) 
            plt.xticks([])
            plt.yticks([])
            plt.pause(0.001)
            # =================================== do something  =================================
            if len(image_front_queue)>1 and len(image_front_left_queue)>1 and len(image_front_right_queue)>1 and \
               len(image_back_queue)>1 and len(image_back_left_queue)>1 and len(image_back_right_queue)>1 and  \
               len(lidar_queue)>1 and len(image_sematic_queue)>1 :
                
                image_front_left  = image_front_left_queue[-1]
                image_front       = image_front_queue[-1]
                image_front_right = image_front_right_queue[-1]
                image_back_left   = image_back_left_queue[-1]
                image_back        = image_back_queue[-1]
                image_back_right  = image_back_right_queue[-1]
                image_sematic     = image_sematic_queue[-1]
                lidar_image = lidar_queue[-1]
                img1 = np.hstack([image_front_left, image_front, image_front_right])
                img2 = np.hstack([image_back_left, image_back, image_back_right])
                img3 = np.hstack([lidar_image, image_sematic ])
                img4 = np.vstack([img1, img2, img3])
                cv2.imshow("input",img4)
                # cv2.imshow("lidar_image",lidar_image)
                cv2.waitKey(1)
                # TODO ****************************************************************



            # ====================================================================================
            time_end = time.time()
            print('v_num:', len(vehicles), 'w_num:',len(walkers),  '  obs_v', len(surrounding_vehicle), ' time cost:', round(time_end-time_start,3), 
                  ' len: ', len(image_front_queue), len(image_sematic_queue))

    finally:
        for actor in actor_list:
            actor.destroy()

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
            'ego_traj_horizon': 5,
            'ego_traj_dt': 0.3,
            'distance_threshold': 50,
            'use_socket': True,
            'im_width': 320,
            'im_hight': 240,
            'queue_length': 5,
            }

        run_simulation(cfg, client)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
