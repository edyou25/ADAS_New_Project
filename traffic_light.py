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

def run_simulation(cfg, client):
    """This function performed one test run using the args parameters
    and connecting to the carla client passed.
    """
    actor_list = []

    try:
        # Getting the world and
        world = client.get_world()
        original_settings = world.get_settings()

        if cfg['sync_mode'] == True:
            print('It is sync mode')
            traffic_manager = client.get_trafficmanager(8000)
            settings = world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            settings.sync_modehronous_mode = True
            settings.fixed_delta_seconds = 0.010
            world.apply_settings(settings)

        bp = world.get_blueprint_library().filter('charger_2020')[0]
        vehicle = world.spawn_actor(bp, random.choice(world.get_map().get_spawn_points()))
        actor_list.append(vehicle)
        vehicle.set_autopilot(True)

        # SensorManager: RGBCamera
        IM_WIDTH = cfg['im_width']
        IM_HEIIGHT = cfg['im_hight']
        QUEUE_LEN = cfg['queue_length']
        camera_rgb = init_camera(world, 'RGBCamera', carla.Transform(carla.Location(x=0, z=2.4), 
                                 carla.Rotation(yaw=0)), vehicle, IM_WIDTH, IM_HEIIGHT, 70 )
        actor_list.append(camera_rgb)
        image_rgb_queue = []
        camera_rgb.listen(lambda image: process_rgb_image(image, image_rgb_queue, QUEUE_LEN))
        
        #Simulation loop
        while True:
            # Carla Tick
            if cfg['sync_mode']  == True:
                world.tick()     
            time_start = time.time()

            ego_tf = vehicle.get_transform()
            ego_velocity_3d = vehicle.get_velocity()
            ego_velocity_ms = math.sqrt(ego_velocity_3d.x**2 + ego_velocity_3d.y**2 + ego_velocity_3d.z**2)
            ego_velocity_kmh = round(ego_velocity_ms * 3.6, 1)
            wheel_angle = vehicle.get_wheel_steer_angle(carla.VehicleWheelLocation.Front_Wheel)

            # top view spectator
            spectator = world.get_spectator()
            ego_yaw = math.radians(vehicle.get_transform().rotation.yaw)
            spectator.set_transform(carla.Transform(ego_tf.location + 
                    carla.Location(x=-10*math.cos(ego_yaw),y=-10*math.sin(ego_yaw),z=7.5),
                    carla.Rotation(pitch=-30,yaw=math.degrees(ego_yaw))))

            # =================================== do something  =================================
            if len(image_rgb_queue)>1:
                image_rgb = image_rgb_queue[-1]


                speed_inform =       "Speed:       " + str(ego_velocity_kmh) + " km/h"
                wheel_angle_inform = "Wheel angle: " + str(round(wheel_angle,2))
                cv2.putText(image_rgb, speed_inform, (30, 35), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 1)
                cv2.putText(image_rgb, wheel_angle_inform, (30, 60), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 1)
                cv2.imshow("output",image_rgb)
                cv2.waitKey(5)
                # TODO T推理*********************************************************************
                # image_rgb 就是当前的输入

            # ====================================================================================
            time_end = time.time()
            print('speed:', ego_velocity_kmh,  ' time cost:', round(time_end-time_start,3), 
                  ' len: ', len(image_rgb_queue))

    finally:
        for actor in actor_list:
            actor.destroy()
        cv2.destroyAllWindows()
        print('destroy all actor')


def main():
    try:
        host = '127.0.0.1'
        # host = '192.168.5.5'
        port = 2000
        client = carla.Client(host, port)
        client.set_timeout(10.0)
        cfg = {
            'sync_mode': False,  
            'im_width': 640,
            'im_hight': 480,
            # 'im_width': 1920,
            # 'im_hight': 1080,
            'queue_length': 5,
            }

        run_simulation(cfg, client)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
