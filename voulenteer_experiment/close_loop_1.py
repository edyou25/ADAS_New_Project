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

import torch
import torchvision

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


try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image
from utils import *
from fiery.trainer import TrainingModule
from visualise import plot_prediction
import cv2

# TODO
images_camare_list = []
images_camare_tensor  = torch.zeros([   1, 6, 3, 224, 480])
images_history_list = []
images_history_tensor = torch.zeros([1, 3, 6, 3, 224, 480])

# intrinsics_tensor = torch.zeros([1, 3, 6, 3, 3])
# extrinsics_tensor = torch.zeros([1, 3, 6, 4, 4])
last_ego_transform = carla.Transform(carla.Location(x=0, y=0, z=0.0), carla.Rotation(yaw=0))
future_egomotion_list = []
future_egomotion_tensor = torch.zeros([1, 3, 6])


class CustomTimer:
    def __init__(self):
        try:
            self.timer = time.perf_counter
        except AttributeError:
            self.timer = time.time

    def time(self):
        return self.timer()

class DisplayManager:
    def __init__(self, grid_size, window_size):
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)

        self.grid_size = grid_size
        self.window_size = window_size
        self.sensor_list = []

    def get_window_size(self):
        return [int(self.window_size[0]), int(self.window_size[1])]

    def get_display_size(self):
        return [int(self.window_size[0]/self.grid_size[1]), int(self.window_size[1]/self.grid_size[0])]

    def get_display_offset(self, gridPos):
        dis_size = self.get_display_size()
        return [int(gridPos[1] * dis_size[0]), int(gridPos[0] * dis_size[1])]

    def add_sensor(self, sensor):
        self.sensor_list.append(sensor)

    def get_sensor_list(self):
        return self.sensor_list

    def render(self):
        if not self.render_enabled():
            return

        for s in self.sensor_list:
            s.render()

        pygame.display.flip()

    def destroy(self):
        for s in self.sensor_list:
            s.destroy()

    def render_enabled(self):
        return self.display != None

class SensorManager:
    def __init__(self, world, display_man, sensor_type, transform, attached, sensor_options, display_pos, fov, model):
        self.surface = None
        self.world = world
        self.display_man = display_man
        self.display_pos = display_pos
        self.sensor = self.init_sensor(sensor_type, transform, attached, sensor_options, fov)
        self.sensor_options = sensor_options
        self.timer = CustomTimer()

        self.time_processing = 0.0
        self.tics_processing = 0

        self.display_man.add_sensor(self)
        self.ego_vehicle = attached
        self.prediction_model = model

        # Normalising input images
        self.normalise_image = torchvision.transforms.Compose(
            [torchvision.transforms.ToPILImage(),
             torchvision.transforms.ToTensor(),
             torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def init_sensor(self, sensor_type, transform, attached, sensor_options, fov):
        if sensor_type == 'RGBCamera':
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            disp_size = self.display_man.get_display_size()
            # camera_bp.set_attribute('image_size_x', str(disp_size[0]))
            # camera_bp.set_attribute('image_size_y', str(disp_size[1]))

            camera_bp.set_attribute('image_size_x', str(1600))
            camera_bp.set_attribute('image_size_y', str(900))

            camera_bp.set_attribute("fov", str(fov))


            for key in sensor_options:
                camera_bp.set_attribute(key, sensor_options[key])

            camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
            camera.listen(self.parse_image)


            return camera
        else:
            return None

    def get_sensor(self):
        return self.sensor

    # def save_rgb_image(self, image):
    #     t_start = self.timer.time()

    #     image.convert(carla.ColorConverter.Raw)
    #     array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
    #     array = np.reshape(array, (image.height, image.width, 4))
    #     array = array[:, :, :3]

    #     # 缩放图像
    #     print('111', array.shape)
    #     array = cv2.resize(array, (480, 224), interpolation=cv2.INTER_LINEAR)
    #     print('333', array.shape)


    #     array = array[:, :, ::-1]
        
    #     # print(array.shape)

    #     if self.display_man.render_enabled():
    #         self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
    #     t_end = self.timer.time()
    #     self.time_processing += (t_end-t_start)
    #     self.tics_processing += 1

    #     self.parse_image(array)
  

    def parse_image(self, image):
        global images_camare_list
        global images_camare_tensor
        global images_history_list
        global images_history_tensor
        global last_ego_transform
        global future_egomotion_list
        global future_egomotion_tensor

        t_start = self.timer.time()
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]

        # 缩放图像
        print('111', array.shape)
        array = cv2.resize(array, (480, 224), interpolation=cv2.INTER_LINEAR)
        print('333', array.shape)

        array = array[:, :, ::-1]


        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1


        # TODO RGB 几个通道顺序的确定

        # step1.1 resize 和 crop，已经通过sensor设置间接完成
        # step1.2 normalise 
        normalised_img = self.normalise_image(array)      #(3, H, W)
        # step1.3 压入list & 扩维
        # print('====================len_images_list: ', len(images_camare_list))

        CAMERA_NUMBER = 6
        if len(images_camare_list) < CAMERA_NUMBER:
            # 压入 (1, 1, 3, H, W)
            images_camare_list.append(normalised_img.unsqueeze(0).unsqueeze(0))
        elif len(images_camare_list) == CAMERA_NUMBER:
            # 扩维 (1, 6, 3, H, W)
            images_camare_tensor = torch.cat(images_camare_list, dim=1)
            images_camare_list = []

        HISTORY_NUMBER = 3
        if len(images_history_list) >= HISTORY_NUMBER:
            images_history_list.pop(0)
            images_history_list.append(images_camare_tensor.unsqueeze(0))
        else:
            images_history_list.append(images_camare_tensor.unsqueeze(0))


        
        # step 2 内、外参
        intrinsics_tensor = get_intrinsic_tensor()
        extrinsics_tensor = get_extrinsic_tensor()

        # step 3.1
        future_egomotion_tensor_i = get_future_egomotion(last_ego_transform, self.ego_vehicle.get_transform())
        last_ego_transform = self.ego_vehicle.get_transform()
        # step 3.2
        if len(future_egomotion_list) >= HISTORY_NUMBER:
            future_egomotion_list.pop(0)
            future_egomotion_list.append(future_egomotion_tensor_i.unsqueeze(0))
        else:
            future_egomotion_list.append(future_egomotion_tensor_i.unsqueeze(0))

        

           
        




    def render(self):
        if self.surface is not None:
            offset = self.display_man.get_display_offset(self.display_pos)
            self.display_man.display.blit(self.surface, offset)

    def destroy(self):
        self.sensor.destroy()

def run_simulation(args, client):
    """This function performed one test run using the args parameters
    and connecting to the carla client passed.
    """
    global images_camare_list
    global images_camare_tensor
    global images_history_list
    global images_history_tensor
    global last_ego_transform
    global future_egomotion_list
    global future_egomotion_tensor

    display_manager = None
    vehicle = None
    vehicle_list = []
    timer = CustomTimer()

    try:

        # Getting the world and
        world = client.get_world()
        original_settings = world.get_settings()

        if args.sync:
            traffic_manager = client.get_trafficmanager(8000)
            settings = world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.5
            world.apply_settings(settings)


        # Instanciating the vehicle to which we attached the sensors
        bp = world.get_blueprint_library().filter('charger_2020')[0]
        vehicle = world.spawn_actor(bp, random.choice(world.get_map().get_spawn_points()))
        vehicle_list.append(vehicle)
        vehicle.set_autopilot(True)

        # prediction model
        checkpoint_path = './fiery.ckpt'
        trainer = TrainingModule.load_from_checkpoint(checkpoint_path, strict=True)
        device = torch.device('cuda:0')
        trainer = trainer.to(device)
        trainer.eval()



        # Display Manager organize all the sensors an its display in a window
        # If can easily configure the grid and the total window size
        display_manager = DisplayManager(grid_size=[2, 3], window_size=[args.width, args.height])

        # Then, SensorManager can be used to spawn RGBCamera, LiDARs and SemanticLiDARs as needed
        # and assign each of them to a grid position, 
        SensorManager(world, display_manager, 'RGBCamera', carla.Transform(carla.Location(x=1.5, y=-0.7, z=2.0), carla.Rotation(yaw=-55)), 
                      vehicle, {}, display_pos=[0, 0], fov=70, model=trainer)
        SensorManager(world, display_manager, 'RGBCamera', carla.Transform(carla.Location(x=1.5, y=0.0,  z=2.0), carla.Rotation(yaw=+00)), 
                      vehicle, {}, display_pos=[0, 1], fov=70, model=trainer)
        SensorManager(world, display_manager, 'RGBCamera', carla.Transform(carla.Location(x=1.5, y=0.7,  z=2.0), carla.Rotation(yaw=+55)), 
                      vehicle, {}, display_pos=[0, 2], fov=70, model=trainer)
        
        SensorManager(world, display_manager, 'RGBCamera', carla.Transform(carla.Location(x=-0.7, y=0.0, z=2.0), carla.Rotation(yaw=-110)), 
                      vehicle, {}, display_pos=[1, 0], fov=70, model=trainer)
        SensorManager(world, display_manager, 'RGBCamera', carla.Transform(carla.Location(x=-1.5, y=0.0, z=2.0), carla.Rotation(yaw=180)), 
                      vehicle, {}, display_pos=[1, 1], fov=110, model=trainer)
        SensorManager(world, display_manager, 'RGBCamera', carla.Transform(carla.Location(x=-0.7, y=0.0, z=2.0), carla.Rotation(yaw=110)), 
                      vehicle, {}, display_pos=[1, 2], fov=70, model=trainer)




        #Simulation loop
        call_exit = False
        time_init_sim = timer.time()
        
        idx = 0
        while idx < 20:
            print('idx = ', idx)
            idx += 1
            # Carla Tick
            if args.sync:
                world.tick()
            else:
                world.wait_for_tick()

            # Render received data
            display_manager.render()

            # top view
            spectator = world.get_spectator()
            ego_tf = vehicle.get_transform()
            ego_yaw = vehicle.get_transform().rotation.yaw
            spectator.set_transform(carla.Transform(ego_tf.location + carla.Location(x=-20,y=0,z=10),
                                                    carla.Rotation(pitch=-20,yaw=0)))
            
            # fellow view
            # spectator = world.get_spectator()
            # camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
            # camera_transform = carla.Transform(carla.Location(x=0, z=10.0), carla.Rotation(yaw=0))
            # camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
            # spectator.set_transform(camera.get_transform())
            HISTORY_NUMBER = 3
            intrinsics_tensor = get_intrinsic_tensor()
            extrinsics_tensor = get_extrinsic_tensor()
            while len(images_history_list) > HISTORY_NUMBER:
                images_history_list.pop(0)
            while len(future_egomotion_list) > HISTORY_NUMBER:
                future_egomotion_list.pop(0)
            if len(images_history_list) == HISTORY_NUMBER:
                images_history_tensor = torch.cat(images_history_list,  dim=1)
            if len(future_egomotion_list) == HISTORY_NUMBER:
                future_egomotion_tensor = torch.cat(future_egomotion_list,  dim=1)

            print('history_list:           ',len(images_history_list))
            print('future_egomotion_list:  ',len(future_egomotion_list))
            print('images:                 ',images_history_tensor.shape)
            print('intrinsics:             ',intrinsics_tensor.shape)
            print('extrinsics:             ',extrinsics_tensor.shape)
            print('future_egomotion:       ',future_egomotion_tensor.shape)

            device = torch.device('cuda:0')
            if len(images_history_list)==HISTORY_NUMBER and len(future_egomotion_list)==HISTORY_NUMBER:
                with torch.no_grad():
                    output = trainer.model(images_history_tensor.to(device).to(torch.float32),
                                            intrinsics_tensor.to(device).to(torch.float32),
                                            extrinsics_tensor.to(device).to(torch.float32),
                                            future_egomotion_tensor.to(device).to(torch.float32))
                    


                    images_history_tensor.to(device),
                    intrinsics_tensor.to(device),
                    extrinsics_tensor.to(device),
                    future_egomotion_tensor.to(device)




                print('==============================================================')
                
                figure_numpy = plot_prediction(images_history_tensor, output, trainer.cfg)
                os.makedirs('./output_vis', exist_ok=True)
                output_filename = os.path.join('./output_vis', str(time.time())) + '.png'
                print('  Output_filename:', output_filename)
                Image.fromarray(figure_numpy).save(output_filename)
                print(f'  Saved output in  {output_filename}')
                print('***************************************************************')

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    call_exit = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == K_ESCAPE or event.key == K_q:
                        call_exit = True
                        break

            if call_exit:
                break
            time.sleep(0.2)

    finally:
        if display_manager:
            display_manager.destroy()

        client.apply_batch([carla.command.DestroyActor(x) for x in vehicle_list])

        world.apply_settings(original_settings)



def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Sensor tutorial')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--async',
        dest='sync',
        action='store_false',
        help='Asynchronous mode execution')
    argparser.set_defaults(sync=True)
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1600x500',
        help='window resolution (default: 1280x720)')

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(5.0)

        run_simulation(args, client)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
