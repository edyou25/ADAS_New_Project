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
import argparse
import random
import time
import math
import numpy as np
import torch
import torchvision
import PIL
from PIL import Image
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
import carla

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

from risk_assessment.collision_octagon import CollisionOctagon


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
    def __init__(self, world, display_man, sensor_type, transform, attached, sensor_options, display_pos, fov):
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

    def init_sensor(self, sensor_type, transform, attached, sensor_options, fov):
        if sensor_type == 'RGBCamera':
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            disp_size = self.display_man.get_display_size()
            camera_bp.set_attribute('image_size_x', str('600'))
            camera_bp.set_attribute('image_size_y', str('400'))
            camera_bp.set_attribute("fov", str(fov))

            for key in sensor_options:
                camera_bp.set_attribute(key, sensor_options[key])

            camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached)
            camera.listen(self.save_rgb_image)

            return camera
        else:
            return None

    def get_sensor(self):
        return self.sensor

    def save_rgb_image(self, image):
        t_start = self.timer.time()

        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        
        # print(array.shape)

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1

        # self.parse_image(image)
  

    def parse_image(self, image):
        global images_camare_list
        global images_camare_tensor
        global images_history_list
        global images_history_tensor

        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        img = Image.fromarray(array)
        # img = img.resize((480, 270), resample=PIL.Image.BILINEAR)
        # img = img.crop((0, 46, 480, 270))

    def render(self):
        if self.surface is not None:
            offset = self.display_man.get_display_offset(self.display_pos)
            temp = pygame.transform.rotozoom(self.surface, 0,1.0) # 旋转0度，缩放0.5
            self.display_man.display.blit(temp, offset)

    def destroy(self):
        self.sensor.destroy()

def get_prediction(cur_tf, vel, steer, horizon, dt):
    trajectory = []
    L = 4.0
    x0, y0, yaw0 = cur_tf.location.x, cur_tf.location.y, math.radians(cur_tf.rotation.yaw)
    # trajectory.append([x0, y0, yaw0, vel])

    for idx in range(horizon):
        x1 = x0 + vel*math.cos(yaw0)*dt
        y1 = y0 + vel*math.sin(yaw0)*dt
        yaw1 = yaw0 + vel / L * math.tan(steer)*dt
        x0, y0, yaw0 = x1, y1, yaw1
        trajectory.append([x1, y1, yaw1, vel])

    return trajectory

def get_risk(ego_traj, other_trajs, horizon):
    risk_ego = []
    risk_mot = []
    risk_mot_all = []
    for other_traj in other_trajs:
        each_mot_p = []
        sum = 0
        for t in range(len(ego_traj)):
            ego_x, ego_y, _, _ = ego_traj[t]
            other_x, other_y, _, _ = other_traj[t]
            ego_box = np.array([[ego_x-5, ego_y+5],[ego_x+5, ego_y+5],
                               [ego_x+5, ego_y-5],[ego_x-5, ego_y-5]])
            other_box = np.array([[other_x-1, other_y+1],[other_x+1, other_y+1],
                                 [other_x+1, other_y-1],[other_x-1, other_y-1]])
            collision = CollisionOctagon(ego_box, other_box)
            # TODO 
            sigma_x, sigma_y = 30.0, 30.0  
            probability = collision.compute_CSP(sigma_x, sigma_y)
            sum += probability
            each_mot_p.append(probability)
        risk_mot_all.append(each_mot_p)
        risk_mot.append(sum)
    
    for t in range(horizon):
        sum = 0.0
        for mot_id in range(len(risk_mot_all)):
            sum += risk_mot_all[mot_id][t]
        risk_ego.append(sum)
    return risk_ego, risk_mot

def run_simulation(args, client):
    """This function performed one test run using the args parameters
    and connecting to the carla client passed.
    """
    display_manager = None
    vehicle = None
    vehicle_list = []
    timer = CustomTimer()
    ENABLE_PYGAME = False
    VEHICLE_NUMBER = 50
    SIMULATION_LOOP = 100
    DISTANCE_THRESHOLD = 50 
    PRDDICTION_HORIZON = 10
    PREDICTION_dt = 0.2

    try:
        # Getting the world and
        world = client.get_world()
        # world = client.load_world('Town04')
        world.unload_map_layer(carla.MapLayer.Buildings)     # 关闭显示所有建筑物
        # world.load_map_layer(carla.MapLayer.Buildings)
        original_settings = world.get_settings()

        # sync mode
        if args.sync:
            traffic_manager = client.get_trafficmanager(8000)
            settings = world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.1
            world.apply_settings(settings)

        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        # ego vehicle
        bp = world.get_blueprint_library().filter('charger_2020')[0]
        vehicle = world.spawn_actor(bp, spawn_points[0])
        vehicle_list.append(vehicle)
        vehicle.set_autopilot(True)
        # other vehicles
        blueprints = world.get_blueprint_library().filter('vehicle.*')
        number_of_spawn_points = len(spawn_points)
        if number_of_spawn_points < VEHICLE_NUMBER:
            VEHICLE_NUMBER = number_of_spawn_points
        for index in range(1, VEHICLE_NUMBER):
            blueprint_i = random.choice(blueprints)
            vehicle_temp = world.spawn_actor(blueprint_i, spawn_points[index])
            vehicle_temp.set_autopilot(True)
            vehicle_list.append(vehicle_temp)
        print('************** the vehicle number is: ',VEHICLE_NUMBER, '*******************')
        
        # Display Manager organize all the sensors an its display in a window
        if ENABLE_PYGAME == True:
            display_manager = DisplayManager(grid_size=[2, 3], window_size=[args.width, args.height])
            SensorManager(world, display_manager, 'RGBCamera', 
                          carla.Transform(carla.Location(x=1.3, y=-0.4, z=2.8), carla.Rotation(yaw=-55)), 
                          vehicle, {}, display_pos=[0, 0], fov=70)
            SensorManager(world, display_manager, 'RGBCamera', 
                          carla.Transform(carla.Location(x=1.5, y=0.0,  z=2.8), carla.Rotation(yaw=+00)), 
                          vehicle, {}, display_pos=[0, 1], fov=70)
            SensorManager(world, display_manager, 'RGBCamera', 
                          carla.Transform(carla.Location(x=1.3, y=0.4,  z=2.8), carla.Rotation(yaw=+55)), 
                          vehicle, {}, display_pos=[0, 2], fov=70)
            SensorManager(world, display_manager, 'RGBCamera', 
                          carla.Transform(carla.Location(x=-0.85, y=-0.4, z=2.8), carla.Rotation(yaw=-110)), 
                          vehicle, {}, display_pos=[1, 0], fov=70)
            SensorManager(world, display_manager, 'RGBCamera', 
                          carla.Transform(carla.Location(x=-2.00, y=0.0, z=2.8), carla.Rotation(yaw=180)), 
                          vehicle, {}, display_pos=[1, 1], fov=70)
            SensorManager(world, display_manager, 'RGBCamera', 
                          carla.Transform(carla.Location(x=-0.85, y=0.4, z=2.8), carla.Rotation(yaw=110)),
                          vehicle, {}, display_pos=[1, 2], fov=70)

        #Simulation loop
        plt.figure(figsize=(10,5), dpi=150)
        call_exit = False
        idx = 0
        while idx < SIMULATION_LOOP:
            print('\n***********************idx = ', idx)
            start = time.perf_counter()
            idx += 1

            # carla tick
            if args.sync:
                world.tick()
            else:
                world.wait_for_tick()
            # pygame render 
            if ENABLE_PYGAME == True:
                display_manager.render()  

            # disable traffic light
            traffic_light = vehicle.get_traffic_light()
            if traffic_light != None:
                traffic_light.set_state(carla.TrafficLightState.Green)
                traffic_light.set_green_time(100.0)
                traffic_light.set_red_time(0.0)
                traffic_light.set_yellow_time(0.0)
            # top view
            spectator = world.get_spectator()
            ego_tf = vehicle.get_transform()
            ego_yaw = math.radians(vehicle.get_transform().rotation.yaw)
            spectator.set_transform(carla.Transform(ego_tf.location + 
                    carla.Location(x=-10*math.cos(ego_yaw),y=-10*math.sin(ego_yaw),z=7.5),
                    carla.Rotation(pitch=-30,yaw=math.degrees(ego_yaw))))
            

            # vector = vehicle.get_velocity()
            # world.debug.draw_string(ego_tf.location + carla.Location(x=3,y=3,z=1), 
            #                         'hello', 
            #                             draw_shadow=False, life_time=0.1)
            # box = vehicle.bounding_box
            # high = math.fabs(ego_yaw) * 1
            # box.location += ego_tf.location + carla.Location(z=high/2) + carla.Location(x=3,y=3,z=0)
            # box.extent = carla.Vector3D(0.1,0.1, high) 
            # world.debug.draw_box(box, ego_tf.rotation,thickness=0.5, life_time=0.2)

            # ***************************************************
            # Prediction
            # ***************************************************
            surrounding_vehicle = []
            all_trajectorys = []
            for veh_temp in vehicle_list:
                location =  veh_temp.get_transform().location
                distance = math.sqrt((ego_tf.location.x - location.x)**2 + 
                                     (ego_tf.location.y - location.y)**2)
                if(distance) <=  DISTANCE_THRESHOLD:
                    surrounding_vehicle.append(veh_temp)

                    cur_tf = veh_temp.get_transform()
                    vel = veh_temp.get_velocity()
                    speed = math.sqrt(max(0.1, (vel.x**2 + vel.y**2 + vel.y**2)))
                    steer = veh_temp.get_control().steer
                    trajectory = get_prediction(cur_tf, speed, steer, PRDDICTION_HORIZON, PREDICTION_dt)
                    all_trajectorys.append(trajectory)
            
            risk_ego, risk_mot =  get_risk(all_trajectorys[0], all_trajectorys[1:],PRDDICTION_HORIZON)
            print(risk_ego, risk_mot)
            # ***************************************************
            # visualization
            # ***************************************************
            ego_box_w = vehicle.bounding_box.get_world_vertices(ego_tf)
            ego_box_w_x = [ego_box_w[0].x, ego_box_w[2].x, ego_box_w[6].x, ego_box_w[4].x, ego_box_w[0].x]
            ego_box_w_y = [-ego_box_w[0].y, -ego_box_w[2].y, -ego_box_w[6].y, -ego_box_w[4].y, -ego_box_w[0].y]
            # ego_box_b = vehicle.bounding_box.get_local_vertices()
            # ego_box_b_x = [ego_box_b[0].x, ego_box_b[2].x, ego_box_b[6].x, ego_box_b[4].x, ego_box_b[0].x]
            # ego_box_b_y = [-ego_box_b[0].y, -ego_box_b[2].y, -ego_box_b[6].y, -ego_box_b[4].y, -ego_box_b[0].y]
            # ============================== part1 ==============================
            p1 = plt.subplot(1, 2, 1)
            p1.cla()
            p1.plot(ego_box_w_x, ego_box_w_y)
            p1.scatter(ego_tf.location.x, -ego_tf.location.y)
            for  veh_temp in surrounding_vehicle:         
                location =  veh_temp.get_transform().location
                p1.scatter(location.x, -location.y)
            plt.axis('equal')
            plt.grid(True)
            plt.xlim(-150, 150)
            plt.ylim(-150, 150) 

            # ============================== part2 ==============================
            p2 = plt.subplot(1, 2, 2)
            p2.cla()
            ego_box_b_x = [(x - ego_tf.location.x) for x in  ego_box_w_x ]
            ego_box_b_y = [(y + ego_tf.location.y) for y in  ego_box_w_y ]
            p2.plot(ego_box_b_x, ego_box_b_y)
            # p2.scatter(0, 0)
            # for  veh_temp in surrounding_vehicle:
            #     location =  veh_temp.get_transform().location
            #     p2.scatter(location.x-ego_tf.location.x, -(location.y-ego_tf.location.y))
            # ego trajectory
            x, y = [], []
            index = 0
            for temp in all_trajectorys[0]:
                x.append(temp[0] - ego_tf.location.x)
                y.append(-(temp[1] - ego_tf.location.y))
                alpha = risk_ego[index]*20
                alpha = max(alpha, 0.2)
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
                p2.plot(x, y, marker='*', markersize=3)
                
            plt.axis('equal')
            plt.xlim(-60, 60)
            plt.ylim(-60, 60) 
            plt.pause(0.001)
            

            if ENABLE_PYGAME == True:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        call_exit = True
                    elif event.type == pygame.KEYDOWN:
                        if event.key == K_ESCAPE or event.key == K_q:
                            call_exit = True
                            break
                if call_exit:
                    break
            end = time.perf_counter()
            print('用时：{:.4f}s'.format(end-start))
        


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
        # default='10.20.4.216',
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
        default='1800x800',
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
