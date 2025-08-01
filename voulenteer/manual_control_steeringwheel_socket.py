#!/usr/bin/env python

# Copyright (c) 2019 Intel Labs
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
Welcome to CARLA manual control with steering wheel Logitech G29.

To drive start by preshing the brake pedal.
Change your wheel_config.ini according to your steering wheel.

To find out the values of your steering wheel use jstest-gtk in Ubuntu.

"""

from __future__ import print_function


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================

from plot_box_utils import *
import glob
import os
import sys
import json

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
import matplotlib as mpl
import matplotlib.pyplot as plt
from queue import Queue
import cv2
import math
import socket
import datetime
import time

# Force Seat
from ForceSeatMI import *


if sys.version_info >= (3, 0):

    from configparser import ConfigParser

else:

    from ConfigParser import RawConfigParser as ConfigParser

# keyboard
try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    def __init__(self, carla_world, hud, args, hud_info_queue):
        self.world = carla_world
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter

        self.ego_risk = 0.0
        self.scenario_idx = 0
        self.scenario_length = len(args.scenario_config)
        self.scenario_config = args.scenario_config

        self.hud_info_queue = hud_info_queue
        self.enable_recording = False
        

        self.restart()
        self.world.on_tick(hud.on_world_tick)



    def restart(self):
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_index = self.camera_manager.transform_index if self.camera_manager is not None else 0
        blueprint = self.world.get_blueprint_library().filter("charger_2020")[0]
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            # color = random.choice(blueprint.get_attribute('color').recommended_values)
            # blueprint.set_attribute('color', color)
            blueprint.set_attribute('color', '0,39,105')
        # Spawn the player.
        if self.player is not None:
            self.scenario_idx += 1
            self.scenario_idx %= self.scenario_length

            sensors = [ self.camera_manager.sensor,
                        self.collision_sensor.sensor,
                        self.lane_invasion_sensor.sensor,
                        self.gnss_sensor.sensor]
            for sensor in sensors:
                if sensor is not None:
                    sensor.stop()
                    sensor.destroy()

            if self.scenario_length == 0:
                self.player.set_transform(carla.Transform(carla.Location(x=43.3, y=-112.4, z=5), 
                                                          carla.Rotation(pitch=-0.43, yaw=137.3, roll=-0.02)))
            else:
                location_x = self.scenario_config[self.scenario_idx]['ego_start_point']['x']
                location_y = self.scenario_config[self.scenario_idx]['ego_start_point']['y']
                location_z = self.scenario_config[self.scenario_idx]['ego_start_point']['z']
                yaw        = self.scenario_config[self.scenario_idx]['ego_start_point']['yaw']
                self.player.set_transform(carla.Transform(carla.Location(x=location_x, y=location_y, z=location_z), 
                                                        carla.Rotation(pitch=0, yaw=yaw, roll=0)))
            self.player.set_target_velocity(carla.Vector3D(x=0,y=0,z=0))

        while self.player is None:
            if self.scenario_length == 0:
                spawn_point = carla.Transform(carla.Location(x=43.3, y=-112.4, z=5), 
                                            carla.Rotation(pitch=-0.43, yaw=137.3, roll=-0.02))
            else:
                location_x = self.scenario_config[self.scenario_idx]['ego_start_point']['x']
                location_y = self.scenario_config[self.scenario_idx]['ego_start_point']['y']
                location_z = self.scenario_config[self.scenario_idx]['ego_start_point']['z']
                yaw        = self.scenario_config[self.scenario_idx]['ego_start_point']['yaw']
                spawn_point = carla.Transform(carla.Location(x=location_x, y=location_y, z=location_z), 
                                              carla.Rotation(pitch=0.0, yaw=yaw, roll=0.0))
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self.hud_info_queue)
        self.camera_manager.transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy(self):
        sensors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor]
        for sensor in sensors:
            if sensor is not None:
                sensor.stop()
                sensor.destroy()
        if self.player is not None:
            self.player.destroy()

# ==============================================================================
# -- DualControl -----------------------------------------------------------
# ==============================================================================


class DualControl(object):
    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            # world.player.set_autopilot(self._autopilot_enabled)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

        # initialize steering wheel
        pygame.joystick.init()

        joystick_count = pygame.joystick.get_count()
        if joystick_count > 1:
            pass
            # raise ValueError("Please Connect Just One Joystick")

        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()

        self._parser = ConfigParser()
        self._parser.read('wheel_config.ini')
        self._steer_idx = int(
            self._parser.get('G29 Racing Wheel', 'steering_wheel'))
        self._throttle_idx = int(
            self._parser.get('G29 Racing Wheel', 'throttle'))
        self._brake_idx = int(self._parser.get('G29 Racing Wheel', 'brake'))
        self._reverse_idx = int(self._parser.get('G29 Racing Wheel', 'reverse'))
        self._handbrake_idx = int(
            self._parser.get('G29 Racing Wheel', 'handbrake'))

    def parse_events(self, world, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button == 0:
                    world.restart()
                elif event.button == 1:
                    world.hud.toggle_info()
                elif event.button == 2:
                    world.camera_manager.toggle_camera()
                elif event.button == 3:
                    world.next_weather()
                elif event.button == self._reverse_idx:
                    self._control.gear = 1 if self._control.reverse else -1
                elif event.button == 23:
                    world.camera_manager.next_sensor()
                elif event.button == 4:
                    world.enable_recording = not world.enable_recording
                    

            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r:
                    world.camera_manager.toggle_recording()
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p:
                        self._autopilot_enabled = not self._autopilot_enabled
                        # world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._parse_vehicle_wheel()
                self._control.reverse = self._control.gear < 0
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time())
            world.player.apply_control(self._control)

    def _parse_vehicle_keys(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    def _parse_vehicle_wheel(self):
        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]
        # print (jsInputs)
        jsButtons = [float(self._joystick.get_button(i)) for i in
                     range(self._joystick.get_numbuttons())]

        # Custom function to map range of inputs [1, -1] to outputs [0, 1] i.e 1 from inputs means nothing is pressed
        # For the steering, it seems fine as it is
        K1 = 1.0  # 0.55
        steerCmd = K1 * math.tan(1.1 * jsInputs[self._steer_idx])

        K2 = 1.6  # 1.6
        throttleCmd = K2 + (2.05 * math.log10(
            -0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
        brakeCmd = 1.6 + (2.05 * math.log10(
            -0.7 * jsInputs[self._brake_idx] + 1.4) - 1.2) / 0.92

        if throttleCmd < 0:
            throttleCmd = 0
        elif throttleCmd > 1:
            throttleCmd = 1

        if brakeCmd < 0:
            brakeCmd = 0
        elif brakeCmd > 1:
            brakeCmd = 1

        self._control.steer = steerCmd
        self._control.brake = brakeCmd
        self._control.throttle = throttleCmd

        #toggle = jsButtons[self._reverse_idx]

        self._control.hand_brake = bool(jsButtons[self._handbrake_idx])

    def _parse_walker_keys(self, keys, milliseconds):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = 5.556 if pygame.key.get_mods() & KMOD_SHIFT else 2.778
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._speed_font_mono = pygame.font.Font(mono, 25 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._speed_info = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.world.get_map().name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision
            ]
# ==================================================================
        # self._info_text += [
        #     '',
        #     '',
        #     '',
        #     '',
        #     '',
        #     '',
        #     'Speed: %2.0fkm/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))]
# ==================================================================
        # if len(vehicles) > 1:
        #     self._info_text += ['Nearby vehicles:']
        #     distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
        #     vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
        #     for d, vehicle in sorted(vehicles):
        #         if d > 200.0:
        #             break
        #         vehicle_type = get_actor_display_name(vehicle, truncate=22)
        #         self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for index,item in enumerate(self._info_text):
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item and index != len(self._info_text)-1:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))

                if index == len(self._info_text)-1:
                    surface = self._speed_font_mono.render(item, True, (0, 255, 0))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
            # for item in self._speed_info:
            #     if item:  # At this point has to be a str.
            #         surface = self._speed_font_mono.render(item, True, (0, 255, 0))
            #         display.blit(surface, (8, v_offset))
        self._notifications.render(display)
        self.help.render(display)


# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))

# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    def __init__(self, parent_actor, hud, hud_info_queue):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self._camera_transforms = [
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            carla.Transform(carla.Location(x=0.42, y=0.27,z=1.4),carla.Rotation(pitch=-10))]
        # carla.Transform(carla.Location(x=0.3, y=0.35,z=1.3),carla.Rotation(pitch=-15))]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
                bp.set_attribute('fov', str(130))
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '50')
            item.append(bp)
        self.index = None

        self.hud_info_queue = hud_info_queue


    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.sensor.set_transform(self._camera_transforms[self.transform_index])

    def set_sensor(self, index, notify=True):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None \
            else self.sensors[index][0] != self.sensors[self.index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index],
                attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image, self.hud_info_queue))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image, hud_info_queue):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data) # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]

            if len(hud_info_queue) > 1:
                hud_info = hud_info_queue[-1]
                ego_risk = hud_info.ego_risk

                array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
                # array = where_is_risk(array, 0, ego_risk)
                left_image = hud_info.left_mirror
                right_image = hud_info.right_mirror
                big_h, big_w, _ = array.shape
                small_h1, small_w1, _ = left_image.shape
                small_h2, small_w2, _ = right_image.shape
                x1 = 300
                y1 = 500
                x2 = big_w - small_w2 - x1 
                y2 = y1
                array[y1:y1+small_h1, x1:x1+small_w1] = left_image
                array[y2:y2+small_h2, x2:x2+small_w2] = right_image

                x = int(big_w/3*2)  # 指示条的左上角 x 坐标
                y = int(big_h/3)  # 指示条的左上角 y 坐标
                width = 100  # 指示条的宽度
                height = 300  # 指示条的高度
                color = (0, 255, 0)  # 指示条的颜色 (B, G, R)
                if ego_risk > 40 and ego_risk<= 70:
                    color =  (0, 165, 255)
                elif  ego_risk> 70 and ego_risk <= 100:
                    color = (0, 0, 250)
                else:
                    color = (0, 255, 0)

                # 计算填充区域的宽度
                percentage = ego_risk *(height/100)
                fill_height = int(percentage * width / 100)
                # 绘制背景
                cv2.rectangle(array, (x-width, y-height), (x, y), (100, 100, 100), -1)
                # 绘制填充
                cv2.rectangle(array, (x-width, y-fill_height), (x, y), color, -1)
                # 绘制边框
                cv2.rectangle(array, (x-width, y-height), (x, y), (0, 0, 0), thickness=1)
                cv2.putText(array, str(round(ego_risk,2)), (x-90, y+50), cv2.FONT_HERSHEY_COMPLEX, 0.9, color, 1)

                if hud_info.is_recording:
                    cv2.putText(array, 'is_recording', (50, 900), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 1)
                else:
                    cv2.putText(array, 'no_recording', (50, 900), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 255, 0), 1)
                behavior_type = hud_info.behavior_type
                scenario_id = hud_info.scenario_id
                config_name = hud_info.config_name
                cv2.putText(array, 'Behavior: '+ behavior_type + '   ' + str(scenario_id), (50, 950), cv2.FONT_HERSHEY_COMPLEX, 0.7, (250, 255, 250), 1)
                cv2.putText(array, 'Config:   ' + config_name, (50, 1000), cv2.FONT_HERSHEY_COMPLEX, 0.7, (250, 255, 250), 1)

                # 绘制3D 检测结果
                world = self._parent.get_world()
                camera_onboard_tf = self.sensor.get_transform()
                CAMERA_FOV = 130
                K_onboard = get_camera_intrinsic(self.hud.dim[0], self.hud.dim[1], CAMERA_FOV)
                print("Camera Onboard Transform:", camera_onboard_tf)
                # 画3D包围盒
                array, num_actors = draw_fixed_bbox_on_image(
                    array, world, camera_onboard_tf, K_onboard,
                    max_distance=100.0, ego_vehicle=hud_info.ego_vehicle, camera_fov_deg=CAMERA_FOV)
                # print(f"Number of selected actors: {num_actors}")


                array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)
    
            

            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('out/%08d' % image.frame)

class HUD_INFO(object):
    def __init__(self):
        self.ego_risk = 0.0
        self.is_recording = False
        self.config_name = "test"
        self.behavior_type = "test"
        self.scenario_id = 0
        self.left_mirror = np.full((500, 500, 3), 255, dtype=np.uint8)
        self.right_mirror = np.full((500, 500, 3), 255, dtype=np.uint8)
        self.camera_onboard_tf = None
        self.camera_onboard_K = None
        self.camera_onboard_fov = None
        self.ego_vehicle = None

    def is_end (self, cur_x, cur_y, end_point):
        is_end = False
        distance = math.sqrt( (cur_x-end_point['x'])**2 + (cur_y-end_point['y'])**2 )
        if distance < 10:
            is_end = True
        return is_end

# ==============================================================================================
# prediction
def get_prediction(cur_tf, vel, steer, horizon, dt):
    trajectory = []
    L = 4.0
    x0, y0, yaw0 = cur_tf.location.x, cur_tf.location.y, math.radians(cur_tf.rotation.yaw)
    # trajectory.append([x0, y0, yaw0, vel])
    trajectory.append([x0, y0, yaw0, vel])
    for idx in range(horizon):
        x1 = x0 + vel*math.cos(yaw0)*dt
        y1 = y0 + vel*math.sin(yaw0)*dt
        yaw1 = yaw0 + vel / L * math.tan(steer)*dt
        x0, y0, yaw0 = x1, y1, yaw1
        trajectory.append([x1, y1, yaw1, vel])

    return trajectory

def draw_trajectory(trajectory, ego_tf, world):
    idx = 5
    horizon = len(trajectory)
    while idx < horizon-1:
        x0 = trajectory[idx]
        x1 = trajectory[idx+1]
        begin = carla.Location(x=x0[0],y=x0[1],z=ego_tf.location.z+0.5)
        end = carla.Location(x=x1[0], y=x1[1], z=ego_tf.location.z+0.5)
        world.debug.draw_arrow(begin, end, thickness=0.02, arrow_size=0.5, color=carla.Color(0,60, 60),
                                    life_time=0.03)
        idx += 2
# ==============================================================================================


# ==============================================================================================
# init camera
def init_camera(world, sensor_type, transform, attached, image_with, image_hight, fov ):
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
        camera_sematic = world.spawn_actor(camera_bp, transform, attach_to=attached)
        return camera_sematic
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


def where_is_risk(img, angle, risk):
    height, width, channels = img.shape
    new_img = img.copy()

    center = (width//2, height//4*3)
    radius = min(width, height) // 6
    start_angle = 0
    end_angle = 60
    thickness = 30 + math.floor(risk/1)
    color1 = (0, 0, 255, 128)  # 红色,透明度 50%
    color2 = (0, 165, 255, 128)  # 橙色,透明度 50%

    cv2.ellipse(new_img, center, (radius, radius), 0, 15, 75, color1, thickness, cv2.LINE_AA)
    cv2.ellipse(new_img, center, (radius, radius), 0, 105, 165, color2, thickness, cv2.LINE_AA)
    cv2.ellipse(new_img, center, (radius, radius), 0, -15, -75, color2, thickness, cv2.LINE_AA)
    cv2.ellipse(new_img, center, (radius, radius), 0, -105, -165, color1, thickness, cv2.LINE_AA)

    x1, y1 = width//3, 100
    x2, y2 = width//3, 500
    cv2.line(new_img, (x1, y1), (x1+width//3, y1), (0, 250, 0), thickness, cv2.LINE_AA)
    # cv2.line(new_img, (x2, y2), (x2+300, y2), (0, 165, 255), thickness, cv2.LINE_AA)

    alpha = risk/100
    blended = cv2.addWeighted(img, 1-alpha, new_img, alpha, 0)

    return blended

# ==============================================================================================


# ==============================================================================
# -- game_loop() ---------------------------------------------------------------
# ==============================================================================
def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    diy_actor_list = []
    # ==============================================================================================
    # intiallize socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server_address = ('localhost', 12345)  
    # ==============================================================================================
    # Force Seat
    fsmi = ForceSeatMI()
    fsmi.activate_profile('SDK - Positioning')
    fsmi.begin_motion_control()

    position = FSMI_TopTablePositionPhysical()
    position.structSize = sizeof(FSMI_TopTablePositionPhysical)
    position.mask       = FSMI_TopTablePositionPhysicalMask()
    position.maxSpeed   = 10000
    # ==============================================================================================

    try:
        hud_info_queue = []

        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)
        display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        hud = HUD(args.width, args.height)
        world = World(client.get_world(), hud, args, hud_info_queue)
        controller = DualControl(world, args.autopilot)

        # SensorManager: RGBCamera, SemanticCamera
        IM_WIDTH = 640
        IM_HEIIGHT = 480
        QUEUE_LEN = 5
        # camera_top = init_camera(client.get_world(), 'RGBCamera', carla.Transform(carla.Location(x=5, y=0,z=15.0), 
        #                          carla.Rotation(pitch=-0)), world.player, 320, 240, 90 )
        # diy_actor_list.append(camera_top)
        # image_top_queue = []
        # camera_top.listen(lambda image: process_rgb_image(image, image_top_queue, QUEUE_LEN))
        # FRONT
        camera_front = init_camera(client.get_world(), 'RGBCamera', carla.Transform(carla.Location(x=2, y=0.3, z=1.9), 
                                 carla.Rotation(pitch=0)), world.player, IM_WIDTH, IM_HEIIGHT, 120 )
        diy_actor_list.append(camera_front)
        image_front_queue = []
        camera_front.listen(lambda image: process_rgb_image(image, image_front_queue, QUEUE_LEN))
        # LEFT
        camera_left = init_camera(client.get_world(), 'RGBCamera', carla.Transform(carla.Location(x=2, y=-2,z=1.5), 
                                 carla.Rotation(pitch=-0,yaw=-170)), world.player, 640, 480, 90 )
        diy_actor_list.append(camera_left)
        image_left_queue = []
        camera_left.listen(lambda image: process_rgb_image(image, image_left_queue, QUEUE_LEN))

        # RIGHT
        camera_right = init_camera(client.get_world(), 'RGBCamera', carla.Transform(carla.Location(x=2, y=2,z=1.5), 
                                 carla.Rotation(pitch=-0,yaw=-190)), world.player, 640, 480, 90 )
        diy_actor_list.append(camera_right)
        image_right_queue = []
        camera_right.listen(lambda image: process_rgb_image(image, image_right_queue, QUEUE_LEN))

        clock = pygame.time.Clock()
        while True:
            time_start = time.time()
            clock.tick_busy_loop(60)
            if controller.parse_events(world, clock):
                return

            ego_tf = world.player.get_transform()
            ego_acc =  world.player.get_acceleration()
            ego_angle_vel = world.player.get_angular_velocity()
            
            # ==============================================================================================
            # limit speed
            vel = world.player.get_velocity()
            ego_speed = math.sqrt(max(5, (vel.x ** 2 + vel.y ** 2 + vel.z ** 2)))
            limated_speed = 45.0 # km/h
            if ego_speed*3.6 > limated_speed:
                v_error = abs(ego_speed - limated_speed)
                k_p_break = 0.005
                k_p_throttle = 0.6
                out_put = v_error * k_p_break
                out_put = min(1.0, out_put)
                out_put = max(0.0, out_put)
                vehicle_control = world.player.get_control()
                vehicle_control.throttle = vehicle_control.throttle * k_p_throttle
                vehicle_control.brake  = out_put
                world.player.apply_control(vehicle_control)
            # ==============================================================================================
            # Force Seat
            max_angle  = math.radians(5)
            position.pitch = ego_tf.rotation.pitch * 0
            position.roll = -ego_tf.rotation.roll * 0.3
            if position.roll > max_angle:
                position.roll = max_angle
            if position.roll < -max_angle:
                position.roll = -max_angle
            fsmi.send_top_table_pos_phy(position)
            # ==============================================================================================
            # socket publish ego ID 
            message = str(world.player.id).encode()
            sock.sendto(message, server_address)
            # ==============================================================================================


            box = world.player.bounding_box
            box.location += ego_tf.location + carla.Location(x=0, y=0, z=3)
            box.extent = carla.Vector3D(1, 1, 1)
            # carla_word = client.get_world()
            # carla_word.debug.draw_box(box, ego_tf.rotation,thickness=0.005, life_time=0.05)

            # ===================================================================================
            # prediction
            speed_scale = 3
            steer_scale = 0.3
            horizon, dt = 6, 0.5
            vel = world.player.get_velocity()
            speed = math.sqrt(max(5, (vel.x ** 2 + vel.y ** 2 + vel.z ** 2)))
            steer = world.player.get_control().steer
            # ego_trajectory = get_prediction(ego_tf, speed*speed_scale, steer*steer_scale, horizon, dt)
            # draw_trajectory(ego_trajectory, ego_tf, client.get_world())


            # =================================== do something  =================================
            # if len(image_left_queue)>1:
            #     image_left = image_left_queue[-1]
            #     cv2.imshow("<<<<<--- LEFT", image_left)
            # if len(image_right_queue)>1:
            #     image_right = image_right_queue[-1]
            #     cv2.imshow("RIGHT --->>>>>", image_right)


            hud_info = HUD_INFO()
            hud_info.ego_risk = ego_speed*3.6
            hud_info.is_recording = world.enable_recording
            hud_info.behavior_type = world.scenario_config[world.scenario_idx]['behavior_type']
            hud_info.scenario_id = world.scenario_config[world.scenario_idx]['scenario_id']
            hud_info.config_name = args.config
            hud_info.ego_vehicle = world.player

            

            if len(image_left_queue)>1:
                image_left = image_left_queue[-1]
                image_left = cv2.flip(image_left, 1)  # 1 表示水平翻转
                hud_info.left_mirror = image_left
            if len(image_right_queue)>1:
                image_right = image_right_queue[-1]
                image_right = cv2.flip(image_right, 1)  # 1 表示水平翻转
                hud_info.right_mirror = image_right

            end_point = world.scenario_config[world.scenario_idx]['ego_end_point']
            if hud_info.is_end(ego_tf.location.x, ego_tf.location.y, end_point):
                world.enable_recording = False



            hud_info_queue.append(hud_info)
            if len(hud_info_queue) > 3:
                hud_info_queue.pop(0)
            # ===================================================================================



            current_time = datetime.datetime.now()
            time_str = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")
            time_end = time.time()
            print("vehicle_id: ", world.player.id, "   time:",time_str, '  time cost:', round(time_end-time_start,3))   
            # print(world.enable_recording)

            world.tick(clock)
            world.render(display)
            pygame.display.flip()
        


    finally:
        # 关闭套接字
        sock.close()
        # Force Seat
        fsmi.end_motion_control()
        fsmi.delete()

        cv2.destroyAllWindows()

        for actor in diy_actor_list:
            actor.destroy()

        if world is not None:
            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================
def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        # default='10.22.5.52',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='5120x1080',
        # default='3840x2160',
        # default='1920x1080',
        help='window resolution (default: 1920x1080)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--saving_gap',
        default=0.2,
        help='the gap from last saving time')
    argparser.add_argument(
        '--tester_name',
        default='ZHangSan',
        help='saving data path')
    argparser.add_argument(
        '--config',
        default='default_scenario.json',
        help='config name')


    args = argparser.parse_args()

    with open(args.config, 'r') as f:
        scenario_config = json.load(f)

    args.width, args.height = [int(x) for x in args.res.split('x')]

    args.scenario_config = scenario_config

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':

    main()
