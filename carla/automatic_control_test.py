#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Example of automatic vehicle control from client side."""

from __future__ import print_function

import argparse
import collections
import datetime
import glob
import logging
import math
import os
import random
import re
import sys
import weakref
import math
import pandas as pd
import cvxpy as cp
import jax
import jax.numpy as jnp
import haiku as hk
import pickle
from core.dynamics.carla_4state import CarlaDynamics
import matplotlib.pyplot as plt
from ctrl import *

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_TAB
    from pygame.locals import K_q
    from pygame.locals import K_r
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

PATH = './trained_cbf.npy'
PI = 3.1415926

# DELTA_F = 0.3
# DELTA_G = 0.4

# CTE_MAX = 1.6175946161054822
# SPEED_MAX = 7.285775632710312
# THETA_E_MAX = 2.9999982774423595
# D_MAX = 26.73716521658956
# DTHETA_T_MAX = 0.8976939936192778
# INPUT_MAX = 0.23236677428018998

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
from carla import ColorConverter as cc

from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from agents.navigation.roaming_agent import RoamingAgent  # pylint: disable=import-error
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================


def find_weather_presets():
    """Method to find weather presets"""
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    """Method to get actor display name"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name


# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================

class World(object):
    """ Class representing the surrounding environment """

    def __init__(self, carla_world, hud, args, x, y, theta):
        """Constructor method"""
        self.world = carla_world
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = args.filter
        self._gamma = args.gamma
        self.restart(args, x, y, theta)
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0

    def restart(self, args, x, y, theta):
        """Restart the world"""
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_id = self.camera_manager.transform_index if self.camera_manager is not None else 0
        # Set the seed if requested by user
        if args.seed is not None:
            random.seed(args.seed)

        # Get a random blueprint.
        # blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint = self.world.get_blueprint_library().filter(self._actor_filter)[0]
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        # Spawn the player.
        print("Spawning the player")
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)

        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            # spawn_points = self.map.get_spawn_points()
            # spawn_point = carla.Transform(carla.Location(x=-120.7, y=149.3, z=2.0), carla.Rotation(yaw=180))
            spawn_point = carla.Transform(carla.Location(x=x, y=y, z=2.0), carla.Rotation(yaw=theta))
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        self.camera_manager.transform_index = cam_pos_id
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        """Get next weather setting"""
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def tick(self, clock):
        """Method for every tick"""
        self.hud.tick(self, clock)

    def render(self, display):
        """Render world"""
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        """Destroy sensors"""
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        """Destroys all actors"""
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    def __init__(self, world):
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, client, world, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()

    @staticmethod
    def _is_quit_shortcut(key):
        """Shortcut for quitting"""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    """Class for HUD text"""

    def __init__(self, width, height):
        """Constructor method"""
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        """Gets informations from the world at every tick"""
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        """HUD method for every tick"""
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        transform = world.player.get_transform()
        vel = world.player.get_velocity()
        control = world.player.get_control()
        heading = 'N' if abs(transform.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(transform.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > transform.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > transform.rotation.yaw > -179.5 else ''
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
            'Map:     % 20s' % world.map.name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (transform.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (transform.location.x, transform.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % transform.location.z,
            '']
        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', control.throttle, 0.0, 1.0),
                ('Steer:', control.steer, -1.0, 1.0),
                ('Brake:', control.brake, 0.0, 1.0),
                ('Reverse:', control.reverse),
                ('Hand brake:', control.hand_brake),
                ('Manual:', control.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(control.gear, control.gear)]
        elif isinstance(control, carla.WalkerControl):
            self._info_text += [
                ('Speed:', control.speed, 0.0, 5.556),
                ('Jump:', control.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]

        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']

        def dist(l):
            return math.sqrt((l.x - transform.location.x)**2 + (l.y - transform.location.y)
                             ** 2 + (l.z - transform.location.z)**2)
        vehicles = [(dist(x.get_location()), x) for x in vehicles if x.id != world.player.id]

        for dist, vehicle in sorted(vehicles):
            if dist > 200.0:
                break
            vehicle_type = get_actor_display_name(vehicle, truncate=22)
            self._info_text.append('% 4dm %s' % (dist, vehicle_type))

    def toggle_info(self):
        """Toggle info on or off"""
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        """Notification text"""
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        """Error text"""
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        """Render for HUD class"""
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1 - y) * 30) for x, y in enumerate(item)]
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
                        fig = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + fig * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (fig * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    """ Class for fading text """

    def __init__(self, font, dim, pos):
        """Constructor method"""
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        """Set fading text"""
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        """Fading text method for every tick"""
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        """Render fading text method"""
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """ Helper class for text render"""

    def __init__(self, font, width, height):
        """Constructor method"""
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for i, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, i * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        """Toggle on or off the render help"""
        self._render = not self._render

    def render(self, display):
        """Render help text method"""
        if self._render:
            display.blit(self.surface, self.pos)

# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    """ Class for collision sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        """Gets the history of collisions"""
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        """On collision method"""
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)

# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    """Class for lane invasion sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
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
        """On invasion method"""
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
    """ Class for GNSS sensors"""

    def __init__(self, parent_actor):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        """GNSS method"""
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    """ Class for camera management"""

    def __init__(self, parent_actor, hud, gamma_correction):
        """Constructor method"""
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(
                carla.Location(x=-5.5, z=2.5), carla.Rotation(pitch=8.0)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=1.6, z=1.7)), attachment.Rigid),
            (carla.Transform(
                carla.Location(x=5.5, y=1.5, z=1.5)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=-1, y=-bound_y, z=0.5)), attachment.Rigid)]
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
            blp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                blp.set_attribute('image_size_x', str(hud.dim[0]))
                blp.set_attribute('image_size_y', str(hud.dim[1]))
                if blp.has_attribute('gamma'):
                    blp.set_attribute('gamma', str(gamma_correction))
            elif item[0].startswith('sensor.lidar'):
                blp.set_attribute('range', '50')
            item.append(blp)
        self.index = None

    def toggle_camera(self):
        """Activate a camera"""
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        """Set a sensor"""
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (
            force_respawn or (self.sensors[index][0] != self.sensors[self.index][0]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])

            # We need to pass the lambda a weak reference to
            # self to avoid circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        """Get the next sensor"""
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        """Toggle recording on or off"""
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        """Render method"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=assignment-from-no-return
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
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)

# ==============================================================================
# -- Game Loop ---------------------------------------------------------
# ==============================================================================


def game_loop(args):
    """ Main loop for agent"""

    tot_target_reached = 0
    num_min_waypoints = 21
    initial_positions = np.load('initial_positions.npz')
    initial_position_array = initial_positions['arr_0']
    print(initial_position_array.shape)
    npzfile = np.load("map.npz")
    x_map = npzfile['arr_0']
    y_map = npzfile['arr_1']
    waypoints_map = npzfile['arr_2']
    list_cte = []
    list_theta_e = []
    list_dot_phit = []
    list_steer = []
    list_h = []
    list_h_dire = []
    list_g = []

    # loop for 107 starting points
    for i in range(12, 107):
        pygame.init()
        pygame.font.init()
        world = None
        exported_data = []
        try:
            client = carla.Client(args.host, args.port)
            client.set_timeout(4.0)

            display = pygame.display.set_mode(
                (args.width, args.height),
                pygame.HWSURFACE | pygame.DOUBLEBUF)

            hud = HUD(args.width, args.height)
            # world = World(client.get_world(), hud, args)
            world = World(client.load_world('Town06'), hud, args,
                          initial_position_array[0, i], initial_position_array[1, i], initial_position_array[2, i])
            controller = KeyboardControl(world)

            if args.agent == "Roaming":
                agent = RoamingAgent(world.player)
            # elif args.agent == "Basic":
            #     agent = BasicAgent(world.player)
            #     spawn_point = world.map.get_spawn_points()[0]
            #     agent.set_destination((spawn_point.location.x,
            #                            spawn_point.location.y,
            #                            spawn_point.location.z))
            # else:
            #     agent = BehaviorAgent(world.player, behavior=args.behavior)

            #     spawn_points = world.map.get_spawn_points()
            #     random.shuffle(spawn_points)

            #     if spawn_points[0].location != agent.vehicle.get_location():
            #         destination = spawn_points[0].location
            #     else:
            #         destination = spawn_points[1].location

            #     agent.set_destination(agent.vehicle.get_location(), destination, clean=True)

            clock = pygame.time.Clock()

            while True and len(exported_data) <= 1487:
                clock.tick_busy_loop(60)
                if controller.parse_events(client, world, clock):
                    return

                # As soon as the server is ready continue!
                if not world.world.wait_for_tick(10.0):
                    continue

                if args.agent == "Roaming" or args.agent == "Basic":
                    if controller.parse_events(client, world, clock):
                        return

                    # as soon as the server is ready continue!
                    world.world.wait_for_tick(10.0)

                    world.tick(clock)
                    world.render(display)
                    pygame.display.flip()
                    control, _ie = agent.run_step()

                    control.manual_gear_shift = True
                    control.gear = 3
                    control.brake = 0.0
                    location = world.player.get_transform()
                    velocity = world.player.get_velocity()
                    x_rear_wheel = location.location.x - 1.26 * math.cos(location.rotation.yaw / 180 * PI)
                    y_rear_wheel = location.location.y - 1.26 * math.sin(location.rotation.yaw / 180 * PI)
                    theta = location.rotation.yaw / 180 * PI

                    # incorporate the learned controller
                    v = np.sqrt(velocity.x * velocity.x + velocity.y * velocity.y)
                    d = -_ie
                    distance = np.linalg.norm(waypoints_map - np.array([x_rear_wheel, y_rear_wheel]), axis=1)
                    index_tp = np.argmin(distance)
                    # calculate theta_e and cte
                    if index_tp < x_map.shape[0] - 1:
                        waypoint_vector_x = x_map[index_tp + 1] - x_map[index_tp]
                        waypoint_vector_y = y_map[index_tp + 1] - y_map[index_tp]
                        theta_waypoint = np.arctan2(waypoint_vector_y, waypoint_vector_x)
                        theta_e = theta - theta_waypoint
                        start_waypoint_to_car_location_x = x_rear_wheel - x_map[index_tp]
                        start_waypoint_to_car_location_y = y_rear_wheel - y_map[index_tp]
                        area = -(start_waypoint_to_car_location_x * waypoint_vector_y -
                                 start_waypoint_to_car_location_y * waypoint_vector_x)
                        cte = area / np.linalg.norm(np.array([waypoint_vector_x, waypoint_vector_y]), axis=0)
                    else:
                        waypoint_vector_x = x_map[index_tp] - x_map[index_tp - 1]
                        waypoint_vector_y = y_map[index_tp] - y_map[index_tp - 1]
                        theta_waypoint = np.arctan2(waypoint_vector_y, waypoint_vector_x)
                        theta_e = theta - theta_waypoint
                        start_waypoint_to_car_location_x = x_rear_wheel - x_map[index_tp]
                        start_waypoint_to_car_location_y = y_rear_wheel - y_map[index_tp]
                        area = -(start_waypoint_to_car_location_x * waypoint_vector_y -
                                 start_waypoint_to_car_location_y * waypoint_vector_x)
                        cte = area / np.linalg.norm(np.array([waypoint_vector_x, waypoint_vector_y]), axis=0)
                    while theta_e > 1.5:
                        theta_e -= PI
                    while theta_e < -1.5:
                        theta_e += PI
                    # calculate dotphi_t
                    if index_tp < x_map.shape[0] - 2:
                        waypoint_vector_x = x_map[index_tp + 1] - x_map[index_tp]
                        waypoint_vector_y = y_map[index_tp + 1] - y_map[index_tp]
                        next_waypoint_vector_x = x_map[index_tp + 2] - x_map[index_tp + 1]
                        next_waypoint_vector_y = y_map[index_tp + 2] - y_map[index_tp + 1]
                    elif index_tp < x_map.shape[0] - 1:
                        waypoint_vector_x = x_map[index_tp + 1] - x_map[index_tp]
                        waypoint_vector_y = y_map[index_tp + 1] - y_map[index_tp]
                        next_waypoint_vector_x = x_map[0] - x_map[index_tp + 1]
                        next_waypoint_vector_y = y_map[0] - y_map[index_tp + 1]
                    else:
                        waypoint_vector_x = x_map[0] - x_map[index_tp]
                        waypoint_vector_y = y_map[0] - y_map[index_tp]
                        next_waypoint_vector_x = x_map[1] - x_map[0]
                        next_waypoint_vector_y = y_map[1] - y_map[0]
                    diff_theta_waypoint = np.arctan2(
                        next_waypoint_vector_y, next_waypoint_vector_x) - np.arctan2(waypoint_vector_y, waypoint_vector_x)
                    while diff_theta_waypoint > 1.5:
                        diff_theta_waypoint -= PI
                    while diff_theta_waypoint < -1.5:
                        diff_theta_waypoint += PI
                    dot_phi_t = diff_theta_waypoint * v / \
                        np.linalg.norm(np.array([waypoint_vector_x, waypoint_vector_y]))
                    # get the control from the learned controller
                    net = hk.without_apply_rng(hk.transform(lambda x: net_fn()(x)))
                    with open(PATH, 'rb') as handle:
                        loaded_params = pickle.load(handle)

                    def learned_h(x): return jnp.sum(net.apply(loaded_params, x))
                    zero_ctrl = get_zero_controller()
                    safe_ctrl = make_safe_controller(zero_ctrl, learned_h)
                    # steering_wheel_input, h, h_dire, g = safe_ctrl(jnp.array([cte, v, theta_e, d]), dot_phi_t)
                    steering_wheel_input, h = safe_ctrl(jnp.array([cte, v, theta_e, d]), dot_phi_t)
                    control.steer = np.arctan(steering_wheel_input[0]) / 70 * 180 / PI
                    list_cte.append(cte)
                    list_theta_e.append(theta_e)
                    list_dot_phit.append(dot_phi_t)
                    list_steer.append(control.steer)
                    list_h.append(h)
                    # list_h_dire.append(h_dire)
                    # list_g.append(g)

                    # last_control = world.player.get_control()
                    # acceleration = world.player.get_acceleration()
                    # target_speed = agent._local_planner._target_speed
                    # target_waypoint = agent._local_planner.target_waypoint
                    # # x_rear_wheel = (world.player.get_physics_control(
                    # # ).wheels[2].position.x + world.player.get_physics_control().wheels[3].position.x) / 200
                    # # y_rear_wheel = (world.player.get_physics_control(
                    # # ).wheels[2].position.y + world.player.get_physics_control().wheels[3].position.y) / 200
                    # exported_data.append([pygame.time.get_ticks() / 1000, acceleration.x, acceleration.y, acceleration.z, velocity.x, velocity.y, velocity.z, x_rear_wheel, y_rear_wheel, location.location.x, location.location.y,
                    #                       location.location.z, location.rotation.yaw / 180 *
                    #                       PI, target_speed, target_waypoint.transform.location.x, target_waypoint.transform.location.y, target_waypoint.transform.location.z,
                    #                       last_control.throttle, last_control.brake, last_control.steer * 70 / 180 *
                    #                       PI, control.throttle, control.brake, control.steer *
                    #                       70 / 180 * PI, np.tan(control.steer * 70 / 180 * PI),
                    #                       np.sqrt(velocity.x * velocity.x + velocity.y * velocity.y), np.sqrt(acceleration.x * acceleration.x + acceleration.y * acceleration.y), -_ie])
                    world.player.apply_control(control)
                else:
                    agent.update_information()

                    world.tick(clock)
                    world.render(display)
                    pygame.display.flip()

                    # Set new destination when target has been reached
                    if len(agent.get_local_planner().waypoints_queue) < num_min_waypoints and args.loop:
                        agent.reroute(spawn_points)
                        tot_target_reached += 1
                        world.hud.notification("The target has been reached " +
                                               str(tot_target_reached) + " times.", seconds=4.0)

                    elif len(agent.get_local_planner().waypoints_queue) == 0 and not args.loop:
                        print("Target reached, mission accomplished...")
                        break

                    speed_limit = world.player.get_speed_limit()
                    agent.get_local_planner().set_speed(speed_limit)

                    control = agent.run_step()
                    world.player.apply_control(control)

        finally:
            print(list_steer)
            plt.figure()
            plt.plot(list_cte)
            plt.plot(list_theta_e)
            plt.plot(list_dot_phit)
            plt.plot(list_steer)
            plt.plot(list_h)
            # plt.plot(list_h_dire)
            # plt.plot(list_g)
            # plt.legend(("cte", "theta_e", "dot_phi_t", "steer", "h", "h_derivative", "g"))
            plt.legend(("cte", "theta_e", "dot_phi_t", "steer", "h"))
            plt.show()
            if world is not None:
                world.destroy()

            # print("saving recorded data" + str(i + 1) + ":")
            # exported_data = np.array(exported_data)
            # df = pd.DataFrame(data=exported_data, columns=['Ticks(s)', 'x-acc(m/s^2)', 'y-acc(m/s^2)', 'z-acc(m/s^2)', 'x-vel(m/s)', 'y-vel(m/s)', 'z-vel(m/s)', 'x-loc(m)', 'y-loc(m)', 'x-loc-center(m)', 'y-loc-center(m)', 'z-loc-center(m)',
            #                                                'theta(radians)', 'target-speed(m/s)', 'target-x-loc(m)', 'target-y-loc(m)', 'target-z-loc(m)', 'past-throttle', 'past_brake', 'past-delta(radians)', 'throttle', 'brake', 'delta(radians)', 'input', 'speed(m/s)', 'acceleration(m/s^2)', 'd'])
            # df.to_pickle('_out/Data_Collection_noisy_controller_' + str(i + 1) + '.pd')

            pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


def main():
    """Main method"""

    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
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
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='Actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--gamma',
        default=2.2,
        type=float,
        help='Gamma correction of the camera (default: 2.2)')
    argparser.add_argument(
        '-l', '--loop',
        action='store_true',
        dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)')
    argparser.add_argument(
        '-b', '--behavior', type=str,
        choices=["cautious", "normal", "aggressive"],
        help='Choose one of the possible agent behaviors (default: normal) ',
        default='normal')
    argparser.add_argument("-a", "--agent", type=str,
                           choices=["Behavior", "Roaming", "Basic"],
                           help="select which agent to run",
                           default="Behavior")
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


# def make_safe_controller(nominal_ctrl, h):
#     """Create a safe controller using learned hybrid CBF."""

#     dh = jax.grad(h, argnums=0)
#     dyn = CarlaDynamics()

#     def safe_ctrl(x, d):
#         """Solves HCBF-QP to map an input state to a safe action u.

#         Params:
#             x: state.
#             d: disturbance.
#         """

#         # compute action used by nominal controller
#         u_nom = nominal_ctrl(x)

#         # compute function values
#         f_of_x, g_of_x = dyn.f(x, d), dyn.g(x)
#         h_of_x = h(x)
#         dh_of_x = dh(x)

#         # setup and solve HCBF-QP with CVXPY
#         u_mod = cp.Variable(len(u_nom))
#         obj = cp.Minimize(cp.sum_squares(u_mod - u_nom))
#         constraints = [jnp.dot(dh_of_x, f_of_x) + u_mod.T @ jnp.dot(g_of_x.T, dh_of_x) + h_of_x >= 0]
#         prob = cp.Problem(obj, constraints)
#         prob.solve(solver=cp.SCS, verbose=False, max_iters=20000, eps=1e-10)

#         if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
#             return u_mod.value, h_of_x, np.linalg.norm(dh_of_x), np.linalg.norm(g_of_x)
#         return jnp.array([0.]), h_of_x, np.linalg.norm(dh_of_x), np.linalg.norm(g_of_x)

#     return safe_ctrl


# def net_fn(net_dims=[32, 16]):
#     """Feed-forward NN architecture."""

#     layers = []
#     for dim in net_dims:
#         layers.extend([hk.Linear(dim), jnp.tanh])
#     layers.append(hk.Linear(1))

#     return hk.Sequential(layers)


# def get_zero_controller():
#     """Returns a zero controller"""

#     return lambda state: jnp.array([0.])

# def make_safe_controller(nominal_ctrl, h):
#     """Create a safe controller using learned hybrid CBF."""

#     dh = jax.grad(h, argnums=0)
#     dyn = CarlaDynamics()
#     def alpha(x): return x
#     def norm(x): return jnp.linalg.norm(x)
#     def cpnorm(x): return cp.norm(x)
#     def dot(x, y): return jnp.dot(x, y)

#     def safe_ctrl(x, d):
#         """Solves HCBF-QP to map an input state to a safe action u.

#         Params:
#             x: state.
#             d: disturbance.
#         """

#         cte, v, θ_e, d_var = x
#         x = jnp.array([
#             cte / CTE_MAX,
#             v / SPEED_MAX,
#             θ_e / THETA_E_MAX,
#             d_var / D_MAX
#         ]).reshape(x.shape)

#         d /= DTHETA_T_MAX

#         # compute action used by nominal controller
#         u_nom = nominal_ctrl(x)

#         # setup and solve HCBF-QP with CVXPY
#         u_mod = cp.Variable(len(u_nom))
#         obj = cp.Minimize(cp.sum_squares(u_mod - u_nom))
#         constraints = [
#             dot(dh(x), dyn.f(x, d)) + u_mod.T @ dot(dyn.g(x).T, dh(x)) +
#             alpha(h(x)) - norm(dh(x)) * (DELTA_F + DELTA_G * cpnorm(u_mod)) >= 0
#         ]

#         prob = cp.Problem(obj, constraints)
#         prob.solve(solver=cp.SCS, verbose=True, max_iters=20000, eps=1e-10)

#         if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
#             return u_mod.value
#         return jnp.array([0.])

#     return safe_ctrl


# def net_fn(net_dims=[32, 16]):
#     """Feed-forward NN architecture."""

#     layers = []
#     for dim in net_dims:
#         layers.extend([hk.Linear(dim), jnp.tanh])
#     layers.append(hk.Linear(1))

#     return hk.Sequential(layers)


# def get_zero_controller():
#     """Returns a zero controller"""

#     return lambda state: jnp.array([0.])


if __name__ == '__main__':
    main()
