# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
from numpy.random import choice
import random
from isaacgym import terrain_utils
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg
from .custom_terrain import *


class Terrain:
    def __init__(self, cfg: LeggedRobotCfg.terrain, num_robots) -> None:

        self.cfg = cfg
        self.num_robots = num_robots
        self.type = cfg.mesh_type
        if self.type in ["none", 'plane']:
            return
        self.env_length = cfg.terrain_length # 10
        self.env_width = cfg.terrain_width # 10
        self.proportions = [np.sum(cfg.terrain_proportions[:i+1]) for i in range(len(cfg.terrain_proportions))]

        self.cfg.num_sub_terrains = cfg.num_rows * cfg.num_cols
        self.env_origins = np.zeros((cfg.num_rows, cfg.num_cols, 3))

        self.width_per_env_pixels = int(self.env_width / cfg.horizontal_scale) # 10/0.1
        self.length_per_env_pixels = int(self.env_length / cfg.horizontal_scale) # 10/0.1

        self.border = int(cfg.border_size/self.cfg.horizontal_scale) # 2.5/0.1
        self.tot_cols = int(cfg.num_cols * self.width_per_env_pixels) + 2 * self.border
        self.tot_rows = int(cfg.num_rows * self.length_per_env_pixels) + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows , self.tot_cols), dtype=np.int16)
        
        if cfg.curriculum:
            self.curriculum_terrain()
        else:    
            self.randomized_terrain()   
        
        self.heightsamples = self.height_field_raw
        if self.type=="trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(   self.height_field_raw,
                                                                                            self.cfg.horizontal_scale,
                                                                                            self.cfg.vertical_scale,
                                                                                            self.cfg.slope_treshold)
    def poisson_disk_sampling(self, width, height, r, k=30):
        """Poisson Disk Sampling using Bridson's algorithm.
        
        Parameters:
        width, height: Dimensions of the region to sample points.
        r: Minimum distance between points.
        k: Maximum number of samples before rejection (default 30).
        
        Returns:
        A list of sampled points [(x1, y1), (x2, y2), ...].
        """
        cell_size = r / np.sqrt(2)
        grid_width = int(np.ceil(width / cell_size))
        grid_height = int(np.ceil(height / cell_size))
        
        grid = [[None for _ in range(grid_height)] for _ in range(grid_width)]
        process_list = []
        sample_points = []
        
        def get_grid_coords(point):
            return int(point[0] // cell_size), int(point[1] // cell_size)
        
        def in_neighbourhood(point):
            gx, gy = get_grid_coords(point)
            for i in range(max(0, gx - 2), min(grid_width, gx + 3)):
                for j in range(max(0, gy - 2), min(grid_height, gy + 3)):
                    neighbor = grid[i][j]
                    if neighbor is not None:
                        dist = np.linalg.norm(np.array(point) - np.array(neighbor))
                        if dist < r:
                            return True
            return False
        
        # Initialize by picking a random point
        first_point = (random.uniform(0, width), random.uniform(0, height))
        process_list.append(first_point)
        sample_points.append(first_point)
        grid[get_grid_coords(first_point)[0]][get_grid_coords(first_point)[1]] = first_point
        
        while process_list:
            point = process_list.pop(random.randint(0, len(process_list) - 1))
            for _ in range(k):
                angle = random.uniform(0, 2 * np.pi)
                radius = random.uniform(r, 2 * r)
                new_point = (point[0] + radius * np.cos(angle), point[1] + radius * np.sin(angle))
                
                if 0 <= new_point[0] < width and 0 <= new_point[1] < height and not in_neighbourhood(new_point):
                    process_list.append(new_point)
                    sample_points.append(new_point)
                    grid[get_grid_coords(new_point)[0]][get_grid_coords(new_point)[1]] = new_point
        
        return sample_points

    def randomized_terrain(self):
        proportions = np.array(self.cfg.terrain_proportions) / np.sum(self.cfg.terrain_proportions)
        for k in range(self.cfg.num_sub_terrains):
            print('generating randomized terrains %d / %d     '%(k, self.cfg.num_sub_terrains), end='\r')
            # Env coordinates in the world
            (i, j) = np.unravel_index(k, (self.cfg.num_rows, self.cfg.num_cols))

            terrain_type = np.random.choice(self.cfg.terrain_types, p=proportions)
            difficulty = np.random.choice([0.5, 0.75, 0.9])
            terrain = self.make_terrain(terrain_type, difficulty)
            self.add_terrain_to_map(terrain, i, j)
        print('\n generated all randomized terrains!')
        
    def curiculum(self):
        for j in range(self.cfg.num_cols):
            for i in range(self.cfg.num_rows):
                difficulty = i / self.cfg.num_rows
                choice = j / self.cfg.num_cols + 0.001

                terrain = self.make_terrain(choice, difficulty)
                self.add_terrain_to_map(terrain, i, j)
    
    def curriculum_terrain(self):
        proportions = np.array(self.cfg.terrain_proportions) / np.sum(self.cfg.terrain_proportions)
        already_taken_porp = 0.0
        start_col = 0
        end_col = 0
        sub_terrain_dict = {}
        for ter in range(len(self.cfg.terrain_types)):
            terrain_type = self.cfg.terrain_types[ter]
            start_col = end_col + 0
            already_taken_porp += proportions[ter]
            while end_col + 0.1 < self.cfg.num_cols * already_taken_porp: end_col += 1
            sub_terrain_dict[terrain_type] = (start_col, end_col)
            print(terrain_type, 'col:',start_col,':', end_col)

        for terrain_type, col_range in sub_terrain_dict.items():
            print('generating curriculum terrains %s    '%(terrain_type), end='\r')
            start_col = col_range[0]
            end_col = col_range[1]
            for j in range(start_col, end_col):
                for i in range(self.cfg.num_rows):
                    difficulty = i / self.cfg.num_rows
                    terrain = self.make_terrain(terrain_type, difficulty)
                    self.add_terrain_to_map(terrain, i, j)
        print('\n generated all curriculum terrains!')
    
    def make_terrain(self, terrain_type, difficulty):
        terrain = terrain_utils.SubTerrain(   "terrain",
                                width=self.width_per_env_pixels,
                                length=self.width_per_env_pixels,
                                vertical_scale=self.cfg.vertical_scale,
                                horizontal_scale=self.cfg.horizontal_scale)

        terrain_func = getattr(self, terrain_type+'_terrain_func')
        terrain_func(terrain, difficulty)

        return terrain

    def add_terrain_to_map(self, terrain, row, col):
        i = row
        j = col
        # Map coordinate system
        start_x = self.border + i * self.length_per_env_pixels # = env_length / horizontal_scale
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        self.height_field_raw[start_x: end_x, start_y:end_y] = terrain.height_field_raw

        env_origin_x = (i + 0.5) * self.env_length # *10
        env_origin_y = (j + 0.5) * self.env_width
        x1 = int((self.env_length/2. - 0.5) / terrain.horizontal_scale)
        x2 = int((self.env_length/2. + 0.5) / terrain.horizontal_scale)
        y1 = int((self.env_width/2. - 0.5) / terrain.horizontal_scale)
        y2 = int((self.env_width/2. + 0.5) / terrain.horizontal_scale)
        env_origin_z = np.max(terrain.height_field_raw[x1:x2, y1:y2])*terrain.vertical_scale
        self.env_origins[i, j] = [env_origin_x, env_origin_y, env_origin_z]

    def select_room(self, row, col):
        i = int(row)
        j = int(col)
        start_x = self.border + i * self.length_per_env_pixels
        end_x = self.border + (i + 1) * self.length_per_env_pixels
        start_y = self.border + j * self.width_per_env_pixels
        end_y = self.border + (j + 1) * self.width_per_env_pixels
        scaled_room = self.height_field_raw[start_x: end_x, start_y:end_y] * self.cfg.vertical_scale
        return scaled_room

    def gap_terrain_func(self, terrain, difficulty):
        gap_size = 1 * difficulty
        platform_size=3.
        gap_size = int(gap_size / terrain.horizontal_scale)
        platform_size = int(platform_size / terrain.horizontal_scale)

        center_x = terrain.length // 2
        center_y = terrain.width // 2
        x1 = (terrain.length - platform_size) // 2
        x2 = x1 + gap_size
        y1 = (terrain.width - platform_size) // 2
        y2 = y1 + gap_size
    
        terrain.height_field_raw[center_x-x2 : center_x + x2, center_y-y2 : center_y + y2] = -1000
        terrain.height_field_raw[center_x-x1 : center_x + x1, center_y-y1 : center_y + y1] = 0

    def pit_terrain_func(self, terrain, difficulty):
        depth = 1 * difficulty
        platform_size=4.
        depth = int(depth / terrain.vertical_scale)
        platform_size = int(platform_size / terrain.horizontal_scale / 2)
        x1 = terrain.length // 2 - platform_size
        x2 = terrain.length // 2 + platform_size
        y1 = terrain.width // 2 - platform_size
        y2 = terrain.width // 2 + platform_size
        terrain.height_field_raw[x1:x2, y1:y2] = -depth

    def flat_terrain_func(self, terrain, difficulty):
        terrain.height_field_raw[:] = 0.

    def rough_terrain_func(self, terrain, difficulty):
        max_height = 0.035 * difficulty / 0.9
        terrain.height_field_raw = np.random.uniform(-max_height*2-0.02, -0.02, terrain.height_field_raw.shape) / terrain.vertical_scale

    def low_obst_terrain_func(self, terrain, difficulty):
        max_height = 0.06 * difficulty / 0.9
        obst_size = terrain.width // 10
        obst_num = 30
        xs = np.random.randint(0, terrain.length-obst_size, (obst_num,)) 
        ys = np.random.randint(0, terrain.width-obst_size, (obst_num,)) 
        terrain.height_field_raw[:] = 0.
        for i in range(obst_num):
            terrain.height_field_raw[xs[i]:xs[i]+obst_size, ys[i]:ys[i]+obst_size] = -max_height / terrain.vertical_scale

    def maze_terrain_func(self, terrain, difficulty):
        terrain.height_field_raw[:] = 1.0 / terrain.vertical_scale
        path_width = int((1.61 - difficulty * 1.0) / terrain.horizontal_scale)
        room_size = int(1.51 / terrain.horizontal_scale/2)
        midroom_size = int(2.01 / terrain.horizontal_scale/2) + path_width//2
        center_x = terrain.length // 2
        center_y = terrain.width // 2
        
        y_low = np.random.randint(-center_y, center_y-path_width, terrain.length)
        y_high = np.random.randint(-center_y, center_y-path_width, terrain.length)
        y_low, y_high = np.minimum(y_low, y_high), np.maximum(y_low, y_high) + path_width
        y_low[center_x-midroom_size:center_x+midroom_size] = - midroom_size
        y_high[center_x-midroom_size:center_x+midroom_size] = + midroom_size
        y_low[-room_size:] =  - room_size
        y_high[-room_size:] = + room_size
        y_low[:room_size] = - room_size
        y_high[:room_size] = + room_size
        for _col in range(0,terrain.length,path_width):
            if _col > path_width-1:
                if y_high[_col] < y_low[_col-path_width] + path_width: y_high[_col] = y_low[_col-path_width] + path_width
                if y_low[_col] > y_high[_col-path_width] - path_width: y_low[_col] = y_high[_col-path_width] - path_width
            terrain.height_field_raw[_col:_col+path_width, center_y+y_low[_col]:center_y+y_high[_col]] = 0.
        terrain.height_field_raw[ :room_size, center_y-room_size:center_y+room_size] = 0.
        terrain.height_field_raw[-room_size:, center_y-room_size:center_y+room_size] = 0.
        terrain.height_field_raw[ room_size:room_size+path_width, 2:-2] = 0.
        terrain.height_field_raw[-room_size-path_width:-room_size, 2:-2] = 0.

    def room_terrain_func(self, terrain, difficulty):
        t = int(3.0/terrain.horizontal_scale)  # Partition offset from border (3.0)
        s = int(4.0/terrain.horizontal_scale)  # Partition length (4.0)
        d = int(0.5/terrain.horizontal_scale)  # Wall/partition thickness
        wall_height = int(1. / terrain.vertical_scale)
        # Plane
        terrain.height_field_raw[:] = 0.

        num_obst = max(int(difficulty*20), 5)
        offset = 4.0
        ori_x = 0.5 * self.env_length # *10
        ori_y = 0.5 * self.env_width

        min_dist = 1.6 / terrain.horizontal_scale
        terrain_length = self.env_length / terrain.horizontal_scale

        obst_pos = self.poisson_disk_sampling(terrain_length, terrain_length, min_dist) 
        obst_pos = np.round(obst_pos).astype(int) 

        if len(obst_pos) > num_obst:
            rand_idx = np.random.choice(len(obst_pos), num_obst, replace=False)
            obst_pos = obst_pos[rand_idx]
        
        upper_left = np.array([ori_x-offset, ori_y-offset]) / terrain.horizontal_scale
        upper_right = np.array([ori_x+offset, ori_y-offset]) / terrain.horizontal_scale
        lower_left = np.array([ori_x-offset, ori_y+offset]) / terrain.horizontal_scale
        lower_right = np.array([ori_x+offset, ori_y+offset]) / terrain.horizontal_scale
        
        corners = np.stack((upper_left, upper_right, lower_left, lower_right), axis=0)
        
        rand_pos = np.random.randn(2) / terrain.horizontal_scale
        rand_pos = np.round(rand_pos).astype(int)

        for k in range(obst_pos.shape[0]):
            for i in range(4):
                while np.linalg.norm(obst_pos[k] - corners[i]) < (0.5 / terrain.horizontal_scale):
                    obst_pos[k] += rand_pos

        # Walls
        terrain.height_field_raw[:, :d] = wall_height 
        terrain.height_field_raw[:, -d:] = wall_height 
        terrain.height_field_raw[:d, :] = wall_height  
        terrain.height_field_raw[-d:, :] = wall_height  
        # Partitions
        terrain.height_field_raw[t:t+d, d:d+s] = wall_height 
        terrain.height_field_raw[d:d+s, -t-d:-t] = wall_height 
        terrain.height_field_raw[-d-s:-d, t:t+d] = wall_height 
        terrain.height_field_raw[-t-d:-t, -d-s:-d] = wall_height 

    def clean_room_terrain_func(self, terrain, difficulty):
        clean_room = create_rand_room(0, grid_size=20, target_size=self.length_per_env_pixels, min_distance=2, set_pos=False)
        terrain.height_field_raw = clean_room * int(1. / terrain.vertical_scale)

    def easy_room_terrain_func(self, terrain, difficulty):
        easy_room = create_rand_room(3, grid_size=20, target_size=self.length_per_env_pixels, min_distance=2, set_pos=False)
        terrain.height_field_raw = easy_room * int(1. / terrain.vertical_scale)
        
    def middle_room_terrain_func(self, terrain, difficulty):
        middle_room = create_rand_room(6, grid_size=20, target_size=self.length_per_env_pixels, min_distance=2, set_pos=False)
        terrain.height_field_raw = middle_room * int(1. / terrain.vertical_scale)

    def hard_room_terrain_func(self, terrain, difficulty):
        hard_room = create_rand_room(9, grid_size=20, target_size=self.length_per_env_pixels, min_distance=2, set_pos=False) # 100*100, obstacle height 0.2m ~ 1.0m
        terrain.height_field_raw = hard_room * int(1. / terrain.vertical_scale) # terrain.vertical_scale = 0.005, so height_field_raw = hard_room * 200 

    def hard_room_rough_terrain_func(self, terrain, difficulty):
        """Keep the original hard-room layout and add low-amplitude roughness."""
        hard_room = create_rand_room(
            9, grid_size=20, target_size=self.length_per_env_pixels,
            min_distance=2, set_pos=False,
        )
        obstacle_height = hard_room * int(1. / terrain.vertical_scale)
        roughness_peak_m = 0.035
        rough_height = np.random.uniform(
            -roughness_peak_m, roughness_peak_m,
            terrain.height_field_raw.shape,
        ) / terrain.vertical_scale
        terrain.height_field_raw = np.rint(
            obstacle_height + rough_height
        ).astype(np.int16)
