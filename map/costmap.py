'''
Author: wenqing-hnu
Date: 2022-10-20
LastEditors: wenqing-hnu
LastEditTime: 2022-11-11
FilePath: /Automated Valet Parking/map/costmap.py
Description: generate cost map

Copyright (c) 2022 by wenqing-hnu, All Rights Reserved. 
'''


'''
thanks Bai Li provides the vehicle data and the map data: https://github.com/libai1943/TPCAP_demo_Python
BSD 2-Clause License

Copyright (c) 2022, Bai Li
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''




import string
import numpy as np
import math
import csv
import shapely.geometry
import matplotlib.pyplot as plt
class Vehicle:
    def __init__(self):
        self.lw = 2.8  # wheelbase
        self.lf = 0.96  # front hang length
        self.lr = 0.929  # rear hang length
        self.lb = 1.942  # width
        self.max_steering_angle = 0.75  # rad
        self.max_angular_velocity = 0.5  # rad/s
        self.max_acc = 1  # m/s^2
        self.max_v = 2.5  # m/s
        self.min_v = -2.5  # m/s
        self.min_radius_turn = self.lw / \
            np.tan(self.max_steering_angle) + self.lb / 2  # m

    def create_polygon(self, x, y, theta):
        '''
        right back, right front, left front, left back, right back
        '''
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)

        points = np.array([
            [-self.lr, -self.lb / 2, 1],
            [self.lf + self.lw, -self.lb / 2, 1],
            [self.lf + self.lw, self.lb / 2, 1],
            [-self.lr, self.lb / 2, 1],
            [-self.lr, -self.lb / 2, 1],
        ]).dot(np.array([
            [cos_theta, -sin_theta, x],
            [sin_theta, cos_theta, y],
            [0, 0, 1]
        ]).transpose())
        return points[:, 0:2]

    def create_anticlockpoint(self, x, y, theta, config: dict = None):
        '''
        Note: this function will expand this vehicle square
        '''
        # transform matrix
        trans_matrix = np.array([[np.cos(theta), np.sin(theta)],
                                 [-np.sin(theta), np.cos(theta)]])

        # local point and expand this square
        # expand the collision check box
        side_dis = config['safe_side_dis']  # m
        fr_dis = config['safe_fr_dis']  # m
        right_rear = np.array([[-self.lr-fr_dis], [-self.lb/2-side_dis]])
        right_front = np.array(
            [[self.lw+self.lf+fr_dis], [-self.lb/2-side_dis]])
        left_front = np.array([[self.lw+self.lf+fr_dis], [self.lb/2+side_dis]])
        left_rear = np.array([[-self.lr-fr_dis], [self.lb/2+side_dis]])

        # original coordinate position
        # inverse of trans_matrix equals to transpose of trans_matrix
        points = []
        rr_point = trans_matrix.transpose().dot(
            right_rear) + np.array([[x], [y]])
        rf_point = trans_matrix.transpose().dot(
            right_front) + np.array([[x], [y]])
        lf_point = trans_matrix.transpose().dot(
            left_front) + np.array([[x], [y]])
        lr_point = trans_matrix.transpose().dot(left_rear) + \
            np.array([[x], [y]])

        points.append([rr_point[0], rr_point[1]])
        points.append([rf_point[0], rf_point[1]])
        points.append([lf_point[0], lf_point[1]])
        points.append([lr_point[0], lr_point[1]])
        points.append([rr_point[0], rr_point[1]])

        return np.array(points)


class Case:
    def __init__(self):
        self.x0, self.y0, self.theta0 = 0, 0, 0
        self.xf, self.yf, self.thetaf = 0, 0, 0
        self.xmin, self.xmax = 0, 0
        self.ymin, self.ymax = 0, 0
        self.obs_num = 0
        self.obs = np.array([])
        self.vehicle = Vehicle()

    @staticmethod
    def read(file):
        case = Case()
        with open(file, 'r') as f:
            reader = csv.reader(f)
            tmp = list(reader)
            v = [float(i) for i in tmp[0]]
            case.x0, case.y0, case.theta0 = v[0:3]
            case.xf, case.yf, case.thetaf = v[3:6]
            case.xmin = min(case.x0, case.xf) - 12
            case.xmax = max(case.x0, case.xf) + 12
            case.ymin = min(case.y0, case.yf) - 12
            case.ymax = max(case.y0, case.yf) + 12

            case.obs_num = int(v[6])
            num_vertexes = np.array(v[7:7 + case.obs_num], dtype=np.int32)
            vertex_start = 7 + case.obs_num + \
                (np.cumsum(num_vertexes, dtype=np.int32) - num_vertexes) * 2
            case.obs = []
            for vs, nv in zip(vertex_start, num_vertexes):
                case.obs.append(
                    np.array(v[vs:vs + nv * 2]).reshape((nv, 2), order='A'))
        return case


class Map:
    def __init__(self,
                 discrete_size: np.float64 = 0.1,
                 file: string = None) -> None:
        self.discrete_size = discrete_size
        self.grid_index = None  # index of each grid
        self.cost_map = np.array([], dtype=np.float64)  # cost value
        self.map_position = np.array([], dtype=np.float64)  # (x,y) value
        self.case = Case.read(file)
        # math.floor: return the largest integer not greater than x
        self.boundary = np.array([math.floor(self.case.xmin),
                                  math.floor(self.case.xmax),
                                  math.floor(self.case.ymin),
                                  math.floor(self.case.ymax)], dtype=np.float64)
        # self.detect_obstacle()
        self._discrete_x = 0
        self._discrete_y = 0
        self.detect_obstacle_edge()

    def discrete_map(self):
        '''
        param: case data is obtained from the csv file
        '''
        x_index = int(
            (self.boundary[1] - self.boundary[0]) / self.discrete_size)
        y_index = int(
            (self.boundary[3] - self.boundary[2]) / self.discrete_size)
        self.cost_map = np.zeros((x_index, y_index), dtype=np.float64)
        # create (x,y) position
        dx_position = np.linspace(self.boundary[0], self.boundary[1], x_index)
        dy_position = np.linspace(self.boundary[2], self.boundary[3], y_index)
        self._discrete_x = dx_position[1] - dx_position[0]
        self._discrete_y = dy_position[1] - dy_position[0]
        # the position of each point in the park map
        self.map_position = (dx_position, dy_position)
        # create grid index
        self.grid_index_max = x_index*y_index

    def detect_obstacle_edge(self):
        # just consider the boundary of the obstacles
        # discrete map
        self.discrete_map()

        # get obstacles edge
        for i in range(0, self.case.obs_num):
            old_obstacle = self.case.obs[i]
            # delete redundant points
            obstacle = np.unique(old_obstacle, axis=0)
            obstacle_point_num = len(obstacle[:, 0])
            # sort the polygan point by counterclockwise direction
            # get the centerpoint
            center_x = np.mean(obstacle[:, 0])
            center_y = np.mean(obstacle[:, 1])

            delta_x = obstacle[:, 0] - center_x
            delta_y = obstacle[:, 1] - center_y
            angle = np.arctan2(delta_y, delta_x) + np.pi
            obstacle = obstacle[np.argsort(angle)]  # sort the obstacle points

            for j in range(obstacle_point_num):
                obstacle_p1 = [obstacle[j, 0], obstacle[j, 1]]
                if j+1 == obstacle_point_num:
                    obstacle_p2 = [obstacle[0, 0], obstacle[0, 1]]
                else:
                    obstacle_p2 = [obstacle[j+1, 0], obstacle[j+1, 1]]

                # get rotate angle
                vector_1 = [obstacle_p2[0]-obstacle_p1[0],
                            obstacle_p2[1]-obstacle_p1[1]]

                rotate_angle = np.arctan2(vector_1[1], vector_1[0])

                rotation_matrix = np.array([[np.cos(rotate_angle), np.sin(rotate_angle)],
                                            [-np.sin(rotate_angle), np.cos(rotate_angle)]])

                translate_matrix = np.array(vector_1).reshape([2, 1])

                new_obstacle_p2 = np.dot(rotation_matrix, translate_matrix)[
                    0].tolist()

                # get positions of points on the edge
                points_num = math.floor(
                    new_obstacle_p2[0] / self._discrete_x)
                points_y = np.zeros(points_num)
                points_x = np.linspace(0, new_obstacle_p2[0], points_num)
                points_position = np.vstack((points_x, points_y))

                _points_position = np.dot(
                    rotation_matrix.transpose(), points_position)

                for k in range(points_num):
                    original_points_position = [_points_position[0][k]+obstacle_p1[0],
                                                _points_position[1][k]+obstacle_p1[1]]

                    points_x_index = np.where((self.map_position[0] < original_points_position[0]) &
                                              (self.map_position[0] > (original_points_position[0]-self._discrete_x)))

                    points_y_index = np.where((self.map_position[1] < original_points_position[1]) &
                                              (self.map_position[1] > (original_points_position[1]-self._discrete_y)))

                    if any(points_x_index) and any(points_y_index):
                        self.cost_map[int(points_x_index[0])
                                      ][int(points_y_index[0])] = 255

    def detect_obstacle(self):
        # discrete map
        self.discrete_map()

        for i in range(0, self.case.obs_num):
            obstacle = self.case.obs[i]
            # get the rectangle of the obstancle
            obstacle_xmin, obstacle_xmax = np.min(
                obstacle[:, 0]), np.max(obstacle[:, 0])
            obstacle_ymin, obstacle_ymax = np.min(
                obstacle[:, 1]), np.max(obstacle[:, 1])
            # find map points in the rectangle
            near_obs_x_index = np.where((self.map_position[0] >= obstacle_xmin) & (
                self.map_position[0] <= obstacle_xmax))
            near_obs_y_index = np.where((self.map_position[1] >= obstacle_ymin) & (
                self.map_position[1] <= obstacle_ymax))
            # determine the near points is in the obstacle or not
            # create polygon
            poly_shape = shapely.geometry.Polygon(obstacle)
            # generate potints
            points_x = self.map_position[0]
            points_y = self.map_position[1]
            for i in near_obs_x_index[0]:
                for j in near_obs_y_index[0]:
                    # print(points_x[i], points_y[j])
                    point = shapely.geometry.Point(points_x[i], points_y[j])
                    if poly_shape.intersects(point):
                        # point in the obstacl, set the cost = 255
                        if self.cost_map[i][j] != 255:
                            self.cost_map[i][j] = 255

        # print(self.cost_map.shape)

    def visual_cost_map(self):
        plt.figure(1)
        for i in range(len(self.map_position[0])):
            for j in range(len(self.map_position[1])):
                if self.cost_map[i][j] == 255:
                    plt.plot(
                        self.map_position[0][i], self.map_position[1][j], 'x', color='k')
        plt.xlim(self.case.xmin, self.case.xmax)
        plt.ylim(self.case.ymin, self.case.ymax)
        plt.draw()
        # plt.show()
        # print('ok')

    def visual_near_vehicle_map(self, xmin, xmax, ymin, ymax):
        plt.figure(1)
        for i in range(len(self.map_position[0])):
            for j in range(len(self.map_position[1])):
                if self.map_position[0][i] >= xmin and self.map_position[0][i] <= xmax and self.map_position[1][j] >= ymin and self.map_position[1][j] <= ymax:
                    plt.plot(
                        self.map_position[0][i], self.map_position[1][j], 'x', color='k')
        plt.xlim(self.case.xmin, self.case.xmax)
        plt.ylim(self.case.ymin, self.case.ymax)

    def convert_position_to_index(self,
                                  grid_x: np.float64,
                                  grid_y: np.float64):
        '''
        param: the upper right corner of the grid position
        return: the index of this grid, its range is from 1 to x_index*y_index
        '''
        index_0 = math.floor((grid_x - self.boundary[0]) / self._discrete_x)
        index_1 = math.floor((self.boundary[3] - grid_y) / self._discrete_y) * (
            int((self.boundary[1] - self.boundary[0]) / self._discrete_x))
        return index_0 + index_1
