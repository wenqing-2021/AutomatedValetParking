'''
Author: wenqing-hnu
Date: 2022-10-20
LastEditors: wenqing-hnu
LastEditTime: 2022-11-08
FilePath: /Automated Valet Parking/collision_check/collision_check.py
Description: collision check

Copyright (c) 2022 by wenqing-hnu, All Rights Reserved. 
'''


from abc import abstractmethod
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from map.costmap import Map, Vehicle


class collision_checker:
    def __init__(self,
                 map: Map,
                 vehicle: Vehicle = None,
                 config: dict = None) -> None:
        self.map = map
        self.config = config
        self.vehicle = vehicle

    def get_near_obstacles(self, node_x, node_y, theta) -> Tuple[list, np.array]:
        '''
        this function is only used for distance check method
        return the obstacles x and y, vehicle boundary
        Note: vehicle boundary is expanded
        '''

        # create_polygon
        vehicle_boundary = self.vehicle.create_anticlockpoint(
            x=node_x, y=node_y, theta=theta, config=self.config)

        '''
        right_rear = vehicle_boundary[0]
        right_front = vehicle_boundary[1]
        left_front = vehicle_boundary[2]
        left_rear = vehicle_boundary[3]
        note: these points have expanded
        '''

        # create AABB square
        x_max = max(vehicle_boundary[:, 0])
        x_min = min(vehicle_boundary[:, 0])
        y_max = max(vehicle_boundary[:, 1])
        y_min = min(vehicle_boundary[:, 1])

        # get obstacle position
        obstacle_index = np.where(self.map.cost_map == 255)
        obstacle_position_x = self.map.map_position[0][obstacle_index[0]]
        obstacle_position_y = self.map.map_position[1][obstacle_index[1]]

        # find those obstacles point in the AABB square
        near_x_position = obstacle_position_x[np.where(
            (obstacle_position_x >= x_min) & (obstacle_position_x <= x_max))]
        near_y_position = obstacle_position_y[np.where(
            (obstacle_position_x >= x_min) & (obstacle_position_x <= x_max))]

        # determine y
        near_obstacle_x = near_x_position[np.where(
            (near_y_position >= y_min) & (near_y_position <= y_max))]
        near_obstacle_y = near_y_position[np.where(
            (near_y_position >= y_min) & (near_y_position <= y_max))]

        near_obstacle_range = [near_obstacle_x, near_obstacle_y]

        return near_obstacle_range, vehicle_boundary

    @abstractmethod
    def check(self, node_x, node_y, theta) -> bool:
        pass


class two_circle_checker(collision_checker):
    '''
    use two circle to present car body for collision check
    '''

    def __init__(self, map: Map, vehicle: Vehicle = None, config: dict = None) -> None:
        super().__init__(map, vehicle, config)

    def check(self, node_x, node_y, theta) -> bool:
        v = self.vehicle

        # compute circle diameter
        Rd = 0.5 * np.sqrt(((v.lr+v.lw+v.lf)/2)**2 + (v.lb**2))
        # compute circle center position
        front_circle = (node_x+1/4*(3*v.lw+3*v.lf-v.lr)*np.cos(theta),
                        node_y+1/4*(3*v.lw+3*v.lf-v.lr)*np.sin(theta))
        rear_circle = (node_x+1/4*(v.lw+v.lf-3*v.lr)*np.cos(theta),
                       node_y+1/4*(v.lw+v.lf-3*v.lr)*np.sin(theta))

        # determine the AABB square of the two circles
        if front_circle[0] >= rear_circle[0]:
            right = front_circle[0] + Rd
            left = rear_circle[0] - Rd
        else:
            right = rear_circle[0] + Rd
            left = front_circle[0] - Rd

        if front_circle[1] >= rear_circle[1]:
            upper = front_circle[1] + Rd
            down = rear_circle[1] - Rd
        else:
            upper = rear_circle[1] + Rd
            down = front_circle[1] - Rd

        # get obstacle position
        obstacle_index = np.where(self.map.cost_map == 255)
        obstacle_position_x = self.map.map_position[0][obstacle_index[0]]
        obstacle_position_y = self.map.map_position[1][obstacle_index[1]]

        # determine x
        near_x_position = obstacle_position_x[np.where(
            (obstacle_position_x > left) & (obstacle_position_x < right))]
        near_y_position = obstacle_position_y[np.where(
            (obstacle_position_x > left) & (obstacle_position_x < right))]
        # determine y
        near_obstacle_x = near_x_position[np.where(
            (near_y_position > down) & (near_y_position < upper))]
        near_obstacle_y = near_y_position[np.where(
            (near_y_position > down) & (near_y_position < upper))]
        # check these points
        collision = False
        for x, y in zip(near_obstacle_x, near_obstacle_y):
            if np.sqrt(pow(x-front_circle[0], 2) + pow(y-front_circle[1], 2)) <= Rd:
                collision = True
            elif np.sqrt(pow(x-rear_circle[0], 2) + pow(y-rear_circle[1], 2)) <= Rd:
                collision = True

        return collision


class distance_checker(collision_checker):
    def __init__(self, map: Map, vehicle: Vehicle = None, config: dict = None) -> None:
        super().__init__(map, vehicle, config)

    def check(self, node_x, node_y, theta) -> bool:
        '''
        caculate the distance between obstacle point and vehicle boundary
        '''
        # compute the boundary straight line
        def compute_k_b(point_1, point_2):
            # k = (y_2 - y_1) / (x_2 - x_1)
            k = (point_2[1] - point_1[1]) / (point_2[0] - point_1[0])
            # b = y_1 - k * x_1
            b = point_1[1] - k * point_1[0]
            b_2 = point_2[1] - k * point_2[0]
            return k, b

        # compute the distance from the point to the line
        def compute_distance(_k, _b, point):
            dis = abs(_k * point[0] + _b - point[1]) / np.sqrt(1+pow(_k, 2))
            return dis

        near_obstacles_range, vehicle_boundary = self.get_near_obstacles(node_x=node_x, node_y=node_y,
                                                                         theta=theta)

        v_lb = np.sqrt(pow((vehicle_boundary[0, 0] - vehicle_boundary[3, 0]), 2) +
                       pow((vehicle_boundary[0, 1] - vehicle_boundary[3, 1]), 2))

        v_length = np.sqrt(pow((vehicle_boundary[3, 0] - vehicle_boundary[2, 0]), 2) +
                           pow((vehicle_boundary[3, 1] - vehicle_boundary[2, 1]), 2))

        # compute line k and b
        '''
        0: right line
        1: front line
        2: left line
        3: rear line
        '''
        line_k = []
        line_b = []
        for i in range(4):
            if i < 3:
                k_i, b_i = compute_k_b(
                    vehicle_boundary[i], vehicle_boundary[i+1])
                line_k.append(k_i)
                line_b.append(b_i)
            else:
                k_i, b_i = compute_k_b(
                    vehicle_boundary[i], vehicle_boundary[0])
                line_k.append(k_i)
                line_b.append(b_i)

        near_obstacle_x = near_obstacles_range[0]
        near_obstacle_y = near_obstacles_range[1]

        collision = False
        # collision check
        for x, y in zip(near_obstacle_x, near_obstacle_y):
            dis2rl = compute_distance(line_k[0], line_b[0], [x, y])
            dis2ll = compute_distance(line_k[2], line_b[2], [x, y])
            dis2fl = compute_distance(line_k[1], line_b[1], [x, y])
            dis2bl = compute_distance(line_k[3], line_b[3], [x, y])
            check_1 = True if abs(dis2rl - dis2ll) < v_lb - 0.01 else False
            check_2 = True if abs(dis2fl - dis2bl) < v_length - 0.01 else False

            # check point is in the rectangle
            if check_1 and check_2:
                collision = True
                break

            if collision == False:
                on_x = False
                on_y = False
                # check this point is on the corner
                for i in vehicle_boundary[:, 0]:
                    # check x
                    if x == i:
                        on_x = True
                        break

                if on_x:
                    for i in vehicle_boundary[:, 1]:
                        # check y
                        if y == i:
                            on_y = True
                            break

                if on_x and on_y:
                    # if the point on the corner
                    collision = True
                    break

            # check the point on the edge
            if collision == False:
                for i in range(4):
                    k1, _ = compute_k_b([x, y], vehicle_boundary[i])
                    if k1 == line_k[i]:
                        collision = True
                        break

        return collision

# def two_circle_check(node_x, node_y, theta, map: _map) -> bool:
#     '''
#     use two circle to present car body for collision check
#     '''
#     v = Vehicle()

#     # compute circle diameter
#     Rd = 0.5 * np.sqrt(((v.lr+v.lw+v.lf)/2)**2 + (v.lb**2))
#     # compute circle center position
#     front_circle = (node_x+1/4*(3*v.lw+3*v.lf-v.lr)*np.cos(theta),
#                     node_y+1/4*(3*v.lw+3*v.lf-v.lr)*np.sin(theta))
#     rear_circle = (node_x+1/4*(v.lw+v.lf-3*v.lr)*np.cos(theta),
#                    node_y+1/4*(v.lw+v.lf-3*v.lr)*np.sin(theta))

#     # determine the AABB square of the two circles
#     if front_circle[0] >= rear_circle[0]:
#         right = front_circle[0] + Rd
#         left = rear_circle[0] - Rd
#     else:
#         right = rear_circle[0] + Rd
#         left = front_circle[0] - Rd

#     if front_circle[1] >= rear_circle[1]:
#         upper = front_circle[1] + Rd
#         down = rear_circle[1] - Rd
#     else:
#         upper = rear_circle[1] + Rd
#         down = front_circle[1] - Rd

#     # get obstacle position
#     obstacle_index = np.where(map.cost_map == 255)
#     obstacle_position_x = map.map_position[0][obstacle_index[0]]
#     obstacle_position_y = map.map_position[1][obstacle_index[1]]

#     # determine x
#     near_x_position = obstacle_position_x[np.where(
#         (obstacle_position_x > left) & (obstacle_position_x < right))]
#     near_y_position = obstacle_position_y[np.where(
#         (obstacle_position_x > left) & (obstacle_position_x < right))]
#     # determine y
#     near_obstacle_x = near_x_position[np.where(
#         (near_y_position > down) & (near_y_position < upper))]
#     near_obstacle_y = near_y_position[np.where(
#         (near_y_position > down) & (near_y_position < upper))]
#     # check these points
#     collision = False
#     for x, y in zip(near_obstacle_x, near_obstacle_y):
#         if np.sqrt(pow(x-front_circle[0], 2) + pow(y-front_circle[1], 2)) <= Rd:
#             collision = True
#         elif np.sqrt(pow(x-rear_circle[0], 2) + pow(y-rear_circle[1], 2)) <= Rd:
#             collision = True

#     return collision


# def distance_check(node_x, node_y, theta, map: _map, config: dict = None) -> bool:
#     '''
#     caculate the distance between obstacle point and vehicle boundary
#     '''

#     # compute the boundary straight line
#     def compute_k_b(point_1, point_2):
#         # k = (y_2 - y_1) / (x_2 - x_1)
#         k = (point_2[1] - point_1[1]) / (point_2[0] - point_1[0])
#         # b = y_1 - k * x_1
#         b = point_1[1] - k * point_1[0]
#         b_2 = point_2[1] - k * point_2[0]
#         return k, b

#     # compute the distance from the point to the line
#     def compute_distance(_k, _b, point):
#         dis = abs(_k * point[0] + _b - point[1]) / np.sqrt(1+pow(_k, 2))
#         return dis

#     near_obstacles_range, vehicle_boundary = get_near_obstacles(node_x=node_x, node_y=node_y,
#                                                                 theta=theta, map=map, config=config)

#     v_lb = np.sqrt(pow((vehicle_boundary[0, 0] - vehicle_boundary[3, 0]), 2) +
#                    pow((vehicle_boundary[0, 1] - vehicle_boundary[3, 1]), 2))

#     v_length = np.sqrt(pow((vehicle_boundary[3, 0] - vehicle_boundary[2, 0]), 2) +
#                        pow((vehicle_boundary[3, 1] - vehicle_boundary[2, 1]), 2))

#     # compute line k and b
#     '''
#     0: right line
#     1: front line
#     2: left line
#     3: rear line
#     '''
#     line_k = []
#     line_b = []
#     for i in range(4):
#         if i < 3:
#             k_i, b_i = compute_k_b(vehicle_boundary[i], vehicle_boundary[i+1])
#             line_k.append(k_i)
#             line_b.append(b_i)
#         else:
#             k_i, b_i = compute_k_b(vehicle_boundary[i], vehicle_boundary[0])
#             line_k.append(k_i)
#             line_b.append(b_i)

#     near_obstacle_x = near_obstacles_range[0]
#     near_obstacle_y = near_obstacles_range[1]

#     collision = False
#     # collision check
#     for x, y in zip(near_obstacle_x, near_obstacle_y):
#         dis2rl = compute_distance(line_k[0], line_b[0], [x, y])
#         dis2ll = compute_distance(line_k[2], line_b[2], [x, y])
#         dis2fl = compute_distance(line_k[1], line_b[1], [x, y])
#         dis2bl = compute_distance(line_k[3], line_b[3], [x, y])
#         check_1 = True if abs(dis2rl - dis2ll) < v_lb - 0.01 else False
#         check_2 = True if abs(dis2fl - dis2bl) < v_length - 0.01 else False

#         # check point is in the rectangle
#         if check_1 and check_2:
#             collision = True
#             break

#         if collision == False:
#             on_x = False
#             on_y = False
#             # check this point is on the corner
#             for i in vehicle_boundary[:, 0]:
#                 # check x
#                 if x == i:
#                     on_x = True
#                     break

#             if on_x:
#                 for i in vehicle_boundary[:, 1]:
#                     # check y
#                     if y == i:
#                         on_y = True
#                         break

#             if on_x and on_y:
#                 # if the point on the corner
#                 collision = True
#                 break

#         # check the point on the edge
#         if collision == False:
#             for i in range(4):
#                 k1, _ = compute_k_b([x, y], vehicle_boundary[i])
#                 if k1 == line_k[i]:
#                     collision = True
#                     break

#     return collision
