'''
Author: wenqing-hnu
Date: 2022-10-20
LastEditors: wenqing-hnu
LastEditTime: 2022-10-28
FilePath: /TPCAP_demo_Python-main/interpolation/cubic_interpolation.py
Description: interpolation more points on the curve

Copyright (c) 2022 by wenqing-hnu, All Rights Reserved.
'''


import math
from typing import List, Dict, Tuple
import numpy as np
from scipy.linalg import solve
from scipy import spatial
from path_planner.rs_curve import pi_2_pi
from animation.animation import *
from util_math.spline import spine
from util_math.coordinate_transform import coordinate_transform


class interpolation:
    '''
    this class is used for the interpolation, and return the initial solution, including
    x, y, theta, velocity, acceleration, delta, w, time
    '''

    def __init__(self,
                 vehicle,
                 map,
                 config: dict) -> None:
        self.map = map
        self.insert_dt = config["velocity_plan_dt"]
        self.vehicle = vehicle

    def cubic_interpolation(self,
                            path: list,
                            path_i_info: dict,
                            v_a_func,
                            forward: bool = None) -> list:
        '''
        description:
        path: the interporlation path
        path_i_info: dict including the cubic function_list and the rotation matrix_list
        v_a_func: the velocity function and the acceleration function
        return {*} the interpolation function
        '''

        # update the theta of waypoints
        t = 0
        insert_path = []
        cubic_func_list = path_i_info['cubic_list']
        rotation_matrix_list = path_i_info['rotation_matrix_list']
        new_end_list = path_i_info['new_end_list']

        _, a = v_a_func(0)
        if forward:
            direction = 1
        else:
            direction = -1

        trans_path = [[0, 0, 0, 0, a, 0]]  # the first path point
        first_node = True
        for i in range(len(path) - 1):
            cubic_func = cubic_func_list[i]
            new_end = new_end_list[i]
            # insert points in the transformed accordinate
            while True:
                t += self.insert_dt
                v, a = v_a_func(t)
                if first_node:
                    insert_x = trans_path[-1][0] + v * direction * \
                        self.insert_dt * np.cos(trans_path[-1][2])
                    first_node = False
                else:
                    insert_x = trans_path[-1][0] + trans_path[-1][3] * direction * \
                        self.insert_dt * np.cos(trans_path[-1][2])
                    # compute the inserted point

                if insert_x >= new_end[0]:
                    # compute the rest time
                    t_previous = (new_end[0] - trans_path[-1][0]) / \
                        (trans_path[-1][3]*direction*np.cos(trans_path[-1][2]))
                    rest_time = t - t_previous
                    rest_x = insert_x - new_end[0]
                    break
                else:
                    insert_y, insert_theta = cubic_func(insert_x)
                    trans_path.append(
                        [insert_x, insert_y, insert_theta, v, a, t])

            # store the insert points in the transformed path list
            # assume the initial point has the velocity, we need it to compute the steering angle
            # trans_path = [[0, 0, 0, v, acc_func(t), t]]
            # delta_dis = v * self.insert_dt * math.cos(trans_path[-1][2])

            # add transformed end points

        # inverse transform these insert points
        invers_trans_path = coordinate_transform.inverse_trans(
            trans_path, rotation_matrix, start=start)
        if i > 0:
            insert_path.extend(invers_trans_path[1:])
        else:
            insert_path.extend(invers_trans_path)

        # compute steering angle and check the theta
        for i in range(len(insert_path) - 1):
            steering_angle = math.atan((insert_path[i+1][2] - insert_path[i][2]) * self.vehicle.lw /
                                       (insert_path[i][3] * (insert_path[i+1][-1] - insert_path[i][-1])))
            steering_angle = pi_2_pi(steering_angle)
            insert_path[i].insert(5, steering_angle)
            if i > 0:
                delta_steering_angle = insert_path[i][5] - insert_path[i-1][5]
                delta_time = insert_path[i][6] - insert_path[i-1][6]
                omega = delta_steering_angle / delta_time
                insert_path[i-1].insert(6, omega)

        # make sure the velocity, acceleration, steering angle is zero at the change gear point
        insert_path[-1][3] = 0  # velocity
        insert_path[-1][4] = 0  # acceleration
        # keep the steering angle of last point is the same as the previous point
        insert_path[-1].insert(5, insert_path[-2][-2])
        insert_path[-1].insert(6, 0)  # omega
        omega = (insert_path[-1][5] - insert_path[-2][5]) / \
                (insert_path[-1][-1] - insert_path[-2][-1])
        insert_path[-2].insert(6, omega)

        # plot_final_path(path=insert_path, map=self.map, color='blue')
        return insert_path

    def cubic_fitting(self,
                      path: List[List] = None) -> Tuple[np.float64, Dict]:
        '''
        description: the path is the splited path
        return {*} the arc length of this period path and the path info, 
        including the rotation matrix and the cubic function
        '''
        cubic_func_list = []
        rotation_matrix_list = []
        arc_lenth_list = []
        new_end_list = []
        start_point = path[0]
        arc_lenth = 0
        for i in range(1, len(path)):
            end_point = path[i]
            cubic_func, rotation_matrix, new_end = spine.cubic_spline(
                start=start_point, end=end_point)
            arc_lenth_i = spine.Simpson_integral(cubic_func, [0, 0], new_end)

            cubic_func_list.append(cubic_func)
            rotation_matrix_list.append(rotation_matrix)
            new_end_list.append(new_end)
            arc_lenth_list.append(arc_lenth_i)

            arc_lenth += arc_lenth_i

            start_point = end_point

        path_i_info = {'cubic_list': cubic_func_list,
                       'rotation_matrix_list': rotation_matrix_list,
                       'arc_len_list': arc_lenth_list,
                       'new_end_list': new_end_list}

        return arc_lenth, path_i_info
