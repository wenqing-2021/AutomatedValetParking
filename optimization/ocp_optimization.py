'''
Author: wenqing-hnu
Date: 2022-10-20
LastEditors: wenqing-hnu
LastEditTime: 2022-11-06
FilePath: /Automated Valet Parking/optimization/ocp_optimization.py
Description: use ipopt to solve the optimization problem

Copyright (c) 2022 by wenqing-hnu, All Rights Reserved. 
'''


from __future__ import division
from map.costmap import _map, Vehicle
import math
import copy
import numpy as np

import pyomo.environ as pyo
solver_path = 'optimization/ipopt'

# wheel base
Lw = 2.8


class ocp_optimization:
    def __init__(self,
                 park_map: _map,
                 vehicle: Vehicle,
                 config: dict) -> None:
        self.config = config
        self.map = park_map
        self.vehicle = vehicle
        self.expand_dis = config['expand_dis']  # m

    def compute_collision_H(self, path):
        '''
        use AABB block to find those map points near the vehicle
        and then find the shortest distance from these points to
        the vehicle square. noted as [f_d, b_d, r_d, l_d]
        f_d is the shortest distance from obstacles to the front edge
        b_d is to the rear edge, and r_d is to the right edge, l_d is
        to the left edge.
        [E;-E] X <= [H_max;-H_min]
        return: x_max, y_max, x_min, y_min
        '''

        # get near obstacles and vehicle
        def get_near_obstacles(node_x, node_y, theta, map: _map, config):
            '''
            this function is only used for distance check method
            return the obstacles x and y, vehicle boundary
            Note: vehicle boundary is expanded
            '''

            # create vehicle boundary
            v = Vehicle()

            # create_polygon
            vehicle_boundary = v.create_anticlockpoint(
                x=node_x, y=node_y, theta=theta, config=config)

            '''
            right_rear = vehicle_boundary[0]
            right_front = vehicle_boundary[1]
            left_front = vehicle_boundary[2]
            left_rear = vehicle_boundary[3]
            note: these points have expanded
            '''

            # create AABB squaref
            x_max = max(vehicle_boundary[:, 0]) + self.expand_dis
            x_min = min(vehicle_boundary[:, 0]) - self.expand_dis
            y_max = max(vehicle_boundary[:, 1]) + self.expand_dis
            y_min = min(vehicle_boundary[:, 1]) - self.expand_dis

            # get obstacle position
            obstacle_index = np.where(map.cost_map == 255)
            obstacle_position_x = map.map_position[0][obstacle_index[0]]
            obstacle_position_y = map.map_position[1][obstacle_index[1]]

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

        # compute the parameters of boundary line
        def compute_k_b(point_1, point_2):
            # k = (y_2 - y_1) / (x_2 - x_1)
            k = (point_2[1] - point_1[1]) / (point_2[0] - point_1[0])
            # b = y_1 - k * x_1
            b = point_1[1] - k * point_1[0]
            b_2 = point_2[1] - k * point_2[0]
            return k, b

        def get_area_boundary(point1, point2):
            area_x_min = min(point1[0], point2[0])
            area_x_max = max(point1[0], point2[0])
            area_y_min = min(point1[1], point2[1])
            area_y_max = max(point1[1], point2[1])
            return [area_x_min, area_x_max, area_y_min, area_y_max]

        def compute_distance(_k, _b, point):
            dis = abs(_k * point[0] + _b - point[1]) / np.sqrt(1+pow(_k, 2))
            return dis

        # compute the distance from the point to the line
        def compute_hori_ver_dis(point, k, b, theta):
            shortest_dis = compute_distance(_k=k, _b=b, point=point)
            vertical_dis = shortest_dis / abs(np.cos(theta))
            horizon_dis = shortest_dis / abs(np.sin(theta))
            return float(horizon_dis), float(vertical_dis)

        original_path = path
        X_max = []
        Y_max = []
        X_min = []
        Y_min = []
        # create AABB boundary and get the near obstacles position
        for p in original_path:
            x, y, theta = p[0], p[1], p[2]
            near_obstacles_range, vehicle_boundary = get_near_obstacles(node_x=x, node_y=y,
                                                                        theta=theta, map=self.map, config=self.config)
            # compute k and b
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

            # compute the obstacles points in which area
            # there are four situations about the car heading
            if theta >= -math.pi and theta < -math.pi / 2:
                case = 3
            elif theta >= -math.pi/2 and theta < 0:
                case = 4
            elif theta >= 0 and theta < math.pi / 2:
                case = 1
            elif theta >= math.pi/2 and theta <= math.pi:
                case = 2

            x_min, x_max = self.expand_dis, self.expand_dis
            y_min, y_max = self.expand_dis, self.expand_dis
            # compute each the boundary of each area
            each_area_boundary = []
            '''
            right area --- 0
            front area --- 1
            left area --- 2
            rear area --- 3
            '''
            for i in range(4):
                if i < 3:
                    _area = get_area_boundary(
                        vehicle_boundary[i], vehicle_boundary[i+1])
                else:
                    _area = get_area_boundary(
                        vehicle_boundary[i], vehicle_boundary[0])

                each_area_boundary.append(_area)

            # total 4 cases and four areas in each case to compute x,y boundary
            # theta is in [0, pi/2)
            if case == 1:
                for obs_x, obs_y in zip(near_obstacle_x, near_obstacle_y):
                    k = 0

                    for area in each_area_boundary:
                        '''
                        k = 0: right line
                        k = 1: front line
                        k = 2: left line
                        k = 3: rear line
                        '''
                        if k == 0:
                            obs_x_min = area[0]
                            obs_x_max = area[1] + self.expand_dis
                            obs_y_min = area[2] - self.expand_dis
                            obs_y_max = area[3]
                            if obs_x > obs_x_min and obs_x < obs_x_max and obs_y > obs_y_min and obs_y < obs_y_max:
                                hori_dis, ver_dis = compute_hori_ver_dis(
                                    (obs_x, obs_y), line_k[k], line_b[k], theta)
                                if hori_dis < x_max:
                                    x_max = hori_dis
                                if ver_dis < y_min:
                                    y_min = ver_dis
                                break

                        if k == 1:
                            obs_x_min = area[0]
                            obs_x_max = area[1] + self.expand_dis
                            obs_y_min = area[2]
                            obs_y_max = area[3] + self.expand_dis
                            if obs_x > obs_x_min and obs_x < obs_x_max and obs_y > obs_y_min and obs_y < obs_y_max:
                                hori_dis, ver_dis = compute_hori_ver_dis(
                                    (obs_x, obs_y), line_k[k], line_b[k], theta)
                                if hori_dis < x_max:
                                    x_max = hori_dis
                                if ver_dis < y_max:
                                    y_max = ver_dis
                                break

                        if k == 2:
                            obs_x_min = area[0] - self.expand_dis
                            obs_x_max = area[1]
                            obs_y_min = area[2]
                            obs_y_max = area[3] + self.expand_dis
                            if obs_x > obs_x_min and obs_x < obs_x_max and obs_y > obs_y_min and obs_y < obs_y_max:
                                hori_dis, ver_dis = compute_hori_ver_dis(
                                    (obs_x, obs_y), line_k[k], line_b[k], theta)
                                if hori_dis < x_min:
                                    x_min = hori_dis
                                if ver_dis < y_max:
                                    y_max = ver_dis
                                break

                        if k == 3:
                            obs_x_min = area[0] - self.expand_dis
                            obs_x_max = area[1]
                            obs_y_min = area[2] - self.expand_dis
                            obs_y_max = area[3]
                            if obs_x > obs_x_min and obs_x < obs_x_max and obs_y > obs_y_min and obs_y < obs_y_max:
                                hori_dis, ver_dis = compute_hori_ver_dis(
                                    (obs_x, obs_y), line_k[k], line_b[k], theta)
                                if hori_dis < x_min:
                                    x_min = hori_dis
                                if ver_dis < y_min:
                                    y_min = ver_dis
                                break

                        k += 1
            # theta is in [pi/2, pi]
            elif case == 2:
                for obs_x, obs_y in zip(near_obstacle_x, near_obstacle_y):
                    k = 0

                    for area in each_area_boundary:
                        '''
                        k = 0: right line
                        k = 1: front line
                        k = 2: left line
                        k = 3: rear line
                        '''
                        if k == 0:
                            obs_x_min = area[0]
                            obs_x_max = area[1] + self.expand_dis
                            obs_y_min = area[2]
                            obs_y_max = area[3] + self.expand_dis
                            if obs_x > obs_x_min and obs_x < obs_x_max and obs_y > obs_y_min and obs_y < obs_y_max:
                                hori_dis, ver_dis = compute_hori_ver_dis(
                                    (obs_x, obs_y), line_k[k], line_b[k], theta)
                                if hori_dis < x_max:
                                    x_max = hori_dis
                                if ver_dis < y_max:
                                    y_max = ver_dis
                                break

                        if k == 1:
                            obs_x_min = area[0] - self.expand_dis
                            obs_x_max = area[1]
                            obs_y_min = area[2]
                            obs_y_max = area[3] + self.expand_dis
                            if obs_x > obs_x_min and obs_x < obs_x_max and obs_y > obs_y_min and obs_y < obs_y_max:
                                hori_dis, ver_dis = compute_hori_ver_dis(
                                    (obs_x, obs_y), line_k[k], line_b[k], theta)
                                if hori_dis < x_min:
                                    x_min = hori_dis
                                if ver_dis < y_max:
                                    y_max = ver_dis
                                break

                        if k == 2:
                            obs_x_min = area[0] - self.expand_dis
                            obs_x_max = area[1]
                            obs_y_min = area[2] - self.expand_dis
                            obs_y_max = area[3]
                            if obs_x > obs_x_min and obs_x < obs_x_max and obs_y > obs_y_min and obs_y < obs_y_max:
                                hori_dis, ver_dis = compute_hori_ver_dis(
                                    (obs_x, obs_y), line_k[k], line_b[k], theta)
                                if hori_dis < x_min:
                                    x_min = hori_dis
                                if ver_dis < y_min:
                                    y_min = ver_dis
                                break

                        if k == 3:
                            obs_x_min = area[0]
                            obs_x_max = area[1] + self.expand_dis
                            obs_y_min = area[2] - self.expand_dis
                            obs_y_max = area[3]
                            if obs_x > obs_x_min and obs_x < obs_x_max and obs_y > obs_y_min and obs_y < obs_y_max:
                                hori_dis, ver_dis = compute_hori_ver_dis(
                                    (obs_x, obs_y), line_k[k], line_b[k], theta)
                                if hori_dis < x_max:
                                    x_max = hori_dis
                                if ver_dis < y_min:
                                    y_min = ver_dis
                                break

                        k += 1
            # theta is in [-pi, -pi/2)
            elif case == 3:
                for obs_x, obs_y in zip(near_obstacle_x, near_obstacle_y):
                    k = 0

                    for area in each_area_boundary:
                        '''
                        k = 0: right line
                        k = 1: front line
                        k = 2: left line
                        k = 3: rear line
                        '''
                        if k == 0:
                            obs_x_min = area[0] - self.expand_dis
                            obs_x_max = area[1]
                            obs_y_min = area[2]
                            obs_y_max = area[3] + self.expand_dis
                            if obs_x > obs_x_min and obs_x < obs_x_max and obs_y > obs_y_min and obs_y < obs_y_max:
                                hori_dis, ver_dis = compute_hori_ver_dis(
                                    (obs_x, obs_y), line_k[k], line_b[k], theta)
                                if hori_dis < x_min:
                                    x_min = hori_dis
                                if ver_dis < y_max:
                                    y_max = ver_dis
                                break

                        if k == 1:
                            obs_x_min = area[0] - self.expand_dis
                            obs_x_max = area[1]
                            obs_y_min = area[2] - self.expand_dis
                            obs_y_max = area[3]
                            if obs_x > obs_x_min and obs_x < obs_x_max and obs_y > obs_y_min and obs_y < obs_y_max:
                                hori_dis, ver_dis = compute_hori_ver_dis(
                                    (obs_x, obs_y), line_k[k], line_b[k], theta)
                                if hori_dis < x_min:
                                    x_min = hori_dis
                                if ver_dis < y_min:
                                    y_min = ver_dis
                                break

                        if k == 2:
                            obs_x_min = area[0]
                            obs_x_max = area[1] + self.expand_dis
                            obs_y_min = area[2] - self.expand_dis
                            obs_y_max = area[3]
                            if obs_x > obs_x_min and obs_x < obs_x_max and obs_y > obs_y_min and obs_y < obs_y_max:
                                hori_dis, ver_dis = compute_hori_ver_dis(
                                    (obs_x, obs_y), line_k[k], line_b[k], theta)
                                if hori_dis < x_max:
                                    x_max = hori_dis
                                if ver_dis < y_min:
                                    y_min = ver_dis
                                break

                        if k == 3:
                            obs_x_min = area[0]
                            obs_x_max = area[1] + self.expand_dis
                            obs_y_min = area[2]
                            obs_y_max = area[3] + self.expand_dis
                            if obs_x > obs_x_min and obs_x < obs_x_max and obs_y > obs_y_min and obs_y < obs_y_max:
                                hori_dis, ver_dis = compute_hori_ver_dis(
                                    (obs_x, obs_y), line_k[k], line_b[k], theta)
                                if hori_dis < x_max:
                                    x_max = hori_dis
                                if ver_dis < y_max:
                                    y_max = ver_dis
                                break

                        k += 1
            # theta is in [-pi/2, 0)
            elif case == 4:
                for obs_x, obs_y in zip(near_obstacle_x, near_obstacle_y):
                    k = 0
                    for area in each_area_boundary:
                        '''
                        k = 0: right line
                        k = 1: front line
                        k = 2: left line
                        k = 3: rear line
                        '''
                        if k == 0:
                            obs_x_min = area[0] - self.expand_dis
                            obs_x_max = area[1]
                            obs_y_min = area[2] - self.expand_dis
                            obs_y_max = area[3]
                            if obs_x > obs_x_min and obs_x < obs_x_max and obs_y > obs_y_min and obs_y < obs_y_max:
                                hori_dis, ver_dis = compute_hori_ver_dis(
                                    (obs_x, obs_y), line_k[k], line_b[k], theta)
                                if hori_dis < x_min:
                                    x_min = hori_dis
                                if ver_dis < y_min:
                                    y_min = ver_dis
                                break

                        if k == 1:
                            obs_x_min = area[0]
                            obs_x_max = area[1] + self.expand_dis
                            obs_y_min = area[2] - self.expand_dis
                            obs_y_max = area[3]
                            if obs_x > obs_x_min and obs_x < obs_x_max and obs_y > obs_y_min and obs_y < obs_y_max:
                                hori_dis, ver_dis = compute_hori_ver_dis(
                                    (obs_x, obs_y), line_k[k], line_b[k], theta)
                                if hori_dis < x_max:
                                    x_max = hori_dis
                                if ver_dis < y_min:
                                    y_min = ver_dis
                                break

                        if k == 2:
                            obs_x_min = area[0]
                            obs_x_max = area[1] + self.expand_dis
                            obs_y_min = area[2]
                            obs_y_max = area[3] + self.expand_dis
                            if obs_x > obs_x_min and obs_x < obs_x_max and obs_y > obs_y_min and obs_y < obs_y_max:
                                hori_dis, ver_dis = compute_hori_ver_dis(
                                    (obs_x, obs_y), line_k[k], line_b[k], theta)
                                if hori_dis < x_max:
                                    x_max = hori_dis
                                if ver_dis < y_max:
                                    y_max = ver_dis
                                break

                        if k == 3:
                            obs_x_min = area[0] - self.expand_dis
                            obs_x_max = area[1]
                            obs_y_min = area[2]
                            obs_y_max = area[3] + self.expand_dis
                            if obs_x > obs_x_min and obs_x < obs_x_max and obs_y > obs_y_min and obs_y < obs_y_max:
                                hori_dis, ver_dis = compute_hori_ver_dis(
                                    (obs_x, obs_y), line_k[k], line_b[k], theta)
                                if hori_dis < x_min:
                                    x_min = hori_dis
                                if ver_dis < y_max:
                                    y_max = ver_dis
                                break

                        k += 1

            X_max.append(x_max+x)
            Y_max.append(y_max+y)
            X_min.append(x-x_min)
            Y_min.append(y-y_min)

        # X_max_matrix = np.array(X_max).reshape(len(X_max), 1)
        # Y_max_matrix = np.array(Y_max).reshape(len(Y_max), 1)
        # X_min_matrix = np.array(X_min).reshape(len(X_min), 1)
        # Y_min_matrix = np.array(Y_min).reshape(len(Y_min), 1)
        # H_collision_matrix = np.vstack((H_max_matrix, -H_min_matrix))
        # slack_H_collision_matrix = np.vstack((H_max_matrix, 999*np.ones((points_n-2, 1)),
        #                                       -H_min_matrix, np.zeros((points_n-2, 1))))

        return X_max, Y_max, X_min, Y_min

    def solution(self, path: list):
        '''
        input: path is a list, [[x,y,theta,v,a,sigma,omega,t],[x,y...],...,[x,y...]]
        '''
        # create a model
        model = pyo.ConcreteModel()

        # define the initial solution
        initial_path = np.array(path)[:, :-1]
        tf_initial = path[-1][-1]
        '''
        initial_solution: [x_1,y_1,theta_1,v_1,a_1,
            sigma_1,omega_1,x_2,y_2,...,omega_n,tf]
        '''
        initial_solution = np.append(initial_path, tf_initial)
        variable_n = len(initial_solution)
        points_n = int((variable_n - 1) / 7)
        final_pose_sin = math.sin(initial_solution[-6])
        final_pose_cos = math.cos(initial_solution[-6])

        model.index_x = pyo.RangeSet(0, variable_n-1, 1)
        x = np.linspace(0, variable_n-1, variable_n, dtype=np.int32)

        def initial_x(model, i):
            model.variables[i] = initial_solution[int(i)]
            return model.variables[i]

        # get collision bounds
        x_max, y_max, x_min, y_min = self.compute_collision_H(path=path)
        small_v = 0.0001  # m/s

        def bound(model, i):
            i = int(i)
            if (i-0) % 7 == 0 and i < (variable_n-1):
                k = int((i-0) / 7)
                x_bounds = (x_min[k], x_max[k])
                return x_bounds
            elif (i-1) % 7 == 0 and i < (variable_n-1):
                k = int((i-1) / 7)
                y_bounds = (y_min[k], y_max[k])
                return y_bounds
            elif (i-2) % 7 == 0:
                theta_bounds = (-3.1415926, 3.1415926)
                return theta_bounds
            elif (i-3) % 7 == 0:
                if i == 3:
                    v_bounds = (0, small_v)
                else:
                    v_bounds = (-2.5, 2.5)
                return v_bounds
            elif (i-4) % 7 == 0:
                a_bounds = (-1, 1)
                return a_bounds
            elif (i-5) % 7 == 0:
                sigma_bounds = (-0.75, 0.75)
                return sigma_bounds
            elif (i-6) % 7 == 0:
                omega_bounds = (-0.5, 0.5)
                return omega_bounds
            elif i == (variable_n-1):
                tf_bounds = (0, 200)
                return tf_bounds

        model.variables = pyo.Var(x, within=pyo.Reals,
                                  initialize=initial_x, bounds=bound)

        # fix the initial pose and the goal pose
        model.variables[0].fix(initial_solution[0])
        model.variables[1].fix(initial_solution[1])
        model.variables[2].fix(initial_solution[2])
        model.variables[variable_n-8].fix(initial_solution[-8])
        model.variables[variable_n-7].fix(initial_solution[-7])
        # model.variables[variable_n-6].fix(initial_solution[-6])
        model.variables[variable_n-5].fix(0)
        model.variables[variable_n-4].fix(0)
        model.variables[variable_n-2].fix(0)

        dt = model.variables[variable_n-1] / (points_n - 1)

        def objective(model):
            '''
            the objective funtion is min: t + a^2+w^2+v^2 + \sigma^2
            input: x, y, theta, v, a, delta, w, ..., tf
            '''
            expr = 0
            expr = model.variables[variable_n-1]
            # print(len(model.variables))
            for i in range(variable_n):
                if (i-4) % 7 == 0:
                    expr += model.variables[i] ** 2
                elif (i-3) % 7 == 0:
                    expr += model.variables[i] ** 2
                elif (i-5) % 7 == 0:
                    expr += model.variables[i] ** 2
                elif (i-6) % 7 == 0:
                    expr += model.variables[i] ** 2
            return expr

        model.obj1 = pyo.Objective(rule=objective, sense=pyo.minimize)

        def kinematic_constraint(model, i):
            if i < 7:
                return pyo.Constraint.Skip
            elif i % 7 == 0:
                delta_s = model.variables[i-4] * dt
                delta_x = delta_s * pyo.cos(model.variables[i-5])
                # return abs(model.variables[i] - model.variables[i-7] - delta_x) <= 1e-6
                return model.variables[i] == model.variables[i-7] + delta_x
            elif (i-1) % 7 == 0:
                delta_s = model.variables[i-5] * dt
                delta_y = delta_s * pyo.sin(model.variables[i-6])
                return model.variables[i] == model.variables[i-7] + delta_y
            elif (i-2) % 7 == 0:
                delta_s = model.variables[i-6] * dt
                delta_theta = delta_s * pyo.tan(model.variables[i-4]) / Lw
                return model.variables[i] == model.variables[i-7] + delta_theta
            elif (i-3) % 7 == 0:
                delta_v = model.variables[i-6] * dt
                return model.variables[i] == model.variables[i-7] + delta_v
            elif (i-5) % 7 == 0:
                delta_sigma = model.variables[i-6] * dt
                return model.variables[i] == model.variables[i-7] + delta_sigma
            else:
                return pyo.Constraint.Skip
        model.eq_kinematic = pyo.Constraint(
            model.index_x, rule=kinematic_constraint)

        model.final_theta1 = pyo.Constraint(expr=pyo.sin(
            model.variables[variable_n-6]) == final_pose_sin)
        model.final_theta2 = pyo.Constraint(expr=pyo.cos(
            model.variables[variable_n-6]) == final_pose_cos)

        # solution
        model.variables.pprint()
        model.obj1.pprint()
        # opt = pyo.SolverFactory(
        #     'ipopt', executable=solver_path)  # 指定 ipopt 作为求解器
        solver = pyo.SolverFactory('ipopt')
        solver.options['max_iter'] == 1000
        solution = solver.solve(model)
        solution.write()

        optimal_tf = pyo.value(model.variables[variable_n-1])
        optimal_dt = optimal_tf / (points_n-1)
        optimal_traj = []
        points = []

        for index in range(variable_n-1):
            if index % 7 == 0 and index > 0:
                optimal_traj.append(points)
                points = []
            points.append(pyo.value(model.variables[index]))
            print('solution', index)
            print(pyo.value(model.variables[index]))

        optimal_traj.append(points)

        print('solved ocp problem')
        print('minimum value', pyo.value(model.obj1))

        return optimal_traj, optimal_tf, optimal_dt
