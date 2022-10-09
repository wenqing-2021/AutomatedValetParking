# Author: Yuansj
# 2022/10.01
# 问题记录：求解失败，因为内存溢出。

'''
This function is used to find an optimal traj for the parking. the initial solution is from path_optimazition.py.
the usage of cyipopt is https://cyipopt.readthedocs.io/en/stable/tutorial.html#problem-interface
'''

import jax.numpy as np
from jax import jit, grad, jacfwd, jacrev
from cyipopt import minimize_ipopt
from costmap import _map, Vehicle
import math

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
        def objective(x):
            '''
            the objective funtion is min: t + a^2+w^2+v^2 + \sigma^2
            input: x, y, theta, v, a, delta, w, ..., tf
            '''
            t = x[-1]
            length = len(x) - 1
            n = int(length / 7)
            vector_v = np.array([0, 0, 0, 1, 0, 0, 0])
            vector_a = np.array([0, 0, 0, 0, 1, 0, 0])
            vector_s = np.array([0, 0, 0, 0, 0, 1, 0])
            vector_w = np.array([0, 0, 0, 0, 0, 0, 1])
            Matrix_v = np.zeros((1, len(x)))
            Matrix_a = np.zeros((1, len(x)))
            Matrix_s = np.zeros((1, len(x)))
            Matrix_w = np.zeros((1, len(x)))

            for i in range(n):
                previous = np.zeros(7 * i)
                after = np.zeros(7 * (n-1-i))

                row_a = np.hstack(
                    (previous, vector_a, after, 0))  # 0 is the tf
                Matrix_a = np.vstack((Matrix_a, row_a))

                row_v = np.hstack((previous, vector_v, after, 0))
                Matrix_v = np.vstack((Matrix_v, row_v))

                row_s = np.hstack((previous, vector_s, after, 0))
                Matrix_s = np.vstack((Matrix_s, row_s))

                row_w = np.hstack((previous, vector_w, after, 0))
                Matrix_w = np.vstack((Matrix_w, row_w))

            Matrix_a = Matrix_a[1:]
            Matrix_v = Matrix_v[1:]
            Matrix_s = Matrix_s[1:]
            Matrix_w = Matrix_w[1:]

            a2 = np.inner(Matrix_a, x)
            a2_sum = np.inner(a2, a2)

            v2 = np.inner(Matrix_v, x)
            v2_sum = np.inner(v2, v2)

            s2 = np.inner(Matrix_s, x)
            s2_sum = np.inner(s2, s2)

            w2 = np.inner(Matrix_w, x)
            w2_sum = np.inner(w2, w2)

            return t + a2_sum + v2_sum + s2_sum + w2_sum

        def kinematic_constraints_x(x):
            '''
            kinematic constraint
            '''
            N = int((len(x) - 1) / 7)
            dt = x[-1] / (N - 1)  # tf / (N -1 )
            vector_x = np.array([1, 0, 0, 0, 0, 0, 0])
            vector_v = np.array([0, 0, 0, 1, 0, 0, 0])
            vector_theta = np.array([0, 0, 1, 0, 0, 0, 0])
            Matrix_x = np.zeros((1, len(x)))
            Matrix_v = np.zeros((1, len(x)))
            Matrix_theta = np.zeros((1, len(x)))
            for i in range(N):
                previous = np.zeros(7 * i)
                after = np.zeros(7 * (N-1-i))

                row_x = np.hstack((previous, vector_x, after, 0))  # 0 means tf
                row_v = np.hstack((previous, vector_v, after, 0))
                row_theta = np.hstack((previous, vector_theta, after, 0))

                Matrix_x = np.vstack((Matrix_x, row_x))
                Matrix_v = np.vstack((Matrix_v, row_v))
                Matrix_theta = np.vstack((Matrix_theta, row_theta))

            Matrix_x = Matrix_x[1:]
            Matrix_v = Matrix_v[1:]
            Matrix_theta = Matrix_theta[1:]
            all_x = np.inner(Matrix_x, x)
            all_v = np.inner(Matrix_v, x)
            all_theta = np.inner(Matrix_theta, x)

            # all_x[1:] = all_x[:-1] + all_v[:-1] * dt * np.cos(all_theta[:-1])

            return all_x[1:] - (all_x[:-1] + all_v[:-1] * dt * np.cos(all_theta[:-1]))

        def kinematic_constraints_y(x):
            '''
            kinematic constraint
            '''
            N = int((len(x) - 1) / 7)
            dt = x[-1] / (N - 1)  # tf / (N -1 )
            vector_y = np.array([0, 1, 0, 0, 0, 0, 0])
            vector_v = np.array([0, 0, 0, 1, 0, 0, 0])
            vector_theta = np.array([0, 0, 1, 0, 0, 0, 0])
            Matrix_y = np.zeros((1, len(x)))
            Matrix_v = np.zeros((1, len(x)))
            Matrix_theta = np.zeros((1, len(x)))
            for i in range(N):
                previous = np.zeros(7 * i)
                after = np.zeros(7 * (N-1-i))

                row_y = np.hstack((previous, vector_y, after, 0))
                row_v = np.hstack((previous, vector_v, after, 0))
                row_theta = np.hstack((previous, vector_theta, after, 0))

                Matrix_y = np.vstack((Matrix_y, row_y))
                Matrix_v = np.vstack((Matrix_v, row_v))
                Matrix_theta = np.vstack((Matrix_theta, row_theta))

            Matrix_y = Matrix_y[1:]
            Matrix_v = Matrix_v[1:]
            Matrix_theta = Matrix_theta[1:]
            all_y = np.inner(Matrix_y, x)
            all_v = np.inner(Matrix_v, x)
            all_theta = np.inner(Matrix_theta, x)

            # all_y[1:] = all_y[:-1] + all_v[:-1] * dt * np.sin(all_s[:-1])

            return all_y[1:] - (all_y[:-1] + all_v[:-1] * dt * np.sin(all_theta[:-1]))

        def kinematic_constraints_theta(x):
            '''
            kinematic constraint
            '''
            N = int((len(x) - 1) / 7)
            dt = x[-1] / (N - 1)  # tf / (N -1 )
            vector_theta = np.array([0, 0, 1, 0, 0, 0, 0])
            vector_v = np.array([0, 0, 0, 1, 0, 0, 0])
            vector_s = np.array([0, 0, 0, 0, 0, 1, 0])
            Matrix_theta = np.zeros((1, len(x)))
            Matrix_v = np.zeros((1, len(x)))
            Matrix_s = np.zeros((1, len(x)))
            for i in range(N):
                previous = np.zeros(7 * i)
                after = np.zeros(7 * (N-1-i))

                row_theta = np.hstack((previous, vector_theta, after, 0))
                row_v = np.hstack((previous, vector_v, after, 0))
                row_s = np.hstack((previous, vector_s, after, 0))

                Matrix_theta = np.vstack((Matrix_theta, row_theta))
                Matrix_v = np.vstack((Matrix_v, row_v))
                Matrix_s = np.vstack((Matrix_s, row_s))

            Matrix_theta = Matrix_theta[1:]
            Matrix_v = Matrix_v[1:]
            Matrix_s = Matrix_s[1:]
            all_theta = np.inner(Matrix_theta, x)
            all_v = np.inner(Matrix_v, x)
            all_s = np.inner(Matrix_s, x)

            # all_theta[1:] = all_theta[:-1] + all_v[:-1] * dt * np.tan(all_s[:-1]) / Lw

            return all_theta[1:] - (all_theta[:-1] + all_v[:-1] * dt * np.tan(all_s[:-1]) / Lw)

        def initial_pose_eq_constraints(x):
            previous = np.array([1, 1, 1])
            last = np.zeros(len(x) - len(previous))
            vector_initial_pose = np.hstack((previous, last))
            initial_pose = np.inner(vector_initial_pose, x)
            constraint_initial_pose = np.array(path[0][:3])

            return initial_pose - constraint_initial_pose

        def goal_pose_eq_constraint(x):
            previous = np.array([1, 1, 1, 0, 0, 0, 0, 0])
            last = np.zeros(len(x) - len(previous))
            vector_goal_pose = np.hstack((previous, last))
            goal_pose = np.inner(vector_goal_pose, x)
            constraint_goal_pose = np.array(path[-1][:3])

            return goal_pose - constraint_goal_pose

        def ineq_constrains(x):
            pass

        initial_path = np.array(path)[:, :-1]
        tf = path[-1][-1]
        '''
        initial_solution: [x_1,y_1,theta_1,v_1,a_1,sigma_1,omega_1,x_2,y_2,...,omega_n,tf]
        '''
        initial_solution = np.append(initial_path, tf)

        # get collision bounds
        x_max, y_max, x_min, y_min = self.compute_collision_H(path=path)
        bnds = []
        for i in range(len(initial_path)):
            bnds.append((x_min[i], x_max[i]))
            bnds.append((y_min[i], y_max[i]))
            bnds.append((-math.pi, math.pi))
            bnds.append((-2.5, 2.5))
            bnds.append((-1, 1))
            bnds.append((-0.75, 0.75))
            bnds.append((-0.5, 0.5))

        # tf bounds
        bnds.append((0, 300))

        # solution
        # jit the functions
        obj_jit = jit(objective)
        con_eq_x_jit = jit(kinematic_constraints_x)
        con_eq_y_jit = jit(kinematic_constraints_y)
        con_eq_theta_jit = jit(kinematic_constraints_theta)
        con_eq_initial_pose_jit = jit(initial_pose_eq_constraints)
        con_eq_goal_pose_jit = jit(goal_pose_eq_constraint)

        # build the derivatives and jit them
        obj_grad = jit(grad(obj_jit))  # objective gradient
        obj_hess = jit(jacrev(jacfwd(obj_jit)))  # objective hessian

        con_eq_x_jac = jit(jacfwd(con_eq_x_jit))  # jacobian
        con_eq_y_jac = jit(jacfwd(con_eq_y_jit))  # jacobian
        con_eq_theta_jac = jit(jacfwd(con_eq_theta_jit))  # jacobian
        con_eq_initial_pose_jac = jit(
            jacfwd(con_eq_initial_pose_jit))  # jacobian
        con_eq_goal_pose_jac = jit(jacfwd(con_eq_goal_pose_jit))  # jacobian

        con_eq_x_hess = jacrev(jacfwd(con_eq_x_jit))  # hessian
        con_eq_x_hessvp = jit(lambda x, v: con_eq_x_hess(
            x) * v[0])  # hessian vector-product
        con_eq_y_hess = jacrev(jacfwd(con_eq_y_jit))  # hessian
        con_eq_y_hessvp = jit(lambda x, v: con_eq_y_hess(
            x) * v[0])  # hessian vector-product
        con_eq_theta_hess = jacrev(jacfwd(con_eq_theta_jit))  # hessian
        con_eq_theta_hessvp = jit(lambda x, v: con_eq_theta_hess(
            x) * v[0])  # hessian vector-product
        con_eq_initial_pose_hess = jacrev(
            jacfwd(con_eq_initial_pose_jac))  # hessian
        con_eq_initial_pose_hessvp = jit(lambda x, v: con_eq_initial_pose_hess(
            x) * v[0])  # hessian vector-product
        con_eq_goal_pose_hess = jacrev(jacfwd(con_eq_goal_pose_jac))  # hessian
        con_eq_goal_pose_hessvp = jit(lambda x, v: con_eq_goal_pose_hess(
            x) * v[0])  # hessian vector-product

        # constraints
        cons = [{'type': 'eq', 'fun': con_eq_x_jit, 'jac': con_eq_x_jac, 'hess': con_eq_x_hessvp},
                {'type': 'eq', 'fun': con_eq_y_jit,
                    'jac': con_eq_y_jac, 'hess': con_eq_y_hessvp},
                {'type': 'eq', 'fun': con_eq_theta_jit,
                    'jac': con_eq_theta_jac, 'hess': con_eq_theta_hessvp},
                {'type': 'eq', 'fun': con_eq_initial_pose_jit,
                    'jac': con_eq_initial_pose_jac, 'hess': con_eq_initial_pose_hessvp},
                {'type': 'eq', 'fun': con_eq_goal_pose_jit, 'jac': con_eq_goal_pose_jac, 'hess': con_eq_goal_pose_hessvp}, ]

        # starting point
        x0 = initial_solution

        # executing the solver
        res = minimize_ipopt(obj_jit, jac=obj_grad, hess=obj_hess, x0=x0, bounds=bnds,
                             constraints=cons, options={'disp': 5})

        print('solved ocp problem')

        return res
