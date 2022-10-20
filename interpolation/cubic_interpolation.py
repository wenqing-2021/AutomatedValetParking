# coding:utf-8
# Author: Yuansj
# Last update:2022/07/05
# update: 07/09 因为没有进行坐标变换进行插值，导致某些曲线的导数无穷大，

import math
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
        self.path = None
        self.insert_dt = config["velocity_plan_dt"]
        self.vehicle = vehicle

    def get_cubic_interpolation(self,
                                path: list,
                                velocity_func,
                                acc_func) -> list:
        '''
        insert more points, and compute the initial solution
        '''
        self.path = path
        # check this short path is forward or not
        theta_forward_1 = self.path[0][2] > - \
            math.pi/2 and self.path[0][2] < math.pi/2
        theta_forward_2 = (self.path[0][2] > math.pi/2 and self.path[0][2] < math.pi) or \
                          (self.path[0][2] > -
                           math.pi and self.path[0][2] < -math.pi/2)
        forward = True if (self.path[0][0] < self.path[1][0] and theta_forward_1) or \
                          (self.path[0][0] > self.path[1][0] and theta_forward_2) else False

        # update the theta of waypoints
        self.update_theta(forward)
        t = 0
        insert_path = []
        # assume initial velocity
        v = velocity_func(t+self.insert_dt)

        # insert point between each two waypoints
        for i in range(len(self.path)-1):
            start, end = self.path[i], self.path[i+1]
            cubic_func, rotation_matrix, new_end = spine.cubic_spline(
                start, end)

            if i > 0:
                v = velocity_func(t)

            # store the insert points in the transformed path list
            # assume the initial point has the velocity, we need it to compute the steering angle
            trans_path = [[0, 0, 0, v, acc_func(t), t]]
            delta_dis = v * self.insert_dt * math.cos(trans_path[-1][2])

            if forward:
                while (
                    trans_path[-1][0] + delta_dis < new_end[0]
                ):
                    if v <= 0:
                        break

                    # add the insert point
                    trans_x = trans_path[-1][0] + delta_dis
                    trans_y, trans_theta = cubic_func(trans_x)
                    t = t+self.insert_dt
                    v = velocity_func(t)
                    acc = acc_func(t)
                    trans_path.append(
                        [trans_x, trans_y, trans_theta, v, acc, t])

                    delta_dis = v * self.insert_dt * \
                        math.cos(trans_path[-1][2])

            else:
                while (
                    trans_path[-1][0] - delta_dis > new_end[0]
                ):
                    if v <= 0:
                        break
                    # add the insert point
                    trans_x = trans_path[-1][0] - delta_dis
                    trans_y, trans_theta = cubic_func(trans_x)
                    t = t+self.insert_dt
                    v = velocity_func(t)
                    acc = acc_func(t)
                    trans_path.append(
                        [trans_x, trans_y, trans_theta, v, acc, t])

                    delta_dis = v * self.insert_dt * \
                        math.cos(trans_path[-1][2])

            # add transformed end points
            delta_t = math.sqrt(
                (new_end[0]-trans_path[-1][0])**2+(new_end[1]-trans_path[-1][1])**2) / v
            t = t + delta_t
            v = velocity_func(t)
            acc = acc_func(t)
            trans_path.append([new_end[0], new_end[1], new_end[2], v, acc, t])

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

    # def solve_cubic_spline(self, start, end):
    #     '''
    #     :param start: start node
    #     :param end: end node
    #     :return: target cubic function,include [a,b,c,d]
    #     '''
    #     rotation_matrix, new_end = self.rotation_transfrom(
    #         start=start, end=end)
    #     x0, y0, theta0 = 0, 0, 0
    #     x1, y1, theta1 = new_end[0], new_end[1], new_end[2]
    #     A = np.array([
    #         [x0**3, x0**2, x0, 1],
    #         [x1**3, x1**2, x1, 1],
    #         [3*x0**2, 2*x0, 1, 0],
    #         [3*x1**2, 2*x1, 1, 0]
    #     ])
    #     b = np.array([y0, y1, math.tan(theta0), math.tan(theta1)])
    #     result = solve(A, b)

    #     def cubic_func(x):
    #         a = result[0]
    #         b = result[1]
    #         c = result[2]
    #         d = result[3]
    #         y = a*x**3+b*x**2+c*x+d
    #         k = 3*a*x**2 + 2*b*x + c
    #         theta_angle = math.atan(k)

    #         return y, theta_angle

    #     return cubic_func, rotation_matrix, new_end

    def update_theta(self, forward, path) -> None:
        '''
        this function is used for compute the theta angle of each
        waypoint. For example, if we want to compute the theta angle of
        point_{i}, we define it as the angle between x axis and vector,
        which start from point_{i-1} to point_{i+1}.
        '''
        for i in range(1, len(path)-1):
            # compute vector
            if forward:
                vector_i = (path[i+1][0] - path[i-1][0],
                            path[i+1][1] - path[i-1][1])
            else:
                vector_i = (path[i-1][0] - path[i+1][0],
                            path[i-1][1] - path[i+1][1])
            vector_x = (1, 0)

            cosine = 1 - spatial.distance.cosine(vector_i, vector_x)
            tan_value = vector_i[1] / vector_i[0]
            # return is [-pi/2, pi/2]
            theta_i = math.atan(vector_i[1] / vector_i[0])

            if cosine < 0:
                if tan_value > 0:
                    theta_i -= math.pi
                else:
                    theta_i += math.pi

            # update theta
            path[i][2] = theta_i

    # def rotation_transfrom(self, start, end) -> None:
    #     theta = start[2]
    #     R = np.array([[np.cos(theta), np.sin(theta)],
    #                  [-np.sin(theta), np.cos(theta)]])
    #     trans_end_x_y = np.dot(R, np.array(
    #         [[end[0] - start[0]], [end[1] - start[1]]]))
    #     new_end = [trans_end_x_y[0, 0], trans_end_x_y[1, 0]]
    #     new_end.append(end[2] - theta)

    #     return R, new_end

    # def inverse_trans(self,
    #                   trans_path: list = None,
    #                   rotation_matrix: np.array = None,
    #                   start: list = None) -> list:

    #     start_array = np.ones((len(trans_path), 2))
    #     start_array[:, 0] = start_array[:, 0] * start[0]
    #     start_array[:, 1] = start_array[:, 1] * start[1]
    #     trans_path = np.array(trans_path)
    #     trans_path_position = trans_path[:, :2]
    #     inversed_path_position = (rotation_matrix.transpose().dot(trans_path_position.transpose())).transpose() + \
    #         start_array
    #     trans_path[:, :2] = inversed_path_position
    #     trans_path[:, 2:3] += start[2]
    #     inversed_path = trans_path.tolist()

    #     return inversed_path
