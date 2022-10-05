# coding:utf-8
# Author: Yuansj 
# Last update:2022/06/26

import numpy as np
import math
from cvxopt import matrix, solvers
from costmap import _map, Vehicle

class path_opti:
    def __init__(self,
                 park_map: _map, 
                 vehicle: Vehicle,
                 config: dict) -> None:
        self.original_path = None
        self.map = park_map
        self.vehicle = vehicle
        self.matrix_dict = dict()
        self.expand_dis = config['expand_dis'] # m
        self.config = config
    
    def formate_matrix(self,
                       path: list) -> np.array:
        '''
        QP objective function form: 1/2 X^T P X + Q^T X
        subject to: GX <= H
                    AX = B
        '''
        self.original_path = path
        points_n = len(self.original_path)
        # compute the path smooth function
        smooth_matrix = np.zeros((2*points_n, 2*points_n))
        if points_n > 2:
            for i in range(points_n-2):
                # create left upper original matrix
                zero_matrix_1 = np.zeros((2*i, 2*i))
                # create coffecient matrix
                eye_matrix = np.eye(2)
                coffecient_1 = np.hstack((eye_matrix, -2*eye_matrix, eye_matrix))
                coffecient_2 = np.hstack((-2*eye_matrix, 4*eye_matrix, -2*eye_matrix))
                coffecient = np.vstack((coffecient_1, coffecient_2, coffecient_1))
                # create right down original matrix
                zero_matrix_2 = np.zeros((2*(points_n-i-3), 2*(points_n-i-3)))
                # stack them
                first_row_matrix = np.hstack((zero_matrix_1, np.zeros((2*i, 2*points_n - 2*i))))
                second_row_matrix = np.hstack((np.zeros((6, 2*i)), coffecient, np.zeros((6, 2*(points_n-i-3)))))
                third_row_matrix = np.hstack((np.zeros((2*(points_n-i-3), 2*i+6)), zero_matrix_2))
                smooth_matrix = smooth_matrix + np.vstack((first_row_matrix, second_row_matrix, third_row_matrix))
        
        # compute the path compaction function
        compaction_matrix = np.zeros((2*points_n, 2*points_n))
        for i in range(points_n-1):
            zero_matrix_1 = np.zeros((2*i, 2*i))
            coffecient_1 = np.hstack((eye_matrix, -eye_matrix))
            coffecient_2 = np.hstack((-eye_matrix, eye_matrix))
            coffecient = np.vstack((coffecient_1, coffecient_2))
            zero_matrix_2 = np.zeros((2*(points_n-i-2), 2*(points_n-i-2)))
            # stack
            first_row_matrix = np.hstack((zero_matrix_1, np.zeros((2*i, 2*points_n - 2*i))))
            second_row_matrix = np.hstack((np.zeros((4, 2*i)), coffecient, np.zeros((4, 2*(points_n-i-2)))))
            third_row_matrix = np.hstack((np.zeros((2*(points_n-i-2), 2*i+4)), zero_matrix_2))
            compaction_matrix = compaction_matrix + np.vstack((first_row_matrix, second_row_matrix, third_row_matrix))
        
        # compute the path offset function
        path_offset_matrix = np.eye(2*points_n)
        # compute the P matrix
        smooth_weight = self.config['smooth_cost']
        compact_weight = self.config['compact_cost']
        offset_weight = self.config['offset_cost']
        slack_weight = self.config['slack_cost']
        P_matrix = 2 * (smooth_weight * smooth_matrix + compact_weight*compaction_matrix + offset_weight * path_offset_matrix)
        # compute Q matrix
        Q = []
        B = []
        for i in range(points_n):
            if i == 0 or i == (points_n-1):
                B.append(self.original_path[i][0])
                B.append(self.original_path[i][1])
            Q.append(self.original_path[i][0])
            Q.append(self.original_path[i][1])
        
        Q_matrix = -2 * np.array(Q)
        Q_matrix = offset_weight * Q_matrix.reshape(len(Q_matrix),1)
        slack_Q_matrix = np.vstack((Q_matrix, np.zeros((points_n-2, 1)))) + \
                         np.vstack((np.zeros((2*points_n, 1)), slack_weight * np.ones((points_n-2, 1))))

        # boundary subject
        B_matrix = np.array(B)
        B_matrix = B_matrix.reshape(len(B_matrix),1)
        a_1 = np.hstack((eye_matrix, np.zeros((2, 2*points_n-2))))
        a_2 = np.hstack((np.zeros((2, 2*points_n-2)), eye_matrix))
        A_matrix = np.vstack((a_1, a_2))
        slack_A_matrix = np.hstack((A_matrix, np.zeros((4, points_n-2))))
        slack_B_matrix = B_matrix

        # compute the G and H based on the collision check and curvature limit
        G_1 = np.eye(2*points_n)
        G_2 = -np.eye(2*points_n)
        G_matrix_collision = np.vstack((G_1, G_2))
        H_matrix_collision, slack_H_matrix_collision = self.compute_collision_H()

        G_matrix_curv, H_matrix_curv = self.compute_curvature_H()
        slack_G_matrix_curv = np.hstack((G_matrix_curv, -np.ones((points_n-2, points_n-2))))
        slack_H_matrix_curv = H_matrix_curv
        slack_G_matrix_collision = np.vstack((np.eye(3*points_n-2), -np.eye(3*points_n-2)))
        G_matrix = np.vstack((G_matrix_collision, G_matrix_curv))
        H_matrix = np.vstack((H_matrix_collision, H_matrix_curv))
        slack_G_matrix = np.vstack((slack_G_matrix_collision, slack_G_matrix_curv))
        slack_H_matrix = np.vstack((slack_H_matrix_collision, slack_H_matrix_curv))

        # if we consider the slack variable
        zero_matrix = np.zeros((points_n-2, points_n-2))
        P_first_row = np.hstack((P_matrix, np.zeros((P_matrix.shape[0], points_n-2))))
        P_second_row = np.hstack((np.zeros((points_n-2, P_matrix.shape[1])), zero_matrix))
        slack_P_matrix = np.vstack((P_first_row, P_second_row))

        # record original matrix
        self.matrix_dict = {'P':P_matrix,'Q':Q_matrix,'A':A_matrix,
                            'B':B_matrix,'G':G_matrix,'H':H_matrix}

        # record slack matrix
        self.slack_matrix_dict = {'P':slack_P_matrix,'Q':slack_Q_matrix,
                                  'A':slack_A_matrix,'B':slack_B_matrix,
                                  'G':slack_G_matrix,'H':slack_H_matrix}
        
        return slack_P_matrix, slack_Q_matrix, slack_A_matrix, slack_B_matrix, slack_G_matrix, slack_H_matrix

    def get_result(self, path):
        P,Q,A,B,G,H = self.formate_matrix(path)
        P = matrix(P)
        Q = matrix(Q)
        A = matrix(A)
        B = matrix(B)
        G = matrix(G)
        H = matrix(H)
        solvers.options['maxiters'] = 100
        QP_result=solvers.qp(P,Q,G,H,A,B)
        result_path = QP_result['x']
        points_n = len(self.original_path)
        result_path = result_path[:2*points_n]
        opti_path = []
        for i in range(int(len(result_path)/2)):
            point = [result_path[2*i], result_path[2*i+1], self.original_path[i][2]]
            opti_path.append(point)
        return opti_path

    def compute_collision_H(self):
        '''
        use AABB block to find those map points near the vehicle
        and then find the shortest distance from these points to 
        the vehicle square. noted as [f_d, b_d, r_d, l_d]
        f_d is the shortest distance from obstacles to the front edge
        b_d is to the rear edge, and r_d is to the right edge, l_d is 
        to the left edge.
        [E;-E] X <= [H_max;-H_min]
        '''
        
        # get near obstacles and vehicle
        def get_near_obstacles(node_x,node_y,theta, map:_map, config):
            '''
            this function is only used for distance check method
            return the obstacles x and y, vehicle boundary
            Note: vehicle boundary is expanded
            '''

            # create vehicle boundary 
            v = Vehicle()

            # create_polygon
            vehicle_boundary = v.create_anticlockpoint(x=node_x, y=node_y, theta=theta, config=config)
            
            '''
            right_rear = vehicle_boundary[0]
            right_front = vehicle_boundary[1]
            left_front = vehicle_boundary[2]
            left_rear = vehicle_boundary[3]
            note: these points have expanded
            '''

            # create AABB squaref
            x_max = max(vehicle_boundary[:,0]) + self.expand_dis
            x_min = min(vehicle_boundary[:,0]) - self.expand_dis
            y_max = max(vehicle_boundary[:,1]) + self.expand_dis
            y_min = min(vehicle_boundary[:,1]) - self.expand_dis

            # get obstacle position
            obstacle_index = np.where(map.cost_map == 255)
            obstacle_position_x = map.map_position[0][obstacle_index[0]]
            obstacle_position_y = map.map_position[1][obstacle_index[1]]

            # find those obstacles point in the AABB square
            near_x_position = obstacle_position_x[np.where((obstacle_position_x >= x_min) & (obstacle_position_x <= x_max))]
            near_y_position = obstacle_position_y[np.where((obstacle_position_x >= x_min) & (obstacle_position_x <= x_max))]

            # determine y
            near_obstacle_x = near_x_position[np.where((near_y_position >= y_min) & (near_y_position <= y_max))]
            near_obstacle_y = near_y_position[np.where((near_y_position >= y_min) & (near_y_position <= y_max))]

            near_obstacle_range = [near_obstacle_x, near_obstacle_y]

            return near_obstacle_range, vehicle_boundary
        
        # compute the parameters of boundary line
        def compute_k_b(point_1,point_2):
            # k = (y_2 - y_1) / (x_2 - x_1)
            k =  (point_2[1] - point_1[1]) / (point_2[0] - point_1[0])
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
        def compute_hori_ver_dis(point,k,b,theta):
            shortest_dis = compute_distance(_k=k,_b=b,point=point)
            vertical_dis = shortest_dis / abs(np.cos(theta))
            horizon_dis = shortest_dis / abs(np.sin(theta))
            return float(horizon_dis), float(vertical_dis)
        
        H_max = []
        H_min = []
        points_n = len(self.original_path)
        # create AABB boundary and get the near obstacles position
        for p in self.original_path:
            x,y,theta = p[0], p[1], p[2]
            near_obstacles_range, vehicle_boundary = get_near_obstacles(node_x=x, node_y=y, 
                                                                        theta=theta,map=self.map, config=self.config)
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
                    k_i, b_i = compute_k_b(vehicle_boundary[i], vehicle_boundary[i+1])
                    line_k.append(k_i)
                    line_b.append(b_i)
                else:
                    k_i, b_i = compute_k_b(vehicle_boundary[i], vehicle_boundary[0])
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
            elif theta >=0 and theta < math.pi / 2:
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
                    _area = get_area_boundary(vehicle_boundary[i], vehicle_boundary[i+1])
                else:
                    _area = get_area_boundary(vehicle_boundary[i], vehicle_boundary[0])
                
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
                                hori_dis, ver_dis = compute_hori_ver_dis((obs_x,obs_y), line_k[k], line_b[k],theta)
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
                                hori_dis, ver_dis = compute_hori_ver_dis((obs_x,obs_y), line_k[k], line_b[k],theta)
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
                                hori_dis, ver_dis = compute_hori_ver_dis((obs_x,obs_y), line_k[k], line_b[k],theta)
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
                                hori_dis, ver_dis = compute_hori_ver_dis((obs_x,obs_y), line_k[k], line_b[k],theta)
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
                                hori_dis, ver_dis = compute_hori_ver_dis((obs_x,obs_y), line_k[k], line_b[k],theta)
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
                                hori_dis, ver_dis = compute_hori_ver_dis((obs_x,obs_y), line_k[k], line_b[k],theta)
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
                                hori_dis, ver_dis = compute_hori_ver_dis((obs_x,obs_y), line_k[k], line_b[k],theta)
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
                                hori_dis, ver_dis = compute_hori_ver_dis((obs_x,obs_y), line_k[k], line_b[k],theta)
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
                                hori_dis, ver_dis = compute_hori_ver_dis((obs_x,obs_y), line_k[k], line_b[k],theta)
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
                                hori_dis, ver_dis = compute_hori_ver_dis((obs_x,obs_y), line_k[k], line_b[k],theta)
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
                                hori_dis, ver_dis = compute_hori_ver_dis((obs_x,obs_y), line_k[k], line_b[k],theta)
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
                                hori_dis, ver_dis = compute_hori_ver_dis((obs_x,obs_y), line_k[k], line_b[k],theta)
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
                                hori_dis, ver_dis = compute_hori_ver_dis((obs_x,obs_y), line_k[k], line_b[k],theta)
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
                                hori_dis, ver_dis = compute_hori_ver_dis((obs_x,obs_y), line_k[k], line_b[k],theta)
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
                                hori_dis, ver_dis = compute_hori_ver_dis((obs_x,obs_y), line_k[k], line_b[k],theta)
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
                                hori_dis, ver_dis = compute_hori_ver_dis((obs_x,obs_y), line_k[k], line_b[k],theta)
                                if hori_dis < x_min:
                                    x_min = hori_dis
                                if ver_dis < y_max:
                                    y_max = ver_dis 
                                break
                    
                        k += 1

            H_max.append(x_max+x)
            H_max.append(y_max+y)
            H_min.append(x-x_min)
            H_min.append(y-y_min)

        H_max_matrix = np.array(H_max).reshape(len(H_max),1)
        H_min_matrix = np.array(H_min).reshape(len(H_min),1)
        H_collision_matrix = np.vstack((H_max_matrix, -H_min_matrix))
        slack_H_collision_matrix = np.vstack((H_max_matrix, 999*np.ones((points_n-2, 1)), \
                                              -H_min_matrix, np.zeros((points_n-2, 1))))

        return H_collision_matrix, slack_H_collision_matrix

    def compute_curvature_H(self):
        '''
        We consider the curvature limits, and the final formate is 
        F'(X^r) \dot X <= F'(X^r) \dot X^r -F(X^r). We firstly use the 
        positions of continuous three points to get the equation with 
        the curvature and then use Taylor expansion to formate it as the 
        above fomulation.
        '''
        # formate F(X^r), X^r is the orginal points
        points_n = len(self.original_path)
        max_curvature = 1 / self.vehicle.min_radius_turn
        delta_s = 0.125 # m, which equals to STEP_SIZE in rs_curve also equals to max_v * ddt (2.5m/s * 0.05s)
        
        points_r = np.array(self.original_path)
        points_r_x, points_r_y = points_r[:,0], points_r[:,1]
        F_xr = (points_r_x[2:] -2*points_r_x[1:-1] + points_r_x[:-2])**2 + \
               (points_r_y[2:] -2*points_r_y[1:-1] + points_r_y[:-2])**2 - ((delta_s**2) * max_curvature)**2
        F_xr = F_xr.reshape((points_n-2, 1))

        x_r = points_r[:,0:2].flatten()
        x_r = x_r.reshape((2*points_n,1))
        eye_2 = np.eye(2)
        row_eye = np.hstack((eye_2, -2*eye_2, eye_2))
        constant_matrix = np.vstack((row_eye, -2*row_eye, row_eye))
        F_pie_xr = np.zeros((1,2*points_n))
        # compute derive
        for i in range(points_n-2):
            i = i+1
            first_row_1 = np.zeros((2*i-2, 2*i-2))
            first_row_2 = np.zeros((2*i-2, 2*points_n-2*i+2))
            first_row = np.hstack((first_row_1, first_row_2))
            second_row = np.hstack((np.zeros((6, 2*i-2)),
                                    constant_matrix,
                                    np.zeros((6, 2*points_n-2*i-4))))
            third_row = np.hstack((np.zeros((2*points_n-2*i-4, 2*i+4)), 
                                   np.zeros((2*points_n-2*i-4, 2*points_n-2*i-4))))
            Q_i = np.vstack((first_row, second_row, third_row))
            F_pie_xr_i = np.dot(x_r.transpose(), Q_i)
            F_pie_xr = np.vstack((F_pie_xr, F_pie_xr_i))
        
        F_pie_xr = F_pie_xr[1:]

        G_matrix = F_pie_xr
        H_matrix = np.dot(F_pie_xr, x_r) - F_xr

        return G_matrix, H_matrix
        
        
