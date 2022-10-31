'''
Author: wenqing-hnu
Date: 2022-10-20
LastEditors: wenqing-hnu
LastEditTime: 2022-10-31
FilePath: /HybridAstar/path_planner/path_planner.py
Description: path plan

Copyright (c) 2022 by wenqing-hnu, All Rights Reserved. 
'''


import copy
import numpy as np
from scipy import spatial
from typing import Dict, Tuple, List

from path_planner.hybrid_a_star import hybrid_a_star
from animation.animation import plot_collision_p
from map.costmap import Vehicle, _map
from collision_check import collision_check
from path_planner.rs_curve import PATH


class path_planner:
    def __init__(self,
                 config: dict = None,
                 map: _map = None,
                 vehicle: Vehicle = None) -> None:
        self.config = config
        self.map = map
        self.vehicle = vehicle
        if config['collision_check'] == 'circle':
            self.collision_checker = collision_check.two_circle_checker(map=map,
                                                                        vehicle=vehicle,
                                                                        config=config)
        else:
            self.collision_checker = collision_check.distance_checker(map=map,
                                                                      vehicle=vehicle,
                                                                      config=config)

        self.planner = hybrid_a_star(
            config=config, park_map=map, vehicle=vehicle)

    def path_planning(self) -> Tuple[List[List], Dict, List[List[List]]]:
        final_path, astar_path, rs_path = self.a_star_plan()
        split_path_list, change_gear = self.split_path(final_path)

        path_info = {'astar_path': astar_path,
                     'rs_path': rs_path,
                     'change_gear': change_gear,
                     }
        # insert more points compared with final_path
        out_final_path = sum(split_path_list, [])

        return out_final_path, path_info, split_path_list

    def a_star_plan(self) -> Tuple[List[List], List[List], PATH]:
        '''
        use a star to search a feasible path and use rs curve to reach the goal,
        final_path = astar_path + rs_path
        return: final_path, astar_path, rs_path
        '''
        astar = self.planner

        reach_goal = False

        while not astar.open_list.empty() and not reach_goal:
            # get current node
            current_node = astar.open_list.get()
            # show info
            print('---------------')
            print('current node index:', current_node.index)
            print('distance:', np.sqrt((current_node.x-self.map.case.xf)
                                       ** 2 + (current_node.y - self.map.case.yf)**2))
            print('---------------')

            rs_path, collision, info = astar.try_reach_goal(current_node)

            # plot the collision position
            if collision and self.config['draw_collision']:
                collision_p = info['collision_position']
                plot_collision_p(
                    collision_p[0], collision_p[1], collision_p[2], self.map)

            if not collision and info['in_radius']:
                reach_goal = True
                break

            else:
                # expand node
                child_group = astar.expand_node(current_node)
                path = []
                for i in child_group.queue:
                    x = i.x
                    y = i.y
                    theta = i.theta
                    path.append([x, y, theta])

        a_star_path = astar.finish_path(current_node)
        final_path = copy.deepcopy(a_star_path)
        # final_path = a_star_path + rs_path
        # assemble all path
        for i in range(1, len(rs_path.x)):
            x = rs_path.x[i]
            y = rs_path.y[i]
            theta = rs_path.yaw[i]
            final_path.append([x, y, theta])

        return final_path, a_star_path, rs_path

    def split_path(self, final_path: List[List]) -> Tuple[List[List[List]], int]:
        '''
        split the final path (a star + rs path) into severial single path for optimization
        input: final_path is generated from the planner
        return: split_path, change_gear
        '''
        # split path based on the gear (forward or backward)
        split_path = []
        change_gear = 0
        start = 0
        extend_num = self.config['extended_num']
        # we want to extend node but these points also need collision check
        have_extended_points = 0

        for i in range(len(final_path) - 2):
            vector_1 = (final_path[i+1][0] - final_path[i][0],
                        final_path[i+1][1] - final_path[i][1])

            vector_2 = (final_path[i+2][0] - final_path[i+1][0],
                        final_path[i+2][1] - final_path[i+1][1])

            compute_cosin = 1 - spatial.distance.cosine(vector_1, vector_2)

            # if cosin < 0, it is a gear change
            if compute_cosin < 0:
                change_gear += 1
                end = i+2
                input_path = final_path[start:end]

                if change_gear > 1 and have_extended_points > 0:
                    # add extend node into the input path
                    pre_path = split_path[-1]
                    for j in range(have_extended_points):
                        x_j = pre_path[-(have_extended_points-j)][0]
                        y_j = pre_path[-(have_extended_points-j)][1]
                        theta_j = pre_path[-(have_extended_points-j)][2]
                        input_path.insert(0, [x_j, y_j, theta_j])

                    have_extended_points = 0

                # extend points
                for j in range(extend_num):
                    forward_1 = (final_path[i+1][0] > final_path[i][0]) and (
                        final_path[i][2] > -np.pi/2 and final_path[i][2] < np.pi/2)
                    forward_2 = (final_path[i+1][0] < final_path[i][0]) and (
                        (final_path[i][2] > np.pi/2 and final_path[i][2] < np.pi) or (final_path[i][2] > -np.pi and final_path[i][2] < -np.pi/2))
                    if forward_1 or forward_2:
                        speed = self.vehicle.max_v
                    else:
                        speed = -self.vehicle.max_v

                    td_j = speed * self.planner.ddt * (j+1)
                    theta_j = final_path[i+1][2]
                    x_j = final_path[i+1][0] + td_j * np.cos(theta_j)
                    y_j = final_path[i+1][1] + td_j * np.sin(theta_j)

                    collision = self.collision_checker.check(node_x=x_j,
                                                             node_y=y_j,
                                                             theta=theta_j)

                    if not collision:
                        input_path.append([x_j, y_j, theta_j])
                        have_extended_points += 1

                split_path.append(input_path)
                start = i+1

        # add final episode path
        input_path = final_path[start:]
        pre_path = split_path[-1]

        if have_extended_points > 0:
            for j in range(have_extended_points):
                x_j = pre_path[-(have_extended_points-j)][0]
                y_j = pre_path[-(have_extended_points-j)][1]
                theta_j = pre_path[-(have_extended_points-j)][2]
                input_path.insert(0, [x_j, y_j, theta_j])

        split_path.append(input_path)

        return split_path, int(change_gear)
