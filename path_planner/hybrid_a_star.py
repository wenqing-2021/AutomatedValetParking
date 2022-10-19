# coding:utf-8
# Author: Yuansj
# start: 2022/06/04
# update 2022/06/13: finish path_planning V1.0
# update 2022/06/15: V1.1 fix collision check function and it works for case 1
# Note: V1.1 does not include speed planning
# update 2022/06/25: V1.2 add path optimization but we do not have curvature limit

import numpy as np
import math
import queue
from map.costmap import _map, Vehicle
from collision_check import collision_check
from path_planner.compute_h import Dijkstra
from path_planner import rs_curve
from animation.animation import *


class Node:
    '''
    Node contains: 
                position(x,y);
                vehicle heading theta;
                node index;
                node father index;
                node child index;
                is forward: true or false
                steering angle: rad
                f,g,h value
    '''

    def __init__(self,
                 index: np.int32 = None,
                 x: np.float64 = 0.0,
                 y: np.float64 = 0.0,
                 theta: np.float64 = 0.0,
                 parent_index: np.int32 = None,
                 child_index: np.int32 = None,
                 is_in_openlist: bool = False,
                 is_in_closedlist: bool = False,
                 is_forward: bool = True,
                 steering_angle: np.float64 = None) -> None:

        self.index = index
        self.x = x
        self.y = y
        self.theta = theta
        self.parent_index = parent_index
        self.child_index = child_index
        self.in_open = is_in_openlist
        self.in_closed = is_in_closedlist
        self.forward = is_forward
        self.steering_angle = steering_angle
        self.h = 0
        self.g = 0
        self.f = 0

    def __lt__(self, other):
        '''
        revise compare function for PriorityQueue
        '''
        result = False
        if self.f < other.f:
            result = True
        return result


class hybrid_a_star:
    def __init__(self,
                 config: dict,
                 park_map: _map,
                 vehicle: Vehicle) -> None:

        # create vehicle
        self.vehicle = vehicle

        # discrete steering angle
        self.steering_angle = np.linspace(- self.vehicle.max_steering_angle,
                                          self.vehicle.max_steering_angle,
                                          config['steering_angle_num'])  # rad

        # park_map
        self.park_map = park_map

        # caculate heuristic and store h value
        self.heuristic = Dijkstra(park_map)
        _, self.h_value_list = self.heuristic.compute_path(
            node_x=park_map.case.x0, node_y=park_map.case.y0)

        # default settings
        self.global_index = 0
        self.config = config
        self.open_list = queue.PriorityQueue()
        self.closed_list = []
        self.dt = config['dt']
        self.ddt = config['trajectory_dt']

        # initial node
        self.initial_node = Node(x=park_map.case.x0,
                                 y=park_map.case.y0,
                                 index=0,
                                 theta=rs_curve.pi_2_pi(park_map.case.theta0))
        # final node
        self.goal_node = Node(x=park_map.case.xf,
                              y=park_map.case.yf,
                              theta=rs_curve.pi_2_pi(park_map.case.thetaf))

        self.open_list.put(self.initial_node)
        self.initial_node.in_open = True

        # max delta heading
        self.max_delta_heading = self.vehicle.max_v * \
            np.tan(self.vehicle.max_steering_angle) / self.vehicle.lw * self.dt

        # create collision checker
        if self.config['collision_check'] == 'circle':
            self.collision_checker = collision_check.two_circle_checker(
                vehicle=self.vehicle, map=self.park_map, config=config)
        else:
            self.collision_checker = collision_check.distance_checker(
                vehicle=self.vehicle, map=self.park_map, config=config)

    def expand_node(self,
                    current_node: Node) -> queue.PriorityQueue:
        # caculate <x,y,theta> of the next node
        # next_index = 9 or 10(the first expansion)
        child_group = queue.PriorityQueue()
        next_index = 0
        travle_distance = 0  # v_max * dt
        next_index = int(2 * self.config['steering_angle_num'])
        for i in range(next_index):
            # caculate steering angle and gear
            steering_angle = self.steering_angle[i %
                                                 self.config['steering_angle_num']]
            if i < next_index / 2:
                speed = self.vehicle.max_v
                is_forward = True
            else:
                speed = - self.vehicle.max_v
                is_forward = False

            travle_distance = speed * self.dt
            theta_ = current_node.theta + \
                (self.vehicle.max_v * np.tan(steering_angle)) / \
                self.vehicle.lw * self.dt
            theta_ = rs_curve.pi_2_pi(theta_)
            x_ = current_node.x + travle_distance * np.cos(theta_)
            y_ = current_node.y + travle_distance * np.sin(theta_)

            # if the node is in closedlist or this node beyond the boundary, continue
            find_closednode = False
            for closednode_i in self.closed_list:
                if closednode_i.x == x_ and closednode_i.y == y_ and closednode_i.theta == theta_:
                    find_closednode = True
                    break
                # if beyond the boundary
                elif x_ > self.park_map.boundary[1] or x_ < self.park_map.boundary[0] or \
                        y_ > self.park_map.boundary[3] or y_ < self.park_map.boundary[2]:
                    find_closednode = True
                    break
            if find_closednode == True:
                continue
            else:
                find_opennode = False
                # find node in the open list
                for opennode_i in self.open_list.queue:
                    if opennode_i.x == x_ and opennode_i.y == y_ and opennode_i.theta == theta_:
                        child_node = opennode_i
                        find_opennode = True

            # if the node is firstly visited
            if find_opennode == False:
                # generate new node
                child_node = Node(x=x_,
                                  y=y_,
                                  theta=theta_,
                                  index=self.global_index + i + 1,
                                  parent_index=current_node.index,
                                  is_forward=is_forward,
                                  steering_angle=steering_angle)
                # collision check
                for i in range(math.ceil(self.dt / self.ddt)):
                    # discrete trajectory for collision check
                    # i : 0-9
                    travle_distance_i = speed * self.ddt * (i+1)
                    theta_i = current_node.theta + \
                        (self.vehicle.max_v * np.tan(steering_angle)) / \
                        self.vehicle.lw * self.ddt * (i+1)
                    theta_i = rs_curve.pi_2_pi(theta_i)
                    x_i = current_node.x + travle_distance_i * np.cos(theta_i)
                    y_i = current_node.y + travle_distance_i * np.sin(theta_i)

                    # collision check
                    collision = self.collision_checker.check(
                        node_x=x_i, node_y=y_i, theta=theta_i)

                    if collision:
                        # put the node into the closedlist
                        self.closed_list.append(child_node)
                        child_node.in_closed = True
                        break

                if not collision:
                    # caculate cost
                    child_node.g = self.calc_node_cost(
                        child_node, father_theta=current_node.theta, father_gear=current_node.forward)
                # caculate heuristic
                    child_node.h = self.calc_node_heuristic(child_node)
                # caculate f value
                    child_node.f = child_node.g + child_node.h
                # add this node into openlist
                    self.open_list.put(child_node)
                    child_node.in_open = True

            # if this node has been explored
            else:
                new_h = self.calc_node_heuristic(child_node)
                new_g = self.calc_node_cost(
                    child_node, father_theta=current_node.theta, father_gear=current_node.forward)
                new_f = new_h + new_g
                if new_f < child_node.f:
                    child_node.f = new_f
                    child_node.g = new_g
                    child_node.h = new_h
                    child_node.parent_index = current_node.index
                    child_node.forward = is_forward
                    child_node.steering_angle = steering_angle
            if child_node.in_closed == False and child_node.in_open == True:
                child_group.put(child_node)

        # put the current node into closed list
        current_node.in_closed = True
        current_node.in_open = False
        self.closed_list.append(current_node)

        self.global_index += next_index

        return child_group

    def calc_node_cost(self, node: Node, father_theta, father_gear) -> np.float64:
        '''
        input: child node
        output: the cost value of this node
        We consider two factors, gear and the delta of heading
        '''
        cost = 0
        cost_gear = 0
        gear = node.forward
        if gear != father_gear:
            cost_gear = 1

        cost_heading = abs(node.theta - father_theta)

        cost = cost_gear + 0.5 * cost_heading

        return 10 * cost

    def calc_node_heuristic(self, current_node: Node) -> np.float64:
        '''
        We use Dijkstra algorithm and RS curve length to calculate the heuristic value 
        '''
        # convert node to grid
        grid_x = np.float64("%.1f" % (current_node.x + 0.05))
        grid_y = np.float64("%.1f" % (current_node.y + 0.05))
        h_value = 0
        find_grid = False
        for i in range(len(self.h_value_list)):
            find_x = self.h_value_list[i].grid_x == grid_x
            find_y = self.h_value_list[i].grid_y == grid_y
            if find_x and find_y:
                find_grid = True
                h_value_1 = self.h_value_list[i].distance
                break
        if find_grid == False:
            h_value_1, self.h_value_list = self.heuristic.compute_path(
                node_x=current_node.x, node_y=current_node.y)

        max_c = 1 / self.vehicle.min_radius_turn
        rs_path = rs_curve.calc_optimal_path(sx=current_node.x,
                                             sy=current_node.y,
                                             syaw=current_node.theta,
                                             gx=self.goal_node.x,
                                             gy=self.goal_node.y,
                                             gyaw=self.goal_node.theta,
                                             maxc=max_c)

        h_value_2 = rs_path.L
        h_value_1 = h_value_1 / 100
        h_value = max(h_value_1, h_value_2)

        return h_value

    def try_reach_goal(self, current_node: Node) -> bool:
        '''
        if node is near the goal node, we check whether the rs curve could reach it
        '''
        collision = False
        rs_path = None
        in_radius = False
        collision_p = None
        distance = np.sqrt((current_node.x - self.goal_node.x)
                           ** 2+(current_node.y-self.goal_node.y)**2)
        if distance < self.config['flag_radius']:
            in_radius = True
            rs_path, collision, collision_p = self.try_rs_curve(current_node)

        info = {'in_radius': in_radius,
                'collision_position': collision_p}
        return rs_path, collision, info

    def try_rs_curve(self, current_node: Node):
        '''
        generate rs curve and collision check
        return: rs_path is a class and collision is true or false
        '''
        collision = False
        # generate max curvature based on min turn radius
        max_c = 1 / self.vehicle.min_radius_turn
        rs_path = rs_curve.calc_optimal_path(sx=current_node.x,
                                             sy=current_node.y,
                                             syaw=current_node.theta,
                                             gx=self.goal_node.x,
                                             gy=self.goal_node.y,
                                             gyaw=self.goal_node.theta,
                                             maxc=max_c)

        # collision check
        for i in range(len(rs_path.x)):
            path_x = rs_path.x[i]
            path_y = rs_path.y[i]
            path_theta = rs_path.yaw[i]
            path_theta = rs_curve.pi_2_pi(path_theta)
            collision = self.collision_checker.check(
                node_x=path_x, node_y=path_y, theta=path_theta)

            if collision:
                collision_position = [path_x, path_y, path_theta]
                break
            else:
                collision_position = None

        return rs_path, collision, collision_position

    def finish_path(self, current_node: Node):
        node = current_node
        all_path_node = []
        while node.index != 0:
            all_path_node.append(node)
            parent_index = node.parent_index
            for node_i in self.closed_list:
                if node_i.index == parent_index:
                    node = node_i
                    break
        all_path_node.append(node)

        all_path = [[node.x, node.y, node.theta]]

        for i in range(len(all_path_node)):
            # k is index
            k = len(all_path_node) - 1 - i
            if k == 0:
                break
            for j in range(math.ceil(self.dt/self.ddt)):
                # discrete trajectory to store each waypoint
                # i : 0-9
                if all_path_node[k-1].forward:
                    speed = self.vehicle.max_v
                else:
                    speed = -self.vehicle.max_v

                td_j = speed * self.ddt * (j+1)
                theta_0 = all_path_node[k].theta
                steering_angle = all_path_node[k-1].steering_angle
                theta_j = theta_0 + \
                    (self.vehicle.max_v * np.tan(steering_angle)) / \
                    self.vehicle.lw * self.ddt * (j+1)
                theta_j = rs_curve.pi_2_pi(theta_j)
                x_j = all_path_node[k].x + td_j * np.cos(theta_j)
                y_j = all_path_node[k].y + td_j * np.sin(theta_j)
                all_path.append([x_j, y_j, theta_j])

        return all_path
