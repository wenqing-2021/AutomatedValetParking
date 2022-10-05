import copy
import hybird_a_star
from animation import plot_collision_p
from costmap import _map
import numpy as np
from scipy import spatial
from collision_check import distance_check, two_circle_check

# main loop
def path_planning(config:dict, park_map:_map):
    # plot_obstacles(park_map)

    ## start hybrid A star algorithm
    solver = hybird_a_star.hybrid_a_star(config=config,
                                         park_map=park_map)

    reach_goal = False

    while not solver.open_list.empty() and not reach_goal:
        # get current node
        current_node = solver.open_list.get()
        # show info
        print('---------------')
        print('current node index:', current_node.index)
        print('distance:', np.sqrt((current_node.x-park_map.case.xf)**2 + (current_node.y - park_map.case.yf)**2))
        print('---------------')

        rs_path, collision, info = solver.try_reach_goal(current_node)

        # plot the collision position
        if collision and config['draw_collision']:
            collision_p = info['collision_position']
            plot_collision_p(collision_p[0], collision_p[1], collision_p[2], park_map)
            
        if not collision and info['in_radius']:
            reach_goal = True
            break
            
        else:
            # expand node
            child_group = solver.expand_node(current_node)
            path = []
            for i in child_group.queue:
                x = i.x
                y = i.y
                theta = i.theta
                path.append([x,y,theta])
    
    _path = solver.finish_path(current_node)
    final_path = copy.deepcopy(_path)
    
    # assemble all path
    for i in range(1, len(rs_path.x)):
        x = rs_path.x[i]
        y = rs_path.y[i]
        theta = rs_path.yaw[i]
        final_path.append([x,y,theta])
    
    # split path based on the gear (forward or backward)
    split_path = []
    change_gear = 0
    start = 0
    extend_num = config['extended_num']
    # we want to extend node but these points also need to check
    have_extended_points = 0

    for i in range(len(final_path) - 2):
        vector_1 = (final_path[i+1][0] - final_path[i][0],\
                    final_path[i+1][1] - final_path[i][1])
        
        vector_2 = (final_path[i+2][0] - final_path[i+1][0],\
                    final_path[i+2][1] - final_path[i+1][1])
        
        compute_cosin = 1 - spatial.distance.cosine(vector_1, vector_2)
        
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
                if final_path[i+1][0] > final_path[i][0]:
                    speed = solver.vehicle.max_v
                else:
                    speed = -solver.vehicle.max_v
                
                td_j = speed * solver.ddt * (j+1)
                theta_j = final_path[i+1][2]
                x_j = final_path[i+1][0] + td_j * np.cos(theta_j)
                y_j = final_path[i+1][1] + td_j * np.sin(theta_j)
                
                if config['collision_check'] == 'circle':
                    collision = two_circle_check(node_x=x_j,
                                                 node_y=y_j,
                                                 theta=theta_j,
                                                 map=park_map)
                else:
                    collision = distance_check(node_x=x_j,
                                               node_y=y_j,
                                               theta=theta_j,
                                               map=park_map,
                                               config=config)
                
                if not collision:
                    input_path.append([x_j, y_j, theta_j])
                    have_extended_points += 1

            split_path.append(input_path)
            start = i+1
    
    input_path = final_path[start:]
    pre_path = split_path[-1]

    if have_extended_points > 0:
        for j in range(have_extended_points):
            x_j = pre_path[-(have_extended_points-j)][0]
            y_j = pre_path[-(have_extended_points-j)][1]
            theta_j = pre_path[-(have_extended_points-j)][2]
            input_path.insert(0, [x_j, y_j, theta_j])

    split_path.append(input_path)
        
    path_info = {"a_start_path": _path, "rs_path":rs_path,
                 "change_gear":change_gear}
    
    out_final_path = copy.deepcopy(split_path[0])
    for k in range(1, len(split_path)):
        out_final_path.extend(split_path[k])

    return out_final_path, solver.vehicle, path_info, split_path