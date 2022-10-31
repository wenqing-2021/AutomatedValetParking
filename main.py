'''
Author: wenqing-hnu
Date: 2022-10
LastEditors: wenqing-hnu
LastEditTime: 2022-10-31
FilePath: /TPCAP_demo_Python-main/main.py
Description: main func for trajectory planning

Copyright (c) 2022 by wenqing-hnu, All Rights Reserved. 
'''


from path_planner import path_planner
from animation.animation import *
from map import costmap
from velocity_planner import velocity_plan
from interpolation import cubic_interpolation
from optimization import path_optimazition, ocp_optimization
from config import read_config

import yaml
import os
import copy

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='hybridAstar')
    parser.add_argument("--config_name", type=str, default="config")
    args = parser.parse_args()

    # initial
    # load configure file to a dict
    config = read_config.read_config(config_name=args.config_name)

    # read benchmark case
    file = os.path.join(config['Benchmark_path'], 'Case4.csv')

    # create the park map
    park_map = costmap._map(
        file=file, discrete_size=config['map_discrete_size'])

    # create vehicle
    ego_vehicle = costmap.Vehicle()

    # create path planner
    planner = path_planner.path_planner(config=config,
                                        map=park_map,
                                        vehicle=ego_vehicle)

    # create path optimizer
    path_optimizer = path_optimazition.path_opti(park_map, ego_vehicle, config)

    # create path interpolation
    interplotor = cubic_interpolation.interpolation(
        config=config, map=park_map, vehicle=ego_vehicle)

    # create velocity planner
    velocity_planner = velocity_plan.velocity_planner(vehicle=ego_vehicle,
                                                      velocity_func_type='sin_func')

    # create path optimization planner
    ocp_planner = ocp_optimization.ocp_optimization(
        park_map=park_map, vehicle=ego_vehicle, config=config)

    # rapare memory to store path
    plot_opt_path = []  # store the optimization path
    plot_insert_path = []  # store the interpolation path
    plot_ocp_path = []  # store ocp path

    # path planning
    optimal_tf = 0
    t = 0
    optimal_time_info = []
    original_path, path_info, split_path = planner.path_planning()
    for path_i in split_path:
        # optimize path
        opti_path, forward = path_optimizer.get_result(path_i)

        # cubic fitting
        path_arc_length, path_i_info = interplotor.cubic_fitting(opti_path)

        # velocity planning
        v_acc_func, terminiate_time = velocity_planner.solve_nlp(
            arc_length=path_arc_length)

        # insert points
        insert_path = interplotor.cubic_interpolation(
            path=opti_path, path_i_info=path_i_info, v_a_func=v_acc_func, forward=forward, terminate_t=terminiate_time, path_arc_length=path_arc_length)

        # ocp problem solve
        ocp_traj, optimal_ti, optimal_dt = ocp_planner.solution(
            path=insert_path)
        optimal_time_info.append([optimal_ti, optimal_dt])
        # add time information
        for ocp_i in range(len(ocp_traj)):
            t += optimal_dt
            ocp_traj[ocp_i].append(t)
        optimal_tf += optimal_ti

        plot_opt_path.extend(opti_path)
        plot_insert_path.extend(insert_path)
        plot_ocp_path.extend(ocp_traj)

    # animation
    print('trajectory_time:', optimal_tf)
    plot_obstacles(map=park_map)
    park_map.visual_cost_map()
    plot_final_path(path=original_path, map=park_map,
                    color='green', show_car=False)
    plot_final_path(path=plot_opt_path, map=park_map,
                    color='blue', show_car=False)
    plot_final_path(path=plot_insert_path, map=park_map,
                    color='red', show_car=False)
    plot_final_path(path=plot_ocp_path, map=park_map,
                    color='gray', show_car=True)

    plt.show()
    print('solved')
