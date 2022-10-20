# coding:utf-8
# Author: Yuansj
# Last update: 2022/10/20

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
    file = os.path.join(config['Benchmark_path'], 'Case1.csv')

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
    insert_point = cubic_interpolation.interpolation(
        config=config, map=park_map, vehicle=ego_vehicle)

    # create velocity planner
    velocity_planner = velocity_plan.velocity_planner(vehicle=ego_vehicle)

    # create path optimization planner
    # ocp_planner = ocp_optimization.ocp_optimization(
    #     park_map=park_map, vehicle=ego_vehicle, config=config)

    # rapare memory to store path
    plot_opt_path = []  # store the optimization path
    plot_insert_path = []  # store the interpolation path
    # plot_ocp_path = []  # store ocp path

    # path planning
    original_path, path_info, split_path = planner.path_planning()
    for path_i in split_path:
        # optimize path
        opti_path = path_optimizer.get_result(path_i)

        # cubic fitting

        # get the total curve length

        # velocity planning
        velocity_func, acc_func = velocity_planner.solve_nlp(path=opti_path)

        # insert points
        insert_path = insert_point.get_cubic_interpolation(
            opti_path, velocity_func, acc_func)

        # ocp problem solve
        # ocp_traj, ocp_tf = ocp_planner.solution(path=insert_path)

        plot_opt_path.extend(opti_path)
        plot_insert_path.extend(insert_path)
        # plot_ocp_path.extend(ocp_traj)

    # animation
    plot_final_path(path=original_path, map=park_map, color='green')
    # plot_final_path(path=plot_opt_path, map=park_map, color='blue')
    plot_final_path(path=plot_insert_path, map=park_map,
                    color='red', show_car=True)
    # plot_final_path(path=plot_ocp_path, map=park_map, color='gray')
    park_map.visual_cost_map()
    plt.show()
    print('solved')
