# coding:utf-8
# Author: Yuansj
# Last update: 2022/08/25

from path_planner import path_planning
from animation import *
from costmap import _map, Vehicle
import velocity_plan
import cubic_interpolation
import path_optimazition
import ocp_optimization

import yaml
import os
import copy


def read_config():
    curPath = os.path.dirname(os.path.realpath(__file__))
    yamlPath = os.path.join(curPath, "config.yaml")
    f = open(yamlPath, 'r', encoding='utf-8')
    config = yaml.load(f.read(), Loader=yaml.FullLoader)
    return config


if __name__ == '__main__':
    # load configure file to a dict
    config = read_config()

    # read benchmark case
    file = os.path.join(config['Benchmark_path'], 'Case1.csv')

    # create the park map
    park_map = _map(file=file, discrete_size=config['map_discrete_size'])

    # path planning
    original_path, ego_vehicle, path_info, split_path = path_planning(
        config, park_map)

    # create velocity planner
    velocity_planner = velocity_plan.velocity_planner(vehicle=ego_vehicle)

    # create path optimization planner
    ocp_planner = ocp_optimization.ocp_optimization(
        park_map=park_map, vehicle=ego_vehicle, config=config)

    plot_opt_path = []  # store the optimization path

    plot_insert_path = []  # store the interpolation path

    path_optimizer = path_optimazition.path_opti(park_map, ego_vehicle, config)
    insert_point = cubic_interpolation.interpolation(
        config=config, map=park_map, vehicle=ego_vehicle)
    for path_i in split_path:
        # optimize path
        opti_path = path_optimizer.get_result(path_i)

        # compute cubic function
        

        # velocity planning
        velocity_func, acc_func = velocity_planner.solve_nlp(path=opti_path)

        # insert points
        insert_path = insert_point.get_cubic_interpolation(
            opti_path, velocity_func, acc_func)

        # ocp problem solve
        res = ocp_planner.solution(path=insert_path)

        plot_opt_path.extend(opti_path)
        plot_insert_path.extend(insert_path)

    # animation
    plot_final_path(path=original_path, map=park_map, color='green')

    # plot_final_path(path=plot_opt_path, map=park_map, color='blue')
    plot_final_path(path=plot_insert_path, map=park_map, color='red')
    park_map.visual_cost_map()
    plt.show()
    print('solved')

    ##### Debug ######
    # collision = distance_check(node_x=park_map.case.x0,node_y=park_map.case.y0,theta=park_map.case.theta0,map=park_map)
    # print(collision)
    # rs_curve.main(map=park_map)
    # dji = Dijkstra(park_map)
    # h_value, _  = dji.compute_path(park_map.case.x0, park_map.case.y0)
