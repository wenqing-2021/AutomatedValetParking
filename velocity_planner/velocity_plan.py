# coding:utf-8
# Author: Yuansj
# Last update:2022/07/04

from map.costmap import Vehicle
import numpy as np
from scipy.optimize import minimize
import math


class velocity_planner:
    def __init__(self,
                 vehicle: Vehicle):
        self.vehicle = vehicle
        self.max_acceleration = vehicle.max_acc
        self.max_v = vehicle.max_v
        self.plan_result = dict()

    def compute_traj_length(self, path):
        traj_length = 0
        for i in range(len(path) - 1):
            s_i = math.sqrt((path[i][0] - path[i+1][0]) **
                            2+(path[i][1] - path[i+1][1])**2)
            traj_length = traj_length + s_i
        return traj_length

    # use v(t) = Asin(wt) to plan the velocity, a(t) = Aw cos(wt)
    def solve_nlp(self, path):
        s = self.compute_traj_length(path)
        e = 1e-10

        # def fun(x): return (x[0] + (x[1]*x[2])**2/2 *
        #                     x[0] + x[1]/4*x[2]**2*math.sin(2*x[1]*x[0]))

        def fun(x): return (x[0])
        cons = ({"type": "ineq", "fun": lambda x: x[2] - e},  # t0 > 0
                {"type": "ineq", "fun": lambda x: x[0] - e},  # A > 0
                {"type": "ineq", "fun": lambda x: x[1] - e},  # W > 0
                # v < max velocity
                {"type": "ineq", "fun": lambda x: self.max_v-x[2]},
                {"type": "ineq", "fun": lambda x: x[1]*x[2]-e},  # Aw > 0
                {"type": "ineq",
                    "fun": lambda x: self.max_acceleration-x[1]*x[2]},  # a < max acceleration
                # goal pose velocity is zero
                {"type": "eq", "fun": lambda x: x[0]*x[1] - math.pi},
                {"type": "eq", "fun": lambda x: s -
                    x[2]/x[1]+x[2]/x[1]*math.cos(x[1]*x[0])},  # distance constraints
                )
        x0 = np.array((4.0, 0.5, 2.0))
        result = minimize(fun=fun, x0=x0, method="SLSQP", constraints=cons)
        optimal_solve = result.x
        A = optimal_solve[2]
        W = optimal_solve[1]
        terminate_t = optimal_solve[0]

        print('terminate_time:', terminate_t)

        self.plan_result = {"A": A, "W": W, "t1": terminate_t}

        def velocity_func(t):
            '''
            :param dt: input the moment
            :return: velocity
            '''
            return A * math.sin(W * t)

        def acc_func(t):
            '''
            :param dt: input the moment
            :return: acceleration function
            '''
            return A*W*math.cos(W*t)

        return velocity_func, acc_func

        # debug
        # t_i = np.linspace(0, terminate_t, 1000)
        # dt = terminate_t / len(t_i)
        # new_s = 0
        # for i in range(len(t_i)):
        #     new_s = new_s + dt * (A*np.sin(W*t_i[i]))
        # print('最小值：', result.fun)
        # print('最优解：', result.x)
        # print('迭代终止是否成功：', result.success)
        # print('迭代终止原因：', result.message)


class velocity_optimize:
    def __init__(self) -> None:
        pass
