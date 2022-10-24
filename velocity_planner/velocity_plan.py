# coding:utf-8
# Author: Yuansj
# Last update:2022/07/04

from typing import List
from map.costmap import Vehicle
import numpy as np
from scipy.optimize import minimize
import math
from enum import Enum, unique

e = 1e-10


@unique
class velocity_type(Enum):
    sin_func = 1
    constant_func = 2
    double_s_func = 3

    @staticmethod
    def sin_func(max_v, max_a, arc_length):
        '''
        description: 
        this function is :
            if 0 < t < pi / (2W) : v(t) = Asin(Wt)
            if pi / (2W) < t < t1 + pi / (2W): v(t) = A
            if t1 + pi / (2W) < t < t1 + pi / W: v(t) = Asin(W(t-t1))
        return {*} obj_func, constraint, x0
        '''

        def obj_func(x):
            '''
            description: the objective function
            param {*} x is a vecor: [t1,A,W]
            return {*} obj_func
            '''

            return x[0] + np.pi / x[2]

        def constraint(max_v, max_a, arc_length):
            cons = ({"type": "ineq", "fun": lambda x: x[0] - e},  # t1 > 0
                    {"type": "ineq", "fun": lambda x: x[1] - e},  # A > 0
                    {"type": "ineq", "fun": lambda x: x[2] - e},  # W > 0
                    # v < max velocity
                    {"type": "ineq", "fun": lambda x: max_v - x[1]},
                    {"type": "ineq", "fun": lambda x: x[1]*x[2]-e},  # Aw > 0
                    {"type": "ineq",
                        "fun": lambda x: max_a-x[1]*x[2]},  # a < max acceleration
                    # goal pose velocity is zero,
                    {"type": "eq", "fun": lambda x: arc_length -
                        x[0]*x[1]+2*x[1]/x[2]},  # distance constraints
                    )


class velocity_planner:
    def __init__(self,
                 vehicle: Vehicle):
        self.vehicle = vehicle
        self.max_acceleration = vehicle.max_acc
        self.max_v = vehicle.max_v
        self.plan_result = dict()
        self.func_type = velocity_type()

    # use v(t) = Asin(wt) to plan the velocity, a(t) = Aw cos(wt)
    def solve_nlp(self,
                  path: List[List] = None,
                  arc_length: np.float64 = None):
        s = arc_length

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

    def velocity_func(self,
                      func_type: int = 1):
        for _type in self.func_type:
            if func_type == _type.value:
                print('chose_velocity_func is:', _type.name)
                break
