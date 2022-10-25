'''
Author: wenqing-hnu
Date: 2022-10-20 00:01:21
LastEditors: wenqing-hnu
LastEditTime: 2022-10-25
FilePath: /TPCAP_demo_Python-main/velocity_planner/velocity_plan.py
Description: description for this file

Copyright (c) 2022 by wenqing-hnu, All Rights Reserved. 
'''

from abc import ABC, abstractmethod
from typing import Dict, List
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


class velocity_func_base(ABC):
    def __init__(self) -> None:
        super().__init__()
        pass

    @abstractmethod
    def obj_func(x):
        pass

    @abstractmethod
    def constraint():
        pass

    @abstractmethod
    def v_func():
        pass


class sin_func(velocity_func_base):
    '''
    description: 
    this function is :
        if 0 < t < pi / (2W) : v(t) = Asin(Wt)
        if pi / (2W) < t < t1 + pi / (2W): v(t) = A
        if t1 + pi / (2W) < t < t1 + pi / W: v(t) = Asin(W(t-t1))
    return {*} obj_func, constraint, x0
    '''

    def __init__(self) -> None:
        super().__init__()
        self.t1 = 0
        self.a = 0
        self.w = 0

    def initial_param(self, t1, a, w):
        self.t1 = t1
        self.a = a
        self.w = w
        self.t0 = np.pi / (2 * w)
        self.tf = t1 + np.pi / w

    def v_t(self, t):
        assert self.t1 != 0
        if t >= 0 and t < self.t0:
            v = self.a * np.sin(self.w * t)
        elif t >= self.t0 and t < (self.t0 + self.t1):
            v = self.a
        elif t >= (self.t0 + self.t1) and t < self.tf:
            v = self.a * np.sin(self.w * (t-self.t1))

        return v

    @staticmethod
    def obj_func(x):
        '''
        description: the objective function
        param {*} x is a vecor: [t1,A,W]
        return {*} obj_func
        '''

        return x[0] + np.pi / x[2]

    @staticmethod
    def constraint(max_v, max_a, arc_length) -> Dict:
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

        return cons


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
