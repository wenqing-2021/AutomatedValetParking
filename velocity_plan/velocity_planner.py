'''
Author: wenqing-hnu
Date: 2022-10-20
LastEditors: wenqing-hnu
LastEditTime: 2022-11-12
FilePath: /Automated Valet Parking/velocity_plan/velocity_planner.py
Description: velocity planner for the path

Copyright (c) 2022 by wenqing-hnu, All Rights Reserved. 
'''


from abc import ABC, abstractmethod
from typing import Dict, List, Tuple
from map.costmap import Vehicle
import numpy as np
from scipy.optimize import minimize
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
    def v_a_func():
        '''
        description: build the velocity and acceleration function
        return {*} the velocity and the acceleration
        '''
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

    def v_a_func(self, t):
        assert self.t1 != 0, 't1 should not be zero'

        if t >= 0 and t < self.t0:
            v = self.a * np.sin(self.w * t)
            acc = self.a * self.w * np.cos(self.w * t)
        elif t >= self.t0 and t < (self.t0 + self.t1):
            v = self.a
            acc = 0
        elif t >= (self.t0 + self.t1) and t <= self.tf:
            v = self.a * np.sin(self.w * (t-self.t1))
            acc = self.a * self.w * np.cos(self.w * (t-self.t1))

        return v, acc

    def obj_func(self):
        '''
        description: the objective function
        param {*} x is a vecor: [t1,A,W]
        return {*} obj_func
        '''
        return lambda x: x[0] + np.pi / x[2]
        # return x[0] + np.pi / x[2]

    def constraint(self, max_v, max_a, arc_length) -> Dict:
        cons = ({"type": "ineq", "fun": lambda x: x[0] - e},  # t1 > 0
                {"type": "ineq", "fun": lambda x: x[1] - e},  # A > 0
                {"type": "ineq", "fun": lambda x: x[2] - e},  # W > 0
                # v < max velocity
                {"type": "ineq", "fun": lambda x: max_v - x[1]},
                {"type": "ineq",
                    "fun": lambda x: max_a-x[1]*x[2]},  # a < max acceleration
                # goal pose velocity is zero,
                {"type": "eq", "fun": lambda x: arc_length -
                    x[0]*x[1]-2*x[1]/x[2]},  # distance constraints
                )

        return cons


class VelocityPlanner:
    def __init__(self,
                 vehicle: Vehicle,
                 velocity_func_type: str = 'sin_func'):
        '''
        description: the velocity function type is sin func
        return {*} None
        '''
        self.vehicle = vehicle
        self.max_acceleration = vehicle.max_acc
        self.max_v = vehicle.max_v
        self.plan_result = dict()
        if velocity_func_type == velocity_type.sin_func.name:
            self.v_func = sin_func()
        else:
            raise Exception("the velocity function type is not defined")

    def solve_nlp(self,
                  arc_length: np.float64 = None):
        '''
        description: solve a nlp problem to find the minimum travel time 
        and the optimal velocity function
        return {*} the velocity function and the terminate time
        '''

        # def fun(x): return (x[0] + (x[1]*x[2])**2/2 *
        #                     x[0] + x[1]/4*x[2]**2*math.sin(2*x[1]*x[0]))

        x0 = np.array((2.0, 0.5, 2.0))

        obj_fun = self.v_func.obj_func()
        cons = self.v_func.constraint(max_a=self.max_acceleration,
                                      max_v=self.max_v,
                                      arc_length=arc_length)

        result = minimize(fun=obj_fun, x0=x0, method="SLSQP", constraints=cons)
        optimal_solve = result.x
        t1 = optimal_solve[0]
        a = optimal_solve[1]
        w = optimal_solve[2]

        terminate_t = t1 + np.pi / w

        print('terminate_time:', terminate_t)

        self.plan_result = {"A": a, "W": w, "t1": t1}
        self.v_func.initial_param(t1, a, w)

        return self.v_func.v_a_func, terminate_t
