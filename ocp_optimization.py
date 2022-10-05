# Author: Yuansj
# 2022/10.01

'''
This function is used to find an optimal traj for the parking. the initial solution is from path_optimazition.py.
the usage of cyipopt is https://cyipopt.readthedocs.io/en/stable/tutorial.html#problem-interface
'''

import jax.numpy as np
from jax import jit, grad, jacfwd
from cyipopt import minimize_ipopt
from costmap import _map, Vehicle


class ocp_optimization:
    def __init__(self,
                 initial_path,
                 park_map: _map,
                 vehicle: Vehicle) -> None:
        self.initial_solution = initial_path
        self.park_map = park_map
        self.vehicle = vehicle

    def solution(self):

        def objective(x):
            '''
            the objective funtion is min: t + a^2+w^2+v^2 + \sigma^2
            input: x, y, theta, v, a, delta, w, ..., tf
            '''
            t = x[-1]
            length = len(x) - 1
            n = length / 7
            vector_v = np.array([0, 0, 0, 1, 0, 0, 0])
            vector_a = np.array([0, 0, 0, 0, 1, 0, 0])
            vector_s = np.array([0, 0, 0, 0, 0, 1, 0])
            vector_w = np.array([0, 0, 0, 0, 0, 0, 1])
            Matrix_v = np.zeros((1, length))
            Matrix_a = np.zeros((1, length))
            Matrix_s = np.zeros((1, length))
            Matrix_w = np.zeros((1, length))

            for i in range(n):
                previous = np.zeros(7 * i)
                after = np.zeros(7 * (n-1-i))

                row_a = np.hstack((previous, vector_a, after, 0))
                Matrix_a = np.vstack((Matrix_a, row_a))

                row_v = np.hstack((previous, vector_v, after, 0))
                Matrix_v = np.vstack((Matrix_v, row_v))

                row_s = np.hstack((previous, vector_s, after, 0))
                Matrix_s = np.vstack((Matrix_s, row_s))

                row_w = np.hstack((previous, vector_w, after, 0))
                Matrix_w = np.vstack((Matrix_w, row_w))

            Matrix_a = Matrix_a[1:]
            Matrix_v = Matrix_v[1:]
            Matrix_s = Matrix_s[1:]
            Matrix_w = Matrix_w[1:]

            a2 = np.inner(Matrix_a, x)
            a2_sum = np.inner(a2, a2)

            v2 = np.inner(Matrix_v, x)
            v2_sum = np.inner(v2, v2)

            s2 = np.inner(Matrix_s, x)
            s2_sum = np.inner(s2, s2)

            w2 = np.inner(Matrix_w, x)
            w2_sum = np.inner(w2, w2)

            return t + a2_sum + v2_sum + s2_sum + w2_sum

        def kinematic_constraints_x(x):
            '''
            kinematic constraint
            '''
            N = (len(x) - 1) / 7
            dt = x[-1] / (N - 1)  # tf / (N -1 )
            vector_x = np.array([1, 0, 0, 0, 0, 0, 0])
            vector_v = np.array([0, 0, 0, 1, 0, 0, 0])
            vector_theta = np.array([0, 0, 1, 0, 0, 0, 0])
            Matrix_x = np.zeros((1, N))
            Matrix_v = np.zeros((1, N))
            Matrix_theta = np.zeros((1, N))
            for i in range(N):
                previous = np.zeros(7 * i)
                after = np.zeros(7 * (N-1-i))

                row_x = np.hstack((previous, vector_x, after, 0))
                row_v = np.hstack((previous, vector_v, after, 0))
                row_theta = np.hstack((previous, vector_theta, after, 0))

                Matrix_x = np.vstack((Matrix_x, row_x))
                Matrix_v = np.vstack((Matrix_v, row_v))
                Matrix_theta = np.vstack((Matrix_theta, row_theta))

            Matrix_x = Matrix_x[1:]
            Matrix_v = Matrix_v[1:]
            Matrix_theta = Matrix_theta[1:]
            all_x = np.inner(Matrix_x, x)
            all_v = np.inner(Matrix_v, x)
            all_theta = np.inner(Matrix_theta, x)

            # all_x[1:] = all_x[:-1] + all_v[:-1] * dt * np.cos(all_theta[:-1])

            return all_x[1:] - (all_x[:-1] + all_v[:-1] * dt * np.cos(all_theta[:-1]))

        def kinematic_constraints_y(x):
            '''
            kinematic constraint
            '''
            N = (len(x) - 1) / 7
            dt = x[-1] / (N - 1)  # tf / (N -1 )
            vector_y = np.array([0, 1, 0, 0, 0, 0, 0])
            vector_v = np.array([0, 0, 0, 1, 0, 0, 0])
            vector_theta = np.array([0, 0, 1, 0, 0, 0, 0])
            Matrix_y = np.zeros((1, N))
            Matrix_v = np.zeros((1, N))
            Matrix_theta = np.zeros((1, N))
            for i in range(N):
                previous = np.zeros(7 * i)
                after = np.zeros(7 * (N-1-i))

                row_y = np.hstack((previous, vector_y, after, 0))
                row_v = np.hstack((previous, vector_v, after, 0))
                row_theta = np.hstack((previous, vector_theta, after, 0))

                Matrix_y = np.vstack((Matrix_y, row_y))
                Matrix_v = np.vstack((Matrix_v, row_v))
                Matrix_theta = np.vstack((Matrix_theta, row_theta))

            Matrix_y = Matrix_y[1:]
            Matrix_v = Matrix_v[1:]
            Matrix_theta = Matrix_theta[1:]
            all_y = np.inner(Matrix_y, x)
            all_v = np.inner(Matrix_v, x)
            all_theta = np.inner(Matrix_theta, x)

            # all_y[1:] = all_y[:-1] + all_v[:-1] * dt * np.sin(all_s[:-1])

            return all_y[1:] - (all_y[:-1] + all_v[:-1] * dt * np.sin(all_theta[:-1]))

        def kinematic_constraints_theta(x):
            '''
            kinematic constraint
            '''
            Lw = 2.8
            N = (len(x) - 1) / 7
            dt = x[-1] / (N - 1)  # tf / (N -1 )
            vector_theta = np.array([0, 0, 1, 0, 0, 0, 0])
            vector_v = np.array([0, 0, 0, 1, 0, 0, 0])
            vector_s = np.array([0, 0, 0, 0, 0, 1, 0])
            Matrix_theta = np.zeros((1, N))
            Matrix_v = np.zeros((1, N))
            Matrix_s = np.zeros((1, N))
            for i in range(N):
                previous = np.zeros(7 * i)
                after = np.zeros(7 * (N-1-i))

                row_theta = np.hstack((previous, vector_theta, after, 0))
                row_v = np.hstack((previous, vector_v, after, 0))
                row_s = np.hstack((previous, vector_s, after, 0))

                Matrix_theta = np.vstack((Matrix_theta, row_theta))
                Matrix_v = np.vstack((Matrix_v, row_v))
                Matrix_s = np.vstack((Matrix_s, row_s))

            Matrix_theta = Matrix_theta[1:]
            Matrix_v = Matrix_v[1:]
            Matrix_s = Matrix_s[1:]
            all_theta = np.inner(Matrix_theta, x)
            all_v = np.inner(Matrix_v, x)
            all_s = np.inner(Matrix_s, x)

            # all_theta[1:] = all_theta[:-1] + all_v[:-1] * dt * np.tan(all_s[:-1]) / Lw

            return all_theta[1:] - (all_theta[:-1] + all_v[:-1] * dt * np.tan(all_s[:-1]) / Lw)

        def ineq_constrains(x):
            pass

        # get collision bounds

        # get maximum and min bounds
