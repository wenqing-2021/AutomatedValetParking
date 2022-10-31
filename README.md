# Hybrid A Star for Parking
## 1. Introduction
This repo provides an algorithm which uses hybrid a star for the initial path and the optimization based method to generate the trajectory. The pipeline of this algorithm is:

Hybrid A star -> Path optimization -> Cubic interpolation -> Velocity plan -> Solve optimization problem (use IPOPT)

---

### 1.1 File Structure
```
.
├── animation
│   ├── animation.py
│   └── __init__.py
├── collision_check
│   ├── collision_check.py
│   └── __init__.py
├── config
│   ├── config.yaml
│   ├── __init__.py
│   └── read_config.py
├── interpolation
│   ├── cubic_interpolation.py
│   └── __init__.py
├── main.py
├── map
│   ├── costmap.py
│   └── __init__.py
├── optimization
│   ├── __init__.py
│   ├── ipopt
│   ├── ocp_optimization.py
│   └── path_optimazition.py
├── path_planner
│   ├── compute_h.py
│   ├── hybrid_a_star.py
│   ├── __init__.py
│   ├── path_planner.py
│   └── rs_curve.py
├── util_math
│   ├── coordinate_transform.py
│   ├── __init__.py
│   └── spline.py
└── velocity_planner
    ├── __init__.py
    └── velocity_plan.py
```

### 1.2 Requirement
Python version >= 3.8
```
pip install scipy shapely pyomo cvxopt

conda install -c conda-forge ipopt
```

### 1.3 Data Structure
The Case1.csv is provided by https://www.tpcap.net/#/benchmarks, and the details of this file are presented by the following:

>The first six rows of the vector record the initial and goal poses of the to-be-parked vehicle. Suppose $V$ is the data vector.
> - $x_{0}$ = $V$[ 1 ], $y_{0}$ = $V$[ 2 ], $\theta_{0}$ = $V$[ 3 ]
> - $x_{f}$ = $V$[ 4 ], $y_f$ = $V$[ 5 ], $\theta_f$ = $V$[ 6 ]. 
> - $V$[ 7 ] records the total number of obstacles in the parking scenario. 
> - $V$[ 7+$i$ ] presents the number of vertexes in the $i$-th obstacle, where the index $i$ ranges from 1 to $V$[7]. 
> - After that, the vertexes of each obstacle are presented by their 2D coordinate values in the $x$ and $y$ axes. 
---
**Note**: you can build your own parking map based on the above rules and store the .csv file in the BenchmarkCase folder.

## 2. Usage
run the main.py to show the animation process

## 3. Todo List
 
- [ ] more spine function
- [ ] more velocity plan function
- [ ] store the trajectory data 

