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
