# Yuansj
# Last update: 06/13
import numpy as np
import matplotlib.pyplot as plt

from map.costmap import Vehicle, _map

def plot_obstacles(map):
    plt.ion()
    plt.figure(1)
    # create original map
    ## create obstacles
    for j in range(0, map.case.obs_num):
        plt.fill(map.case.obs[j][:, 0], map.case.obs[j][:, 1], facecolor = 'k', alpha = 0.5)
    
    ## create start vahicle and terminate vehicle
    temp = map.case.vehicle.create_polygon(map.case.x0, map.case.y0, map.case.theta0)
    plt.plot(temp[:, 0], temp[:, 1], linestyle='--', linewidth = 0.4, color = 'green')
    temp = map.case.vehicle.create_polygon(map.case.xf, map.case.yf, map.case.thetaf)
    plt.plot(temp[:, 0], temp[:, 1], linestyle='--', linewidth = 0.4, color = 'red')

    ## create arrow
    plt.arrow(map.case.x0, map.case.y0, np.cos(map.case.theta0), np.sin(map.case.theta0), width=0.2, color = "gold")
    plt.arrow(map.case.xf, map.case.yf, np.cos(map.case.thetaf), np.sin(map.case.thetaf), width=0.2, color = "gold")

    plt.title("Hybrid A Start Path")
    plt.xlim(map.boundary[0], map.boundary[1])
    plt.ylim(map.boundary[2], map.boundary[3])
    plt.gca().set_aspect('equal', adjustable = 'box')
    plt.gca().set_axisbelow(True)
    plt.draw()

def plot_node(nodes, current_node):
    # plt.ion()
    plt.figure(1)

    plt.plot(current_node.x,current_node.y,'o',color='r')

    # create path
    for i in range(len(nodes)):
        plt.plot(nodes[i][0], nodes[i][1], 'o', color='grey')
    
    plt.draw()

def plot_path(x,y,color='grey'):
    plt.figure(1)
    plt.plot(x,y,'*-',linewidth=1,color=color)
    plt.draw()

def plot_final_path(path, map:_map, color='green', show_car=False):
    # plot cost map
    plot_obstacles(map)
    x,y=[],[]
    v = Vehicle()
    for i in range(len(path)):
        x.append(path[i][0])
        y.append(path[i][1])
        plot_path(x,y,color)
        if show_car:
            points = v.create_polygon(path[i][0], path[i][1], path[i][2])
            plt.plot(points[:, 0], points[:, 1], linestyle='--', linewidth = 0.4, color = color)
        plt.draw()
        plt.pause(0.1)
    
    plt.show()

def plot_collision_p(x,y,theta,map):
    plt.clf()
    v = Vehicle()
    plot_obstacles(map)
    plt.title('Collision Position')
    temp = v.create_polygon(x, y, theta)
    plt.plot(temp[:, 0], temp[:, 1], linestyle='--', linewidth = 0.4, color = 'blue')
    # compute circle diameter
    Rd = 0.5 * np.sqrt(((v.lr+v.lw+v.lf)/2)**2 + (v.lb**2))
    # compute circle center position
    front_circle = (x+1/4*(3*v.lw+3*v.lf-v.lr)*np.cos(theta), 
                    y+1/4*(3*v.lw+3*v.lf-v.lr)*np.sin(theta))
    rear_circle = (x+1/4*(v.lw+v.lf-3*v.lr)*np.cos(theta), 
                   y+1/4*(v.lw+v.lf-3*v.lr)*np.sin(theta))
    figure, axes = plt.figure(1), plt.gca()
    c1 = plt.Circle(front_circle, Rd, fill=False)
    c2 = plt.Circle(rear_circle, Rd, fill=False)
    axes.add_artist(c1)
    axes.add_artist(c2)

    plt.draw()

    