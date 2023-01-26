from __future__ import print_function, division
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#Number of grid cells
m, n = 50, 50

#Probability of hitting the target
p = [[1/(m*n) for _ in range(n)] for _ in range(m)]
world = [[1 if random.random() > 0.2 else 0 for _ in range(n)] for _ in range(m)]
movements = [[0, 0],
             [0, 1],
             [1, 0],
             [-1, 0],
             [0, -1]]

sensor_right = 0.8
sensor_wrong = 1 - sensor_right
p_move = 0.7
p_stay = 1 - p_move

def sense(p, Z):
    '''It takes the probability distribution of the robot's position, and the measurement, and returns the
    probability distribution of the robot's position after the measurement
    
    Parameters
    ----------
    p
        the prior distribution
    Z
        the measurement
    
    Returns
    -------
        The probability of the robot being in a given cell, given the measurement.
    
    '''
    q = [[0 for _ in range(n)] for _ in range(m)]
    norm = 0
    for i in range(m):
        for j in range(n):
            hit = (world[i][j] == Z)
            q[i][j] = p[i][j]*(hit*sensor_right + (1 - hit)*sensor_wrong)
            norm += q[i][j]
    q = [[q[i][j]/norm for j in range(n)] for i in range(m)]
    return q

def move(p, U):
    '''It takes a probability distribution p and a motion vector U, and returns a new probability
    distribution q that is the result of moving p by U
    
    Parameters
    ----------
    p
        the probability distribution of the robot's location
    U
        the action, which is a vector of two elements, the first element is the number of steps to move up,
    the second element is the number of steps to move right.
    
    Returns
    -------
        The probability of the robot being in a given cell after it has moved.
    
    '''
    q = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            q[i][j] = p_move*p[(i-U[0])%m][(j-U[1])%n] + p_stay*p[i][j]
    return q

#Create the animation
fig, ax = plt.subplots()
grid = ax.imshow(p, cmap="GnBu")
line, = ax.plot([], [], 'r.')

i = m//2
j = n//2

def animate(_):
    '''It takes the current state of the robot, and updates it based on the current world and the robot's
    current position
    
    Parameters
    ----------
    _
        This is a dummy variable that is required by the FuncAnimation function.
    
    Returns
    -------
        grid, line,
    
    '''
    global p, i, j
    U = random.choice(movements)
    p = sense(p, world[i][j])
    p = move(p, U)
    i = (i + U[0])%m
    j = (j + U[1])%n
    print(p)
    grid.set_data(p)
    smallest = min(min(row) for row in p)
    largest = max(max(row) for row in p)
    grid.set_clim(vmin=smallest, vmax=largest)
    line.set_xdata(j)
    line.set_ydata(i)
    return grid, line,

def init(): 
    '''initialize animation'''
    ax.set_xlabel("Position X")
    ax.set_ylabel("Position Y")
    ax.set_title("Histogram Localization 2D")
    return grid, line,

anim = animation.FuncAnimation(fig, animate, 300, interval=50, init_func=init)
plt.show()
#anim.save("localization_2d.gif", writer="imagemagick")