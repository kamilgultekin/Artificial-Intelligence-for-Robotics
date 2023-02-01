from math import sqrt, pi, cos, sin, tan, exp, atan2
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


L = 15          #Car length
W = 0.5           #Car width
R = 0.8           #Tire radius

class Robot(object):
    def __init__(self, length=20.0):
        """
        Creates robot and initializes location/orientation to 0, 0, 0.
        """
        self.x = 0.0
        self.y = 0.0
        self.orientation = 0.0
        self.length = length
        self.steering_noise = 0.0
        self.distance_noise = 0.0
        self.steering_drift = 0.0
        self.steering = 0
        self.L = L
        self.W = W
        self.left_side = np.array([[-L/2, L/2],
                                   [W/2, W/2],
                                   [1, 1]])
        self.right_side = np.array([[-L/2, L/2],
                                   [-W/2, -W/2],
                                   [1, 1]])
        self.front_side = np.array([[L/2, L/2],
                                   [W/2, -W/2],
                                   [1, 1]])
        self.back_side = np.array([[-L/2, -L/2],
                                   [W/2, -W/2],
                                   [1, 1]])
        self.wheel = np.array([[-R, R],
                               [0, 0],
                               [1, 1]])

    def set(self, x, y, orientation,new_steering):
        """
        Sets a robot coordinate.
        """
        self.x = x
        self.y = y
        self.orientation = orientation % (2.0 * np.pi)
        self.steering = new_steering

    def set_noise(self, steering_noise, distance_noise):
        """
        Sets the noise parameters.
        """
        # makes it possible to change the noise parameters
        # this is often useful in particle filters
        self.steering_noise = steering_noise
        self.distance_noise = distance_noise

    def set_steering_drift(self, drift):
        """
        Sets the systematical steering drift parameter
        """
        self.steering_drift = drift

    def move(self, steering, distance, tolerance=0.001, max_steering_angle=np.pi / 4.0):
        """
        steering = front wheel steering angle, limited by max_steering_angle
        distance = total distance driven, most be non-negative
        """
        if steering > max_steering_angle:
            steering = max_steering_angle
        if steering < -max_steering_angle:
            steering = -max_steering_angle
        if distance < 0.0:
            distance = 0.0

        # apply noise
        steering2 = random.gauss(steering, self.steering_noise)
        distance2 = random.gauss(distance, self.distance_noise)

        # apply steering drift
        steering2 += self.steering_drift

        # Execute motion
        turn = np.tan(steering2) * distance2 / self.length

        if abs(turn) < tolerance:
            # approximate by straight line motion
            self.x += distance2 * np.cos(self.orientation)
            self.y += distance2 * np.sin(self.orientation)
            self.orientation = (self.orientation + turn) % (2.0 * np.pi)
        else:
            # approximate bicycle model for motion
            radius = distance2 / turn
            cx = self.x - (np.sin(self.orientation) * radius)
            cy = self.y + (np.cos(self.orientation) * radius)
            self.orientation = (self.orientation + turn) % (2.0 * np.pi)
            self.x = cx + (np.sin(self.orientation) * radius)
            self.y = cy - (np.cos(self.orientation) * radius)
        
    def transform(self, x, y, orientation):
        return np.array(
            [[cos(orientation), -sin(orientation), x],
             [sin(orientation), cos(orientation), y],
             [0, 0, 1]])

    def draw(self, lines):
        T = self.transform(self.x, self.y, self.orientation)
        left_side = T.dot(self.left_side)
        right_side = T.dot(self.right_side)
        front_side = T.dot(self.front_side)
        back_side = T.dot(self.back_side)
        d = sqrt((L/2 - R)**2 + (W/2)**2)
        orientation = atan2(W/2, -L/2 + R)
        x = self.x + d*cos(orientation + self.orientation)
        y = self.y + d*sin(orientation + self.orientation)
        T = self.transform(x, y, self.orientation)
        back_left_tire = T.dot(self.wheel)
        orientation = atan2(-W/2, -L/2 + R)
        x = self.x + d*cos(orientation + self.orientation)
        y = self.y + d*sin(orientation + self.orientation)
        T = self.transform(x, y, self.orientation)
        back_right_tire = T.dot(self.wheel)
        orientation = atan2(W/2, L/2 - R)
        x = self.x + d*cos(orientation + self.orientation)
        y = self.y + d*sin(orientation + self.orientation)
        T = self.transform(x, y, self.orientation + self.steering)
        front_left_tire = T.dot(self.wheel)
        orientation = atan2(-W/2, L/2 - R)
        x = self.x + d*cos(orientation + self.orientation)
        y = self.y + d*sin(orientation + self.orientation)
        T = self.transform(x, y, self.orientation + self.steering)
        front_right_tire = T.dot(self.wheel)
        lines[0][0].set_data(left_side[0, :], left_side[1, :])
        lines[1][0].set_data(right_side[0, :], right_side[1, :])
        lines[2][0].set_data(front_side[0, :], front_side[1, :])
        lines[3][0].set_data(back_side[0, :], back_side[1, :])
        lines[4][0].set_data(back_left_tire[0, :], back_left_tire[1, :])
        lines[5][0].set_data(back_right_tire[0, :], back_right_tire[1, :])
        lines[6][0].set_data(front_left_tire[0, :], front_left_tire[1, :])
        lines[7][0].set_data(front_right_tire[0, :], front_right_tire[1, :])
        return lines

    def __repr__(self):
        return '[x=%.5f y=%.5f orient=%.5f]' % (self.x, self.y, self.orientation)

WORLD_SIZE = 100
fig, ax = plt.subplots()
lines = [ax.plot([0, 0], [0, 0], 'k-') for _ in range(8)]
lines[4][0].set_linewidth(5)
lines[5][0].set_linewidth(5)
lines[6][0].set_linewidth(5)
lines[7][0].set_linewidth(5)

traj, = ax.plot([], [], 'b-',label='Trajectory')
des, =  ax.plot([],[],'r*', label='Desired Trajectory')

robot = Robot()
robot.set(0, 1, 0,0)
robot.set_steering_drift(10.0/180.0*3.14) # add steering wheel
tau_p = 0.2
tau_d = 3.0
tau_i = 0.004
n = 200
speed = 1.0
x_trajectory = []
y_trajectory = []
prev_cte = robot.y
int_cte = 0

def run(_):
    global robot, tau_p, tau_d, tau_i, n, speed, x_trajectory, y_trajectory, prev_cte, int_cte, lines 
    cte = robot.y
    diff_cte = (cte - prev_cte) / speed
    prev_cte = cte
    int_cte += cte
    steer = -tau_p * cte - tau_d * diff_cte - tau_i * int_cte
    robot.move(steer, speed)
    x_trajectory.append(robot.x)
    y_trajectory.append(robot.y)
    n = len(x_trajectory)
    lines = robot.draw(lines)
    print(x_trajectory)
    traj.set_data(x_trajectory, y_trajectory)
    des.set_data(x_trajectory, np.zeros(n))
    return lines, traj, des,

def init():
    ax.set_xlim(0, WORLD_SIZE)
    ax.set_ylim(-WORLD_SIZE/50, WORLD_SIZE/50)
    ax.set_xlabel('x axis [m]')
    ax.set_ylabel('y axis [m]')
    ax.set_title('PID Controller Demo')
    ax.legend()
    return lines,

anim = animation.FuncAnimation(fig, run, 200, interval=100,init_func=init)
plt.show()