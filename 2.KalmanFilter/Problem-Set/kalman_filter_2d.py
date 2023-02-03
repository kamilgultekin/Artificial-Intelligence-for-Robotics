from __future__ import print_function, division
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

dt = 0.1
I = np.identity(4)                  #Identity matrix
F = np.array([[1, 0, dt, 0],        #State transition matrix
              [0, 1, 0, dt],
              [0, 0, 1, 0],
              [0, 0, 0, 1]])
H = np.array([[1, 0, 0, 0],         #Measurement function
              [0, 1, 0, 0]])
R = np.array([[100, 0],            #Measurement noise
              [0, 100]])
x = np.array([[50],                 #State estimate
              [50],
              [0],
              [0]])
P = np.array([[1000, 0, 0, 0],      #Uncertainty covariance
              [0, 1000, 0, 0],
              [0, 0, 1000, 0],
              [0, 0, 0, 1000]])
u = np.array([[0],                  #Control
              [0],
              [0],
              [0]])

def predict(x, u, P):
    x = F.dot(x) + u
    P = F.dot(P).dot(F.T)
    return x, P

def update(x, z, P):
    y = z.T - H.dot(x)
    S = H.dot(P).dot(H.T) + R
    K = P.dot(H.T).dot(np.linalg.inv(S))
    x = x + K.dot(y)
    P = (I - K.dot(H)).dot(P)
    return x, P

def gaussian_2d(X, Y, x, P):
    return np.exp(-((X - x[0, 0])**2/(2*P[0, 0]) + (Y - x[1, 0])**2/(2*P[1, 1])))

#plt.ion()
fig, ax = plt.subplots()
X, Y = np.meshgrid([1.0*i for i in range(100)], [1.0*i for i in range(100)])
possible_motions = [[1, 0, 0, 0],
                    [0, 1, 0, 0],
                    [-1, 0, 0, 0],
                    [0, -1, 0, 0],
                    [1, 1, 0, 0],
                    [1, -1, 0, 0],
                    [-1, 1, 0, 0],
                    [-1, -1, 0, 0]]
motions = np.array([random.choice(possible_motions) for _ in range(500)])
measurements = np.array([[[50, 50]] for _ in range(500)])
for i in range(1, 500):
    measurements[i, :] = measurements[i-1, :] + motions[i, :2]

f = gaussian_2d(X, Y, x, P)
grid = ax.imshow(f, cmap="YlGnBu")
dot, = ax.plot(0, 0, 'r*',label='Robot Position')
def animate(i):
    global x, P
    x, P = predict(x, motions[i].reshape(4, 1), P)
    x, P = update(x, measurements[i].reshape(1, 2), P)
    f = gaussian_2d(X, Y, x, P)
    grid.set_data(f)
    # smallest = min(min(row) for row in f)
    # largest = max(max(row) for row in f)
    grid.set_clim(vmin=0, vmax=1)
    dot.set_data(measurements[i, 0, :2])
    return grid, dot,

def init():
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("Kalman Filter - 2D Grid based Environment")
    ax.legend(loc='best')
    ax.text(59, 83, 'Grid with Prediction', bbox={'facecolor': 'white', 'pad': 5})
    return grid, dot,

anim = animation.FuncAnimation(fig, animate, 200, interval=50, init_func=init)
plt.show()
# anim.save("../../docs/images/kalman-localization_2D.gif", writer="imagemagick")