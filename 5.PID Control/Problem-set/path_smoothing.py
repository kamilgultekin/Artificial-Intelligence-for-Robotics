import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

ALPHA = 0.2
BETA = 0.2

ORIGINAL = [[i, 0] for i in range(5)]
ORIGINAL += [[4, i] for i in range(1, 5)]
ORIGINAL += [[i, 4] for i in range(5, 10)]
ORIGINAL += [[9, i] for i in range(5, 10)]
ORIGINAL += [[i, 9] for i in range(10, 15)]
ORIGINAL += [[14, i] for i in range(10, 15)]
ORIGINAL += [[i, 14] for i in range(15, 20)]
ORIGINAL += [[19, i] for i in range(15, 20)]
ORIGINAL = np.array(ORIGINAL, dtype=np.float64)

# # Alternative implementation
# ORIGINAL = [[0, 0]]
# for i in range(0, 20, 3):
#     ORIGINAL += [[0, i]]
# for i in range(0, 20, 2):
#     ORIGINAL += [[i, 20]]
# for i in range(20, 0, -3):
#     ORIGINAL += [[20, i]]
# for i in range(20, 0, -2):
#     ORIGINAL += [[i, 0]]
# ORIGINAL = np.array(ORIGINAL, dtype=np.float64)

N = len(ORIGINAL)
print(ORIGINAL)
smoothed = ORIGINAL.copy()

fig, ax = plt.subplots()
orig, = ax.plot(ORIGINAL[:, 0], ORIGINAL[:, 1], 'b-')
smooth, = ax.plot(smoothed[:, 0], smoothed[:, 1], 'r--')

def init():
    ax.set_xlim(-1, 21)
    ax.set_ylim(-1, 21)
    ax.set_title('Path Smoothing Algorithm', fontsize=20)
    ax.set_ylabel('y [m]')
    ax.set_xlabel('x [m]')
    plt.legend(('Original', 'Smoothed'), loc='best')
    return orig, smooth,

def animate(i):
    print(i)
    i = i%(N - 2) + 1
    smoothed[i] += ALPHA*(ORIGINAL[i] - smoothed[i])
    smoothed[i] += BETA*(smoothed[i+1] + smoothed[i-1] - 2*smoothed[i])
    smooth.set_xdata(smoothed[:, 0])
    smooth.set_ydata(smoothed[:, 1])
    return smooth, 

anim = animation.FuncAnimation(fig, animate, 400, interval=100, init_func=init)
# plt.show()
anim.save("../../docs/images/path_smooth_anim.gif", writer="imagemagick")