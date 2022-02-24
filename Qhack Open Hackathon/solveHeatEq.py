# Solving heat equation by Finite Difference Method (FDM) and produce a gif image.
# Source https://www.raucci.net/2021/10/07/solving-2d-heat-equation/ and slightly changed.

from pennylane import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation

# definitions, init and boundaries

plate_length = 32
time_steps = 1000
alpha = 2
delta_x = 1
u_init = 20.0
u_t = 100.0
u_l = 0.0
u_b = 100.0
u_r = 0.0

delta_t = (delta_x ** 2) / (4 * alpha)
gamma = (alpha * delta_t) / (delta_x ** 2)

u = np.zeros((time_steps, plate_length, plate_length))
# Set the initial condition
u.fill(u_init)

# Set the boundary conditions
u[:, (plate_length - 1):, :] = u_t
u[:, :, :1] = u_l
u[:, :1, 1:] = u_b
u[:, :, (plate_length - 1):] = u_r

# compute heat over time
# point_next = point_act + gamma * ( point_top + point_bottom + point_left + point_right - 4 * point_act)
for k in range(0, time_steps - 1, 1):
    for i in range(1, plate_length - 1, delta_x):
        for j in range(1, plate_length - 1, delta_x):
            u[k + 1, i, j] = gamma * (
                u[k][i + 1][j] + u[k][i - 1][j] + u[k][i][j + 1] + u[k][i][j - 1] - 4 * u[k][i][j]) + u[k][i][j]

# plot
def plotheatmap(u_k, k):
    plt.clf()

    plt.title(f"Temperature at t = {k * delta_t:.3f} unit time")
    plt.xlabel("x")
    plt.ylabel("y")

    plt.pcolormesh(u_k, cmap=plt.cm.jet, vmin=0, vmax=100)
    plt.colorbar()

    return plt

def animate(k):
    plotheatmap(u[k], k)

anim = animation.FuncAnimation(plt.figure(), animate, interval=1, frames=time_steps,  repeat=False)
anim.save("heat_equation_solution.gif")