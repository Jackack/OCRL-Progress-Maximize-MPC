import sys
import time, os
import numpy as np
from acados_settings import *
from plotFcn import *
from tracks.readDataFcn import getTrack
import matplotlib.pyplot as plt


np.set_printoptions(threshold=sys.maxsize)
track = "buggy_track.txt"
[Sref, _, _, _, _] = getTrack(track)

Tf = 2.5 # prediction horizon
N = 20  # number of discretization steps
T = 200 # maximum simulation time[s]

# load model
constraint, model, acados_solver = acados_settings(Tf, N, track)

# dimensions
nx = model.x.rows()
nu = model.u.rows()
ny = nx + nu
Nsim = int(T * N / Tf)

# initialize data structs
simX = np.zeros((Nsim, nx))
simU = np.zeros((Nsim, nu))
x0 = model.x0
s0 = x0[0]
tcomp_sum = 0
tcomp_max = 0

# simulate
for i in range(Nsim):
    # update reference
    sref = s0 + Tf * model.vx_max
    for j in range(N):
        yref = np.array([sref, 0, 0, 0, 0, 0, 0, 0, 0])
        acados_solver.set(j, "yref", yref)
    yref_N = np.array([sref, 0, 0, 0, 0, 0, 0])
    acados_solver.set(N, "yref", yref_N)

    # solve ocp
    t = time.time()

    status = acados_solver.solve()


    elapsed = time.time() - t

    # manage timings
    tcomp_sum += elapsed
    if elapsed > tcomp_max:
        tcomp_max = elapsed

    # get solution
    x0 = acados_solver.get(0, "x")
    u0 = acados_solver.get(0, "u")
    for j in range(nx):
        simX[i, j] = x0[j]
    for j in range(nu):
        simU[i, j] = u0[j]

    # update initial condition
    x0 = acados_solver.get(1, "x")
    print(x0)
    acados_solver.set(0, "lbx", x0)
    acados_solver.set(0, "ubx", x0)
    s0 = x0[0]

    # check if one lap is done
    if x0[0] > Sref[-1] + 0.1:
        break

    if status != 0:
        print("acados returned status {} in closed loop iteration {}.".format(status, i))
        break

# Plot Results
t = np.linspace(0.0, Nsim * Tf / N, Nsim)
plotRes(simX, simU, t)
plotTrackProj(simX, track)

# Print some stats
print("Average computation time: {}".format(tcomp_sum / Nsim))
print("Maximum computation time: {}".format(tcomp_max))
print("Average speed:{}m/s".format(np.average(simX[:, 3])))
print("Lap time: {}s".format(Tf * Nsim / N))

if os.environ.get("ACADOS_ON_CI") is None:
    plt.show()
