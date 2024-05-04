#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

# author: Daniel Kloeser

from casadi import *
from tracks.readDataFcn import getTrack


def bicycle_model(track="LMS_Track.txt"):
    # define structs
    constraint = types.SimpleNamespace()
    model = types.SimpleNamespace()

    model_name = "Spatialbicycle_model"

    # load track parameters
    [s0, _, _, _, kapparef] = getTrack(track)
    length = len(s0)
    pathlength = s0[-1]
    # copy loop to beginning and end
    s0 = np.append(s0, [s0[length - 1] + s0[1:length]])
    kapparef = np.append(kapparef, kapparef[1:length])
    s0 = np.append([-s0[length - 2] + s0[length - 81 : length - 2]], s0)
    kapparef = np.append(kapparef[length - 80 : length - 1], kapparef)

    # compute spline interpolations
    kapparef_s = interpolant("kapparef_s", "bspline", [s0], kapparef)

    ## Race car parameters
    # m = 60 # mass in kg
    # lr = 0.2 # COM to rear axle in m
    # lf = 1.2 # COM to front axle in m
    m = 1
    lr = 0.05
    lf = 0.15
    L = lr + lf
    c_MOI = 6 # MOI constant, educated guess
    I = m * (L ** 2) / c_MOI # moment of inertia

    # tire params
    D = 10 # peak lateral tire force in N, determined via experiment
    C = 1.5 # shape factor, typical value
    B = 0.5


    ## CasADi Model
    # state variables
    s = MX.sym("s")
    r = MX.sym("r")
    e_psi = MX.sym("e_psi")
    vx = MX.sym("vx")
    vy = MX.sym("vy")
    omega = MX.sym("omega")
    delta = MX.sym("delta")
    x = vertcat(s, r, e_psi, vx, vy, omega, delta)

    # control variables
    deltadot_u = MX.sym("deltadot_u")
    a = MX.sym("a")
    u = vertcat(a, deltadot_u)

    # state derivative w.r.t time
    sdot = MX.sym("sdot")
    rdot = MX.sym("rdot")
    e_psidot = MX.sym("e_psidot")
    vxdot = MX.sym("vxdot")
    vydot = MX.sym("vydot")
    omegadot = MX.sym("omegadot")
    deltadot = MX.sym("deltadot")
    xdot = vertcat(sdot, rdot, e_psidot, vxdot, vydot, omegadot, deltadot)

    # algebraic variables
    z = vertcat([])

    # parameters
    p = vertcat([])

    # dynamics
    sdota = (vx * cos(e_psi) - vy * sin(e_psi)) / (1 - r * kapparef_s(s))
    Fx = m * a

    # Pacejka tire model

    # tire slip angles
    alpha_f = atan(vy / vx) - delta
    alpha_r = atan(vy / vx)

    # lateral tire forces
    Ffy = -2 * D * sin(C * atan(B * alpha_f))
    Fry = -2 * D * sin(C * atan(B * alpha_r))

    # construct dynamics
    f_expl = vertcat(
        sdota,
        vx * sin(e_psi) - vy * cos(e_psi),
        omega - kapparef_s(s) * sdota,
        (Fx - Ffy * sin(delta)) / m + vy * omega,
        (Fry - Ffy * cos(delta)) / m - vx * omega,
        (Ffy * lf * cos(delta) - Fry * lr) / I,
        deltadot_u,
    )

    # Model bounds
    model.r_min = -0.12 # width of the track [m]
    model.r_max = 0.12  # width of the track [m]

    # state bounds
    model.delta_min = -0.4  # minimum steering angle [rad]
    model.delta_max = 0.4 # maximum steering angle [rad]
    model.vx_min = 0
    model.vx_max = 3

    # input bounds
    model.a_min = -2  # m/s^2
    model.a_max = 2  # m/s^2
    model.deltadot_min = -2.0  # minimum change rate of stering angle [rad/s]
    model.deltadot_max = 2.0  # maximum change rate of steering angle [rad/s]

    # Define initial conditions
    model.x0 = np.array([-2, 0, 0, 2, 0, 0, 0])

    # define constraints struct
    constraint.pathlength = pathlength
    constraint.expr = vertcat(r, delta, vx)

    # Define model struct
    params = types.SimpleNamespace()

    params.m = m # mass in kg
    params.lr = lr # COM to rear axle in m
    params.lf = lf # COM to front axle in m
    params.L = L
    params.I = I # moment of inertia

    # tire params
    params.D = D # peak lateral tire force in N, determined via experiment
    params.C = C # shape factor, typical value
    params.B = B

    model.f_impl_expr = xdot - f_expl
    model.f_expl_expr = f_expl
    model.x = x
    model.xdot = xdot
    model.u = u
    model.z = z
    model.p = p
    model.name = model_name
    model.params = params
    return model, constraint
