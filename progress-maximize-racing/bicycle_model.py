from casadi import *
from tracks.readDataFcn import getTrack

# modified from race_car example

def bicycle_model(track="buggy_track.txt"):
    # define structs
    constraint = types.SimpleNamespace()
    model = types.SimpleNamespace()
    model_name = "Spatialbicycle_model"

    # load track parameters
    [s0, _, _, _, kapparef] = getTrack(track)
    # compute spline interpolations
    kapparef_s = interpolant("kapparef_s", "bspline", [s0], kapparef)

    ## Race car parameters
    m = 80 # mass in kg
    lr = 0.2 # COM to rear axle in m
    lf = 1.0 # COM to front axle in m
    L = lr + lf
    I = 25

    # tire params
    # fitted from
    # https://www.researchgate.net/publication/379670154_Measurement_of_the_lateral_characteristics_and_identification_of_the_Magic_Formula_parameters_of_city_and_cargo_bicycle_tyres
    D = 350
    C = 1.3
    B = 0.2

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
    alpha_f = atan(vy  / vx) - delta
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
    model.r_min = -6 # width of the track [m]
    model.r_max = 6 # width of the track [m]

    # state bounds
    model.delta_min = -0.2  # minimum steering angle [rad]
    model.delta_max = 0.2 # maximum steering angle [rad]
    model.vx_min = -15
    model.vx_max = 15

    model.e_psi_min = -0.1
    model.e_psi_max = 0.1

    # input bounds
    model.a_min = -5  # m/s^2
    model.a_max = 1  # m/s^2
    model.deltadot_min = -1.0  # minimum change rate of stering angle [rad/s]
    model.deltadot_max = 1.0  # maximum change rate of steering angle [rad/s]

    # Define initial conditions
    model.x0 = np.array([1, 0, 0, 1, 0, 0, 0])

    # define constraints struct
    constraint.expr = vertcat(r, e_psi, delta)

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
