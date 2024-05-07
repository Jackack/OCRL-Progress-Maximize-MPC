from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from bicycle_model import bicycle_model
import scipy.linalg
import numpy as np

# modified from race_car example

def acados_settings(Tf, N, track_file):
    ocp = AcadosOcp()

    # export model
    model, constraint = bicycle_model(track_file)

    # define acados ODE
    model_ac = AcadosModel()
    model_ac.f_impl_expr = model.f_impl_expr
    model_ac.f_expl_expr = model.f_expl_expr
    model_ac.x = model.x
    model_ac.xdot = model.xdot
    model_ac.u = model.u
    model_ac.z = model.z
    model_ac.p = model.p
    model_ac.name = model.name
    ocp.model = model_ac

    # define constraint
    model_ac.con_h_expr = constraint.expr

    # set dimensions
    nx = model.x.rows()
    nu = model.u.rows()
    ny = nx + nu
    ny_e = nx
    nsh = 3
    ocp.dims.N = N

    # set cost
    Q = np.diag([1e-3, 5e0, 1e-3, 1e-8, 1e-8, 1e-8, 1e-3])

    R = np.eye(nu)
    R[0, 0] = 1e-3
    R[1, 1] = 1e0

    Qe = np.diag([1e0, 5e3, 1e0, 1e-8, 1e-8, 1e-8, 1e-3])

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    unscale = N / Tf

    ocp.cost.W = unscale * scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Qe / unscale

    Vx = np.zeros((ny, nx))
    Vx[:nx, :nx] = np.eye(nx)
    ocp.cost.Vx = Vx

    Vu = np.zeros((ny, nu))
    Vu[nx, 0] = 1.0
    Vu[nx+1, 1] = 1.0
    ocp.cost.Vu = Vu

    Vx_e = np.zeros((ny_e, nx))
    Vx_e[:nx, :nx] = np.eye(nx)
    ocp.cost.Vx_e = Vx_e

    # set intial references
    ocp.cost.yref = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0])
    ocp.cost.yref_e = np.array([0, 0, 0, 0, 0, 0, 0])

    # setting constraints


    ocp.constraints.lbx = np.array([model.vx_min])
    ocp.constraints.ubx = np.array([model.vx_max])
    ocp.constraints.idxbx = np.array([3])


    ocp.constraints.lbu = np.array([model.a_min, model.deltadot_min])
    ocp.constraints.ubu = np.array([model.a_max, model.deltadot_max])
    ocp.constraints.idxbu = np.array([0, 1])

    # soft constraints
    ocp.cost.zl = 1e4 * np.ones((nsh,))
    ocp.cost.zu = 1e4 * np.ones((nsh,))
    ocp.cost.Zl = 1e4 * np.ones((nsh,))
    ocp.cost.Zu = 1e4 * np.ones((nsh,))

    ocp.constraints.lh = np.array(
        [
            model.r_min,
            model.e_psi_min,
            model.delta_min,
        ]
    )
    ocp.constraints.uh = np.array(
        [
            model.r_max,
            model.e_psi_max,
            model.delta_max,
        ]
    )
    ocp.constraints.lsh = np.zeros(nsh)
    ocp.constraints.ush = np.zeros(nsh)
    ocp.constraints.idxsh = np.array(range(nsh))

    # set intial condition
    ocp.constraints.x0 = model.x0
    ocp.solver_options.tf = Tf
    ocp.solver_options.qp_solver = "PARTIAL_CONDENSING_HPIPM"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.sim_method_num_stages = 4
    ocp.solver_options.sim_method_num_steps = 4

    # create solver
    acados_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")

    return constraint, model, acados_solver
