/*
 *    This file is part of acados.
 *
 *    acados is free software; you can redistribute it and/or
 *    modify it under the terms of the GNU Lesser General Public
 *    License as published by the Free Software Foundation; either
 *    version 3 of the License, or (at your option) any later version.
 *
 *    acados is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 *    Lesser General Public License for more details.
 *
 *    You should have received a copy of the GNU Lesser General Public
 *    License along with acados; if not, write to the Free Software Foundation,
 *    Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */

#include "acados/ocp_nlp/ocp_nlp_sqp.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "acados/ocp_nlp/ocp_nlp_common.h"
#include "acados/ocp_qp/ocp_qp_common.h"
#include "acados/utils/print.h"
#include "acados/utils/timing.h"
#include "acados/utils/types.h"

void prepare_qp(const ocp_nlp_in *nlp_in, ocp_nlp_sqp_memory *sqp_mem) {

    const int_t N = nlp_in->N;
    const int_t *nx = nlp_in->nx;
    const int_t *nu = nlp_in->nu;
    const int_t *nb = nlp_in->nb;
    const int_t **idxb = nlp_in->idxb;
    const int_t *ng = nlp_in->ng;

    real_t **qp_A = (real_t **) sqp_mem->qp_solver->qp_in->A;
    real_t **qp_B = (real_t **) sqp_mem->qp_solver->qp_in->B;
    real_t **qp_b = (real_t **) sqp_mem->qp_solver->qp_in->b;
    real_t **qp_q = (real_t **) sqp_mem->qp_solver->qp_in->q;
    real_t **qp_r = (real_t **) sqp_mem->qp_solver->qp_in->r;
    real_t **qp_R = (real_t **) sqp_mem->qp_solver->qp_in->R;
    real_t **qp_Q = (real_t **) sqp_mem->qp_solver->qp_in->Q;
    real_t **qp_S = (real_t **) sqp_mem->qp_solver->qp_in->S;
    real_t **qp_lb = (real_t **) sqp_mem->qp_solver->qp_in->lb;
    real_t **qp_ub = (real_t **) sqp_mem->qp_solver->qp_in->ub;
    real_t **qp_Cx = (real_t **) sqp_mem->qp_solver->qp_in->Cx;
    real_t **qp_Cu = (real_t **) sqp_mem->qp_solver->qp_in->Cu;
    real_t **qp_lc = (real_t **) sqp_mem->qp_solver->qp_in->lc;
    real_t **qp_uc = (real_t **) sqp_mem->qp_solver->qp_in->uc;

    real_t **hess_l = (real_t **) sqp_mem->common->hess_l;
    real_t **grad_f = (real_t **) sqp_mem->common->grad_f;
    real_t **jac_h = (real_t **) sqp_mem->common->jac_h;
    real_t **jac_g = (real_t **) sqp_mem->common->jac_g;
    real_t **h = (real_t **) sqp_mem->common->h;
    real_t **g = (real_t **) sqp_mem->common->g;

    real_t **nlp_x = (real_t **) sqp_mem->common->x;
    real_t **nlp_u = (real_t **) sqp_mem->common->u;
    real_t **nlp_lg = (real_t **) nlp_in->lg;
    real_t **nlp_ug = (real_t **) nlp_in->ug;

    // Objective
    for (int_t i = 0; i <= N; i++) {
        for (int_t j = 0; j < nx[i]; j++) {
            for (int_t k = 0; k < nx[i]; k++) {
                qp_Q[i][j * nx[i] + k] = hess_l[i][j * (nx[i] + nu[i]) + k];
            }
            for (int_t k = 0; k < nu[i]; k++) {
                qp_S[i][j * nx[i] + k] = hess_l[i][j * (nx[i] + nu[i]) + nx[i] + k];
            }
        }
        for (int_t j = 0; j < nu[i]; j++) {
            for (int_t k = 0; k < nu[i]; k++) {
                qp_R[i][j * nu[i] + k] = hess_l[i][(nx[i] + j) * (nx[i] + nu[i]) + nx[i] + k];
            }
        }
        for (int_t j = 0; j < nx[i]; j++) qp_q[i][j] = grad_f[i][j];
        for (int_t j = 0; j < nu[i]; j++) qp_r[i][j] = grad_f[i][nx[i] + j];
    }

    // State-continuity constraints, and state/control bounds
    for (int_t i = 0; i < N; i++) {
        for (int_t j = 0; j < nx[i]; j++) {
            qp_b[i][j] = h[i][j] -  nlp_x[i][j];
            for (int_t k = 0; k < nx[i]; k++) {
                qp_A[i][k * nx[i] + j] = jac_h[i][k * nx[i] + j];
            }
            for (int_t k = 0; k < nu[i]; k++) {
                qp_B[i][k * nx[i] + j] = jac_h[i][(nx[i] + k) * nx[i] + j];
            }
        }
        for (int_t j = 0; j < nb[i]; j++) {
// #ifdef FLIP_BOUNDS
            // TODO(nielsvd): I believe #ifdef-section is unnecessary when 
            //                avoiding use of w-vector. VERIFY!
// #else
            if (idxb[i][j] < nu[i]) {
                qp_lb[i][j] = nlp_in->lb[i][j] - nlp_u[i][idxb[i][j]];
                qp_ub[i][j] = nlp_in->ub[i][j] - nlp_u[i][idxb[i][j]];
            } else {
                qp_lb[i][j] = nlp_in->lb[i][j] - nlp_u[i][idxb[i][j]-nu[i]];
                qp_ub[i][j] = nlp_in->ub[i][j] - nlp_u[i][idxb[i][j]-nu[i]];
            }
// #endif
        }
    }

    // Path constraints
    for (int_t i = 0; i <= N; i++) {
        for (int_t j = 0; i < ng[i]; j++) {
            qp_lc[i][j] = nlp_lg[i][j] - g[i][j];
            qp_uc[i][j] = nlp_ug[i][j] - g[i][j];
            for (int_t k = 0; i < nx[i]; k++) {
                qp_Cx[i][k * nx[i] + j] = jac_g[i][k * nx[i] + j];
            }
            for (int_t k = 0; i < nu[i]; k++) {
                qp_Cu[i][k * nx[i] + j] = jac_g[i][(nx[i] + k) * nx[i] + j];
            }
        }
    }
}

void update_variables(const ocp_nlp_in *nlp_in, ocp_nlp_sqp_memory *sqp_mem) {
    const int_t N = nlp_in->N;
    const int_t *nx = nlp_in->nx;
    const int_t *nu = nlp_in->nu;
    const int_t *nb = nlp_in->nb;
    const int_t *ng = nlp_in->ng;
    const int_t **idxb = nlp_in->idxb;

    real_t **nlp_x = (real_t **) sqp_mem->common->x;
    real_t **nlp_u = (real_t **) sqp_mem->common->u;
    real_t **nlp_pi = (real_t **) sqp_mem->common->pi;
    real_t **nlp_lam = (real_t **) sqp_mem->common->lam;

    for (int_t i = 0; i <= N; i++) {
        for (int_t j = 0; j < nx[i]; j++) {
            nlp_x[i][j] += sqp_mem->qp_solver->qp_out->x[i][j];
        }
        for (int_t j = 0; j < nu[i]; j++) {
            nlp_u[i][j] += sqp_mem->qp_solver->qp_out->u[i][j];
        }
        for (int_t j = 0; j < nb[i]; j++) {
            nlp_pi[i][idxb[i][j]] = sqp_mem->qp_solver->qp_out->pi[i][idxb[i][j]];
        }
        for (int_t j = 0; j < ng[i]; j++) {
            nlp_lam[i][j] = sqp_mem->qp_solver->qp_out->lam[i][j];
        }
    }
}

void store_variables(const ocp_nlp_in *nlp_in, ocp_nlp_out *nlp_out, ocp_nlp_sqp_memory *sqp_mem) {
    const int_t N = nlp_in->N;
    const int_t *nx = nlp_in->nx;
    const int_t *nu = nlp_in->nu;
    const int_t *nb = nlp_in->nb;
    const int_t *ng = nlp_in->ng;
    const int_t **idxb = nlp_in->idxb;

    for (int_t i = 0; i <= N; i++) {
        for (int_t j = 0; j < nx[i]; j++) {
            nlp_out->x[i][j] = sqp_mem->common->x[i][j];
        }
        for (int_t j = 0; j < nu[i]; j++) {
            nlp_out->u[i][j] = sqp_mem->common->u[i][j];
        }
        for (int_t j = 0; j < nb[i]; j++) {
            nlp_out->pi[i][idxb[i][j]] = sqp_mem->common->pi[i][idxb[i][j]];
        }
        for (int_t j = 0; j < ng[i]; j++) {
            nlp_out->lam[i][j] = sqp_mem->common->lam[i][j];
        }
    }
}

// Simple fixed-step Gauss-Newton based SQP routine
int_t ocp_nlp_sqp(const ocp_nlp_in *nlp_in, ocp_nlp_out *nlp_out,
                  void *nlp_args_, void *nlp_mem_, void *nlp_work_) {
    int_t return_status = 0;

    // Initialize
    ocp_nlp_sqp_memory *sqp_mem = (ocp_nlp_sqp_memory *)nlp_mem_;

    // SQP iterations
    ocp_nlp_sqp_args * nlp_args = (ocp_nlp_sqp_args *) nlp_args_;
    int_t max_sqp_iterations = nlp_args->maxIter;
    
    for (int_t sqp_iter = 0; sqp_iter < max_sqp_iterations; sqp_iter++) {
        // Compute/update quadratic approximation
        sqp_mem->sensitivity_method->fun(nlp_in, sqp_mem->common,
            sqp_mem->sensitivity_method->args,
            sqp_mem->sensitivity_method->mem,
            sqp_mem->sensitivity_method->work);

        // Prepare QP
        prepare_qp(nlp_in, sqp_mem);

        // Solve QP
        int_t qp_status = sqp_mem->qp_solver->fun(
            sqp_mem->qp_solver->qp_in, sqp_mem->qp_solver->qp_out,
            sqp_mem->qp_solver->args, sqp_mem->qp_solver->mem,
            sqp_mem->qp_solver->work);
        if (qp_status) return_status = qp_status;

        // Update optimization variables (globalization)
        update_variables(nlp_in, sqp_mem);
    }

    // Post-process solution
    store_variables(nlp_in, nlp_out, sqp_mem);

    return return_status;
}

void ocp_nlp_sqp_create_memory(const ocp_nlp_in *in, void *args_,
                               void *memory_) {

}

void ocp_nlp_sqp_free_memory(void *mem_) {

}
