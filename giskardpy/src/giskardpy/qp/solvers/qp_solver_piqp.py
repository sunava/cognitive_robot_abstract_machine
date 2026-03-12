from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import piqp
from line_profiler.explicit_profiler import profile

from giskardpy.qp.exceptions import InfeasibleException
from giskardpy.qp.qp_data import QPDataExplicit
from giskardpy.qp.solvers.qp_solver import QPSolver
from giskardpy.utils.math import fast_sparse_diag


@dataclass
class QPSolverPIQP(QPSolver[QPDataExplicit]):
    """
    min_x 0.5 x^T P x + c^T x
    s.t.  Ax = b
          Gx <= h
          lb <= x <= ub
    """

    @profile
    def solver_call_explicit_interface(self, qp_data: QPDataExplicit) -> np.ndarray:
        weight_matrix = fast_sparse_diag(qp_data.quadratic_weights)
        solver = piqp.SparseSolver()
        solver.settings.eps_abs = 1e-6
        solver.settings.eps_rel = 1e-7
        solver.settings.eps_duality_gap_abs = 1e-4
        solver.settings.eps_duality_gap_rel = 1e-5
        if len(qp_data.neq_upper_bounds) == 0:
            solver.setup(
                P=weight_matrix,
                c=qp_data.linear_weights,
                A=qp_data.eq_matrix,
                b=qp_data.eq_bounds,
                x_l=qp_data.box_lower_constraints,
                x_u=qp_data.box_upper_constraints,
            )
        else:
            solver.setup(
                P=weight_matrix,
                c=qp_data.linear_weights,
                A=qp_data.eq_matrix,
                b=qp_data.eq_bounds,
                G=qp_data.neq_matrix,
                h_l=qp_data.neq_lower_bounds,
                h_u=qp_data.neq_upper_bounds,
                x_l=qp_data.box_lower_constraints,
                x_u=qp_data.box_upper_constraints,
            )

        status = solver.solve()
        if status.value != piqp.PIQP_SOLVED:
            raise InfeasibleException(f"Solver status: {status.value}")
        return solver.result.x

    solver_call = solver_call_explicit_interface
