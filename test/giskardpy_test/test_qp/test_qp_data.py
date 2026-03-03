import numpy as np
import pytest
import scipy.sparse as sp
from giskardpy.qp.qp_data import QPData, Conditioning
from giskardpy.qp.solvers.qp_solver_qpSWIFT import QPSolverQPSwift


@pytest.fixture(scope="module")
def simple_inequality_qp() -> tuple[QPData, np.ndarray]:
    """
    min_{x1, x2} x1^2 + x2^2
    s.t.
        -inf <= x1 <= inf
        -inf <= x2 <= inf
        -inf <= x1 + x2 <= -0.5

    expected solution:
        [-0.25, -0.25]
    """
    return QPData(
        quadratic_weights=np.array([1.0, 1.0]),
        linear_weights=np.array([0.0, 0.0]),
        box_lower_constraints=np.array([-np.inf, -np.inf]),
        box_upper_constraints=np.array([np.inf, np.inf]),
        eq_matrix=sp.csc_matrix(np.zeros((0, 2))),
        eq_bounds=np.array([]),
        neq_matrix=sp.csc_matrix(np.array([[1.0, 1.0]])),
        neq_lower_bounds=np.array([-np.inf]),
        neq_upper_bounds=np.array([-0.5]),
    ), np.array([-0.25, -0.25])


@pytest.fixture(scope="module")
def simple_eq_as_inequality_qp() -> tuple[QPData, np.ndarray]:
    """
    min_{x1, x2} x1^2 + x2^2
    s.t.
        -inf <= x1 <= inf
        -inf <= x2 <= inf
        -inf <= x1 + x2 <= -0.5

    expected solution:
        [-0.25, -0.25]
    """
    return QPData(
        quadratic_weights=np.array([1.0, 1.0]),
        linear_weights=np.array([0.0, 0.0]),
        box_lower_constraints=np.array([-np.inf, -np.inf]),
        box_upper_constraints=np.array([np.inf, np.inf]),
        eq_matrix=sp.csc_matrix(np.zeros((0, 2))),
        eq_bounds=np.array([]),
        neq_matrix=sp.csc_matrix(np.array([[1.0, 1.0]])),
        neq_lower_bounds=np.array([-0.5]),
        neq_upper_bounds=np.array([-0.5]),
    ), np.array([-0.25, -0.25])


@pytest.fixture(scope="module")
def simple_equality_qp() -> tuple[QPData, np.ndarray]:
    """
    min_{x1, x2} x1^2 + x2^2
    s.t.
        -inf <= x1 <= inf
        -inf <= x2 <= inf
        x1 + x2 = -0.5

    expected solution:
        [-0.25, -0.25]
    """
    return QPData(
        quadratic_weights=np.array([1.0, 1.0]),
        linear_weights=np.array([0.0, 0.0]),
        box_lower_constraints=np.array([-np.inf, -np.inf]),
        box_upper_constraints=np.array([np.inf, np.inf]),
        eq_matrix=sp.csc_matrix(np.array([[1.0, 1.0]])),
        eq_bounds=np.array([-0.5]),
        neq_matrix=sp.csc_matrix(np.zeros((0, 2))),
        neq_lower_bounds=np.array([]),
        neq_upper_bounds=np.array([]),
    ), np.array([-0.25, -0.25])


def test_qp_data_inequality(simple_inequality_qp):
    qp_data, expected = simple_inequality_qp
    result = QPSolverQPSwift().solver_call(qp_data)
    assert np.allclose(result, expected)


def test_qp_data_simple_eq_as_inequality_qp(simple_eq_as_inequality_qp):
    qp_data, expected = simple_eq_as_inequality_qp
    result = QPSolverQPSwift().solver_call(qp_data)
    assert np.allclose(result, expected)


def test_qp_data_equality(simple_equality_qp):
    qp_data, expected = simple_equality_qp
    result = QPSolverQPSwift().solver_call(qp_data)
    assert np.allclose(result, expected)


def test_C_conditioning(simple_inequality_qp):
    qp_data, expected = simple_inequality_qp
    conditioning = Conditioning(C=sp.diags([69.0, 23.0]))
    conditioned_qp_data = conditioning.apply(qp_data)
    conditioned_result = QPSolverQPSwift().solver_call(conditioned_qp_data)
    result = conditioning.unapply(conditioned_result)
    assert np.allclose(result, expected)


def test_R_conditioning(simple_equality_qp):
    qp_data, expected = simple_equality_qp
    conditioning = Conditioning(R_eq=sp.diags([23.0]))
    conditioned_qp_data = conditioning.apply(qp_data)
    conditioned_result = QPSolverQPSwift().solver_call(conditioned_qp_data)
    result = conditioning.unapply(conditioned_result)
    assert np.allclose(result, expected)
