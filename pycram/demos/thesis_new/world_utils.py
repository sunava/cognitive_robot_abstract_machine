import numpy as np

from semantic_digital_twin.datastructures.variables import SpatialVariables
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix


def try_get_body(world, name):
    try:
        return world.get_body_by_name(name)
    except Exception:
        return None


def body_local_aabb(body):
    bbc = body.collision.as_bounding_box_collection_in_frame(body)
    mins = np.array([np.inf, np.inf, np.inf], dtype=float)
    maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
    for bb in bbc.bounding_boxes:
        mins = np.minimum(mins, [bb.min_x, bb.min_y, bb.min_z])
        maxs = np.maximum(maxs, [bb.max_x, bb.max_y, bb.max_z])
    return mins, maxs


def sample_semantic_yz(body, semantic_position):
    bbc = body.collision.as_bounding_box_collection_in_frame(body)
    event_yz = bbc.event.marginal(SpatialVariables.yz)
    return semantic_position.sample_point_from_event(event_yz)


def make_identity_pose_stamped(frame_body):
    return HomogeneousTransformationMatrix.from_xyz_quaternion(
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        reference_frame=frame_body,
    )


def _to_np_matrix(x):
    if hasattr(x, "to_np"):
        return np.asarray(x.to_np(), dtype=float)
    if hasattr(x, "toarray"):
        return np.asarray(x.toarray(), dtype=float)
    if hasattr(x, "full"):
        return np.asarray(x.full(), dtype=float)
    return np.asarray(x, dtype=float)


def Rp_from_spatial(spatial_pose_or_T):
    if hasattr(spatial_pose_or_T, "to_homogeneous_matrix"):
        T = spatial_pose_or_T.to_homogeneous_matrix()
    else:
        T = spatial_pose_or_T

    if hasattr(T, "casadi_sx"):
        T_np = _to_np_matrix(T.casadi_sx)
    else:
        T_np = _to_np_matrix(T)

    T_np = T_np.reshape(4, 4)
    R = T_np[:3, :3]
    p = T_np[:3, 3]
    return R, p
