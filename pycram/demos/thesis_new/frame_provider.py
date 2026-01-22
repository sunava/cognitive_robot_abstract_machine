import numpy as np


class Pose:
    def __init__(self, R=None, p=None):
        self.R = (
            np.eye(3, dtype=float)
            if R is None
            else np.asarray(R, dtype=float).reshape(3, 3)
        )
        self.p = (
            np.zeros(3, dtype=float)
            if p is None
            else np.asarray(p, dtype=float).reshape(3)
        )

    def transform_point(self, q_local):
        q_local = np.asarray(q_local, dtype=float).reshape(3)
        return self.p + self.R @ q_local


class FrameProvider:
    def get_pose(self) -> Pose:
        raise NotImplementedError


def pose_from_homogeneous(T):
    T = np.asarray(T, dtype=float).reshape(4, 4)
    return Pose(R=T[:3, :3], p=T[:3, 3])


class WorldTransformFrameProvider(FrameProvider):
    """
    FrameProvider that uses world.transform(...) to compute the pose of a frame in world.root.

    A spatial identity object is created in `source_frame` and transformed into `root_frame`.
    """

    def __init__(
        self, world, source_frame, root_frame=None, make_identity_spatial=None
    ):
        self.world = world
        self.source_frame = source_frame
        self.root_frame = world.root if root_frame is None else root_frame
        if make_identity_spatial is None:
            raise ValueError(
                "make_identity_spatial must be provided (it must return an identity spatial object with reference_frame=source_frame)"
            )
        self.make_identity_spatial = make_identity_spatial

    def get_pose(self) -> Pose:
        ident_in_source = self.make_identity_spatial(self.source_frame)
        ident_in_root = self.world.transform(ident_in_source, self.root_frame)

        if hasattr(ident_in_root, "to_homogeneous_matrix"):
            T = ident_in_root.to_homogeneous_matrix()
            return pose_from_homogeneous(T)

        if hasattr(ident_in_root, "matrix"):
            return pose_from_homogeneous(ident_in_root.matrix)

        if hasattr(ident_in_root, "R") and hasattr(ident_in_root, "p"):
            return Pose(R=ident_in_root.R, p=ident_in_root.p)

        if hasattr(ident_in_root, "rotation_matrix") and hasattr(
            ident_in_root, "translation"
        ):
            return Pose(R=ident_in_root.rotation_matrix, p=ident_in_root.translation)

        T = np.asarray(ident_in_root, dtype=float)
        if T.shape == (4, 4):
            return pose_from_homogeneous(T)

        raise TypeError(
            "Unsupported return type from world.transform(...) for extracting a pose."
        )
