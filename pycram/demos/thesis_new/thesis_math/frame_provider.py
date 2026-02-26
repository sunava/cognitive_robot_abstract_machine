
from demos.thesis_new.thesis_math.world_utils import make_identity_pose_stamped


def _pose_from_spatial(value):
    """Convert common spatial return types to Pose."""
    if hasattr(value, "R") and hasattr(value, "p"):
        return Pose(R=value.R, p=value.p)

    if hasattr(value, "rotation_matrix") and hasattr(value, "translation"):
        return Pose(R=value.rotation_matrix, p=value.translation)

    if hasattr(value, "to_homogeneous_matrix"):
        T = value.to_homogeneous_matrix()
    elif hasattr(value, "matrix"):
        T = value.matrix
    else:
        T = value

    T = _as_float_array(T)
    if T.shape != (4, 4):
        raise TypeError(
            "Unsupported return type from world.transform(...) for extracting a pose."
        )
    return Pose(R=T[:3, :3], p=T[:3, 3])


class WorldTransformFrameProvider(PoseStamped):
    """
    PoseStamped provider that uses world.transform(...) to compute the pose of a frame in world.root.

    A spatial identity object is created in `source_frame` and transformed into `root_frame`.
    """

    def __init__(
        self, world, source_frame, root_frame=None, make_identity_spatial=None
    ):
        self.world = world
        self.source_frame = source_frame
        self.root_frame = world.root if root_frame is None else root_frame
        self.make_identity_spatial = (
            make_identity_pose_stamped
            if make_identity_spatial is None
            else make_identity_spatial
        )

    def get_pose(self) -> Pose:
        """Resolve the source frame pose into the root frame."""
        ident_in_source = self.make_identity_spatial(self.source_frame)
        ident_in_root = self.world.transform(ident_in_source, self.root_frame)
        return _pose_from_spatial(ident_in_root)
