import os

if os.environ.get("ROS_VERSION") == "1":
    from cora_plex.ros.ros1 import *
elif os.environ.get("ROS_VERSION") == "2":
    from cora_plex.ros.ros2 import *
else:
    from cora_plex.ros.no_ros import *
