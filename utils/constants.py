import numpy as np

# imagenet statistics for image normalization
IMG_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMG_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# tcp normalization and gripper width normalization
# TRANS_MIN, TRANS_MAX = np.array([-0.35, -0.35, 0]), np.array([0.35, 0.35, 0.7])
# MAX_GRIPPER_WIDTH = 0.09 # meter
TRANS_MIN, TRANS_MAX = np.array([0.2, -0.4, 0.0]), np.array([0.8, 0.4, 0.7])
# 注意，08.20之前，这里是0.4，这导致高度会出现超过1的数值
# TRANS_MIN, TRANS_MAX = np.array([0.2, -0.4, 0.0]), np.array([0.8, 0.4, 0.4])
# MAX_GRIPPER_WIDTH = 0.09  # meter
# no projection
# TRANS_MIN, TRANS_MAX = np.array([0.2, -0.4, 0]), np.array([0.8, 0.4, 0.4])

# workspace in camera coordinate
WORKSPACE_MIN = np.array([-0.5, -0.5, 0])
WORKSPACE_MAX = np.array([0.5, 0.5, 1.0])

# workspace in world coordinate
WORLD_WORKSPACE_MIN = np.array([0.1, -0.4, -0.1])
WORLD_WORKSPACE_MAX = np.array([0.9, 0.5, 0.9])

# safe workspace in base coordinate
SAFE_EPS = 0.002
# SAFE_WORKSPACE_MIN = np.array([0.2, -0.4, 0])
SAFE_WORKSPACE_MIN = np.array([0.2, -0.4, 0.018])  # 为了不夹到绿布
SAFE_WORKSPACE_MAX = np.array([0.8, 0.4, 0.4])

# gripper threshold (to avoid gripper action too frequently)
GRIPPER_THRESHOLD = 100
# GRIPPER_THRESHOLD = 0.02  # meter
