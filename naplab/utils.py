import numpy as np

def make_homogenous(tensor: np.ndarray) -> np.ndarray:
    if tensor.shape == (3,):
        out = np.zeros(4)
        out[:3] = tensor
        out[3] = 1
        return out
    elif tensor.shape == (3, 3):
        out = np.zeros((4,4))
        out[3][3] = 1
        out[:3, :3] = tensor
        return out
    else:
        raise Exception("tensor should either be a vec3 or mat3, got shape", tensor.shape)

def normalize(vec: np.ndarray):
    return vec / np.linalg.norm(vec)


def utm_to_blender_rotation():
        """Rotation matrix to rotate UTM33N coordinates into Blender's camera-facing setup"""
        # z = -x, x = y, y = z
        return np.array([
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [-1, 0, 0, 0],
            [0, 0, 0, 1]
        ])