import math
import numpy as np
from dataclasses import InitVar, dataclass

@dataclass
class GPSPoint:
    timestamp: int
    position: np.ndarray

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

@dataclass
class FrameData:
    gps_left: InitVar[GPSPoint]
    gps_right: InitVar[GPSPoint]
    next_gps_left: InitVar[GPSPoint]
    next_gps_right: InitVar[GPSPoint]


    timestamp: int = None
    file_name: str = None
    left_point: np.ndarray = None
    right_point: np.ndarray = None
    center: np.ndarray = None
    yaw: float = None
    pitch: float = None
    roll: float = None

    def __post_init__(self, gps_left: GPSPoint, gps_right: GPSPoint, next_gps_left: GPSPoint = None, next_gps_right: GPSPoint = None):
        if gps_left.position.shape != gps_right.position.shape:
            raise Exception("mismatched gps position shape")
        self.left_point = make_homogenous(gps_left.position)
        self.right_point = make_homogenous(gps_right.position)
        self.center = make_homogenous((gps_left.position + gps_right.position) / 2)
        self.timestamp = (gps_left.timestamp + gps_right.timestamp) / 2

        local_up = normalize(np.cross(gps_left.position - gps_right.position,  gps_right.position - next_gps_left.position))
        local_forward = normalize(np.cross(local_up, gps_left.position - gps_right.position))
        self.forward = local_forward
        #print(local_forward)

        self.pitch = math.asin(-local_forward[1])
        self.yaw = math.atan2(local_forward[0], local_forward[2])

        planeRightX = math.sin(self.yaw);
        planeRightY = -math.cos(self.yaw);
        self.roll = math.asin(local_up[0]*planeRightX + local_up[1]*planeRightY);
        if(local_up[2] < 0):
            self.roll = -1 * self.roll * math.pi - self.roll;

    def get_rotation_matrix(self):
        """Create a rotation matrix from roll, pitch, and yaw."""
        # Convert angles from degrees to radians
        roll = self.roll
        pitch = self.pitch
        yaw = self.yaw

        # Create rotation matrices for each axis
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # Combine the rotation matrices
        R = Rz @ Ry @ Rx
        R = make_homogenous(R)
        return R
    
    """
    def get_rotation_matrix(self):
        # rotates from world space to car space
        a, b = normalize(self.center[:3]), normalize(self.direction[:3])
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + np.dot(kmat, kmat) * ((1 - c) / (s ** 2))
        return make_homogenous(rotation_matrix)
    """
    
    def get_translation_matrix(self):
        translation = np.identity(4)
        translation[:, 3] = self.center
        return translation
    
    def get_transform(self):
        return self.get_rotation_matrix() @ self.get_translation_matrix()
    
    def get_car_roll(self):
        # project xyz position to xy plane
        projection = np.array([self.center[0], self.center[1], 0, 1])
        # angle between projection and position vector
        angle = np.arccos(np.dot(projection, self.center) / (np.linalg.norm(projection) * np.linalg.norm(self.center)))
        return angle
    
    def get_inverse_rotation_matrix(self):
        # rotates from car space to world space
        return np.linalg.inv(self.get_rotation_matrix())

class Camera():
    def __init__(self, name: str, cx: float, cy: float, height: int, width: int, translation: tuple, roll_pitch_yaw: tuple, fov=60):
        self.name = name
        self.cx = cx
        self.cy = cy
        self.fx = (width * .5) / np.tan(np.radians(fov) / 2)
        self.fy = (height * .5) / np.tan(np.radians(fov) / 2)
        self.height = height
        self.width = width
        self.translation = make_homogenous(np.array(translation))
        self.roll_pitch_yaw = roll_pitch_yaw
    
    def get_camera_intrinsics(self):
        return {
            "camera_model": "OPENCV",
            "fl_x": self.fx,
            "fl_y": self.fy,
            "cx": self.cx,
            "cy": self.cy,
            "w": self.width,
            "h": self.height,
            "k1": 0,
            "k2": 0,
            "p1": 0,
            "p2": 0,
        }
    
    def get_rotation_matrix(self):
        """Create a rotation matrix from roll, pitch, and yaw."""
        # Convert angles from degrees to radians
        roll, pitch, yaw = self.roll_pitch_yaw
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)

        # Create rotation matrices for each axis
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])
        
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])
        
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # Combine the rotation matrices
        R = Rz @ Ry @ Rx
        return make_homogenous(R)
    
    def get_translation_matrix(self):
        translation_matrix = np.identity(4)
        translation_matrix[:, 3] = self.translation
        return translation_matrix
    
    
    def get_transform_matrix(self, car_translation_matrix: np.ndarray, car_rotation_matrix: np.ndarray):
        """Get the translation matrix from the given position"""
        rotation_matrix = self.get_rotation_matrix()

        translation_matrix = self.get_translation_matrix()
        
        camera_local_transform = rotation_matrix @ translation_matrix
        
        transform_matrix = car_translation_matrix @ car_rotation_matrix @ camera_local_transform

        return transform_matrix
    
    
    def get_camera_position(self, data: FrameData):
        """Get the camera position given initial position (x, y, z)"""
        # :)
        return data.get_translation_matrix() @ data.get_rotation_matrix() @ self.translation
        return np.linalg.inv(self.get_rotation_matrix()) @ self.get_translation_matrix() @ self.get_rotation_matrix() @ data.center
    
    def get_camera_direction_vector(self, data: FrameData):
        """Get the camera direction given rotation matrix"""
        direction = data.get_rotation_matrix() @ self.get_rotation_matrix() @ np.array([1, 0, 0, 1])
        direction = direction / np.linalg.norm(direction)
        return direction
    