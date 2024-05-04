from dataclasses import InitVar, dataclass

import numpy as np
from .utils import make_homogenous
from .gps import GPSPoint, process_gps_data

@dataclass
class FrameData:
    gps_left: InitVar[GPSPoint]
    gps_right: InitVar[GPSPoint]

    timestamp: int = None
    file_name: str = None
    left_point: np.ndarray = None
    right_point: np.ndarray = None
    center: np.ndarray = None
    direction: np.array = None

    def __post_init__(self, gps_left: GPSPoint, gps_right: GPSPoint):
        if gps_left.position.shape != gps_right.position.shape:
            raise Exception("mismatched gps position shape")
        self.left_point = make_homogenous(gps_left.position)
        self.right_point = make_homogenous(gps_right.position)
        self.center = make_homogenous((gps_left.position + gps_right.position) / 2)
        self.timestamp = (gps_left.timestamp + gps_right.timestamp) / 2
        up = np.array([0, 0, 1, 1])
        self.direction = np.cross(up[:3], self.center[:3] - self.left_point[:3])
    
    def get_rotation_matrix(self):
        # rotates from world space to car space
        a, b = self.center[:3], self.direction[:3]
        v = np.cross(a, b)
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        rotation_matrix = np.eye(3) + kmat + np.dot(kmat, kmat) * ((1 - c) / (s ** 2))
        return make_homogenous(rotation_matrix)
    
    def get_translation_matrix(self):
        translation = np.zeros((4, 4))
        translation[:, 3] = self.center
        return translation
    
    def get_car_roll(self):
        # project xyz position to xy plane
        projection = np.array([self.center[0], self.center[1], 0, 1])
        # angle between projection and position vector
        angle = np.arccos(np.dot(projection, self.center) / (np.linalg.norm(projection) * np.linalg.norm(self.center)))
        return angle
    
    def get_inverse_rotation_matrix(self):
        # rotates from car space to world space
        return np.linalg.inv(self.get_rotation_matrix())



def process_frame(file_path_left, file_path_right, verbose=False) -> list[FrameData]:
    gps_lefts = process_gps_data(file_path_left, verbose)
    gps_rights = process_gps_data(file_path_right, verbose)
    # need a way to pick timestamp from gpu_lefts and gpu_rights
    data = []
    for left, right in zip(gps_lefts, gps_rights):
        data.append(FrameData(left, right, timestamp=None))
    return data