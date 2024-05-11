from dataclasses import InitVar, dataclass
import math
from typing import List

import numpy as np
from .utils import make_homogenous, normalize, utm_to_blender_rotation
from .gps import GPSPoint, process_gps_data

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
        center = (gps_left.position + gps_right.position) / 2
        self.center = make_homogenous(center)
        # gps_left and gps_right timestamps should be equal
        self.timestamp = gps_left.timestamp

        next_center = (next_gps_left.position + next_gps_right.position) / 2
        
        self.up = make_homogenous(normalize(np.cross(gps_right.position - center, next_center - center)))
        self.forward = make_homogenous(normalize(next_center - center))

        self.pitch = math.asin(-self.forward[2])
        self.yaw = math.atan2(self.forward[1], self.forward[0])

        planeRightX = math.sin(self.yaw)
        planeRightY = -math.cos(self.yaw)
        self.roll = math.asin(self.up[0]*planeRightX + self.up[1]*planeRightY)
        if(self.up[2] < 0):
            self.roll = -1 * self.roll * math.pi - self.roll

        self.check_matrices()


    def get_rotation_matrix(self):
        # Create a rotation matrix from roll, pitch, and yaw.
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
        return make_homogenous(R)
    
    def check_matrices(self):
        position_from_tranlate = self.get_translation_matrix() @ np.array([0, 0, 0, 1])
        assert np.sum(np.abs(self.get_translation_matrix() @ np.array([0,0,0,1]) - self.center)) < 0.000001, "translation matrix broken"
        assert np.all(self.get_rotation_matrix() @ np.array([0,0,0,1]) == np.array([0,0,0,1])), "rotation matrix translated vector"
        direction_from_rotation = self.get_rotation_matrix() @ np.array([1, 0, 0, 1])
        direction_error = np.linalg.norm(self.forward - direction_from_rotation) 
        if direction_error > 0.0001:
            print(direction_error)
            print("rotation matrix borked")
            print("actual forward:")
            print(self.forward)
            print("calculated forward:")
            print(direction_from_rotation)
            print("pitch: ", self.pitch)
            print("roll:  ", self.roll)
            print()
    
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

class FrameDataList(list):
    def __init__(self, *args):
        super().__init__(*args)
    
    def take(self, n: int, offset=0, stride=1) -> 'FrameDataList':
        return FrameDataList([self[i] for i in range(offset, min(offset + n * stride, len(self)), stride - 1)])


def read_timestamps(file_path_to_timestamps: str) -> List[int]:
    with open(file_path_to_timestamps, 'r') as f:
        lines = f.readlines()
    return [int(line.split()[1]) for line in lines]

def better_process_data(file_path_left, file_path_right, timestamps: List[int], verbose = False) -> FrameDataList:
    gps_lefts = process_gps_data(file_path_left, verbose)
    gps_rights = process_gps_data(file_path_right, verbose)
    timestamps.sort()

    timestamps = list(filter(lambda ts: gps_lefts[0].timestamp < ts and gps_rights[0].timestamp < ts and ts < gps_lefts[-1].timestamp and ts < gps_rights[-1].timestamp, timestamps))
    framedatas = FrameDataList()

    interpolated_left = interpolate_points(gps_lefts, timestamps)
    interpolated_right = interpolate_points(gps_rights, timestamps)

    assert len(interpolated_left) == len(interpolated_right)
    for i in range(len(interpolated_left) - 1):
        framedatas.append(FrameData(interpolated_left[i], interpolated_right[i], interpolated_left[i + 1], interpolated_right[i + 1]))
    return framedatas

def interpolate_points(gps_points: List[GPSPoint], timestamps: List[int]) -> List[GPSPoint]:
    interpolated= []
    ts_i = 0
    gps_i = 0
    while ts_i < len(timestamps) and gps_i < len(gps_points):
        if gps_points[gps_i].timestamp > timestamps[ts_i]:
            if gps_i == 0:
                raise Exception(f"first timestamp before gps timestamp gps: {gps_points[gps_i].timestamp}, timestamps: {timestamps[ts_i]}")
            left_point = gps_points[gps_i - 1]
            right_point = gps_points[gps_i]
            interpolated.append(interpolate(left_point, right_point, timestamps[ts_i]))
            ts_i += 1
            continue
        gps_i += 1
    return interpolated

def interpolate(gps_left: GPSPoint, gps_right: GPSPoint, timestamp: int) -> GPSPoint:
    """
    returns an interpolated GPSPoint from a left and right point.
    """
    if gps_left.timestamp == timestamp:
        return gps_left
    elif gps_right.timestamp == timestamp:
        return gps_right

    ts_left = gps_left.timestamp
    ts_right = gps_right.timestamp
    dist_left = timestamp - ts_left
    dist_right = ts_right - timestamp
    total_distance = ts_right - ts_left
    assert total_distance == dist_left + dist_right
    dist_left_n = dist_left / total_distance
    dist_right_n = dist_right / total_distance
    interpolated_position = gps_left.position * (1 - dist_left_n) + gps_right.position * (1 - dist_right_n)
    return GPSPoint(timestamp, interpolated_position)

