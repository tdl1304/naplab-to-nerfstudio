from dataclasses import InitVar, dataclass
import math

import numpy as np
from .utils import make_homogenous, normalize
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
        self.center = make_homogenous((gps_left.position + gps_right.position) / 2)
        self.timestamp = (gps_left.timestamp + gps_right.timestamp) / 2

        timestamp_diff = (gps_left.timestamp - gps_right.timestamp)
        
        local_up = normalize(np.cross(gps_left.position - gps_right.position,  gps_right.position - next_gps_left.position))
        local_forward = normalize(np.cross(local_up, gps_left.position - gps_right.position))
        self.forward = local_forward
        #print(local_forward)

        self.pitch = math.asin(-local_forward[1])
        self.yaw = math.atan2(local_forward[0], local_forward[2])

        planeRightX = math.sin(self.yaw)
        planeRightY = -math.cos(self.yaw)
        self.roll = math.asin(local_up[0]*planeRightX + local_up[1]*planeRightY)
        if(local_up[2] < 0):
            self.roll = -1 * self.roll * math.pi - self.roll

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
        R = make_homogenous(R)
        return R
    
    # def get_rotation_matrix(self):
    #     a, b = normalize(self.center[:3]), normalize(self.forward[:3])
    #     v = np.cross(a, b)
    #     c = np.dot(a, b)
    #     s = np.linalg.norm(v)
    #     kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    #     rotation_matrix = np.eye(3) + kmat + np.dot(kmat, kmat) * ((1 - c) / (s ** 2))
    #     return make_homogenous(rotation_matrix)

    
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



def process_frame(file_path_left, file_path_right, verbose=False) -> list[FrameData]:
    gps_lefts = process_gps_data(file_path_left, verbose)
    gps_rights = process_gps_data(file_path_right, verbose)
    data = []
    for i in range(len(gps_lefts) - 2):
            data.append(FrameData(gps_lefts[i], gps_rights[i], gps_lefts[i + 1], gps_rights[i + 1]))
    return data

def better_process_data(file_path_left, file_path_right, timestamps: list[int], verbose = False) -> list[FrameData]:
    gps_lefts = process_gps_data(file_path_left, verbose)
    gps_rights = process_gps_data(file_path_right, verbose)
    # Assumes utc time in microseconds
    decasecond = 100_000
    framedatas = []
    interpolate = True
    if interpolate:
        for timestamp in timestamps:
            gps_left = get_interpolated_GPS_Point(timestamp, gps_lefts)
            gps_right = get_interpolated_GPS_Point(timestamp, gps_rights)
            gps_left_next = get_interpolated_GPS_Point(timestamp + decasecond, gps_lefts)
            gps_right_next = get_interpolated_GPS_Point(timestamp + decasecond, gps_rights)
            framedatas.append(FrameData(gps_left, gps_right, gps_left_next, gps_right_next))
    else:
        for timestamp in timestamps:
            gps_left = get_closest_gps_point(timestamp, gps_lefts)
            gps_right = get_closest_gps_point(timestamp, gps_rights)
            gps_left_next = get_closest_gps_point(timestamp + decasecond, gps_lefts)
            gps_right_next = get_closest_gps_point(timestamp + decasecond, gps_rights)
            framedatas.append(FrameData(gps_left, gps_right, gps_left_next, gps_right_next))
    return framedatas

def get_closest_gps_point(timestamp: int, gps_points: list[GPSPoint]) -> GPSPoint:
    timestamps = np.array([point.timestamp for point in gps_points])
    idx = np.nanargmin(np.abs(timestamps - timestamp))
    return gps_points[idx]


def get_interpolated_GPS_Point(timestamp: int, gps_points: list[GPSPoint]) -> GPSPoint:
    timestamps = np.array([point.timestamp for point in gps_points])
    idx = np.nanargmin(np.abs(timestamps - timestamp))
    closest_timestamp = timestamps[idx]
    if timestamp > closest_timestamp:
        left = closest_timestamp
        left_index = idx
        if len(timestamps) - 1 > idx:
            right = timestamps[idx + 1]
            right_index = idx + 1
        else:
            right = timestamps[idx]
            right_index = idx
    else:
        left = timestamps[idx - 1]
        left_index = idx - 1
        right = closest_timestamp
        right_index = idx
    
    left_diff = timestamp - left
    if left_diff == 0:
        return gps_points[left_index]
    right_diff = right - timestamp
    if right_diff == 0:
        return gps_points[right_index]
    diff_sum = left_diff + right_diff
    left_weight = 1 - left_diff / diff_sum
    right_weight = 1 - right_diff / diff_sum
    interpolated_point = gps_points[left_index].position * left_weight + gps_points[right_index].position * right_weight

    return GPSPoint(timestamp, interpolated_point)

def get_test_data(file_path_left, file_path_right, verbose = False):
    gps_lefts = process_gps_data(file_path_left, verbose)
    timestamps = [point.timestamp for point in gps_lefts][20:-20]
    return better_process_data(file_path_left, file_path_right, timestamps, verbose)