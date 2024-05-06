from dataclasses import InitVar, dataclass
import math

import numpy as np
from .utils import make_homogenous, normalize
from .gps import GPSPoint, process_gps_data
import subprocess
import os

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


def read_timestamps(file_path_to_timestamps: str):
    with open(file_path_to_timestamps, 'r') as f:
        lines = f.readlines()
    return [int(line.split()[1]) for line in lines]


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
    timestamps.sort()
    decasecond = 100_000
    framedatas = []

    interpolated_left = []
    ts_i = 0
    gps_i = 0
    while ts_i < len(timestamps) and gps_i < len(gps_lefts):
        if gps_lefts[gps_i].timestamp > timestamps[ts_i]:
            if gps_i == 0:
                ts_i += 1
                continue
            left_point = gps_lefts[gps_i - 1]
            right_point = gps_lefts[gps_i]
            interpolated_left.append(interpolate(left_point, right_point, timestamps[ts_i]))
            ts_i += 1
        gps_i += 1

    interpolated_right = []
    ts_i = 0
    gps_i = 0
    while ts_i < len(timestamps) and gps_i < len(gps_rights):
        if gps_rights[gps_i].timestamp > timestamps[ts_i]:
            if gps_i == 0:
                ts_i += 1
                continue
            left_point = gps_rights[gps_i - 1]
            right_point = gps_rights[gps_i]
            interpolated_right.append(interpolate(left_point, right_point, timestamps[ts_i]))
            ts_i += 1
        gps_i += 1
    assert len(interpolated_left) == len(interpolated_right)
    for i in range(len(interpolated_left) - 1):
        framedatas.append(FrameData(interpolated_left[i], interpolated_right[i], interpolated_left[i + 1], interpolated_right[i + 1]))
    return framedatas

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

def get_test_data(file_path_left, file_path_right, verbose = False):
    gps_lefts = process_gps_data(file_path_left, verbose)
    timestamps = [point.timestamp for point in gps_lefts][100:-100]
    return better_process_data(file_path_left, file_path_right, timestamps, verbose)

def save_frames(video_path, frame_indexes, output_dir='frames_output'):
    try:
        os.makedirs(output_dir, exist_ok=True)
        for index in frame_indexes:
            output_path = os.path.join(output_dir, f"frame_{index}.png")
            subprocess.run(['ffmpeg', '-i', video_path, '-vf', f"select='eq(n\,{index})'", '-vsync', 'vfr', output_path], capture_output=True, text=True)
    except Exception as e:
        print("Error:", e)
