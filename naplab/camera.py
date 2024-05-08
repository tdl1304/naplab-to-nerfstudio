import math
import numpy as np

from naplab.gps import GPSPoint

from .frame_data import FrameData, better_process_data, read_timestamps, save_frames
from .utils import make_homogenous, normalize
import json


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
        self.description = name
    
    def set_description(self, description: str):
        self.description = description
    
    def __repr__(self) -> str:
        return f"Camera({self.decription})"
    
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
        return data.center + data.get_rotation_matrix() @ self.translation
        return np.linalg.inv(self.get_rotation_matrix()) @ self.get_translation_matrix() @ self.get_rotation_matrix() @ data.center
    
    def get_camera_direction_vector(self, data: FrameData):
        """Get the camera direction given rotation matrix"""
        direction = data.get_rotation_matrix() @ self.get_rotation_matrix() @ np.array([1, 0, 0, 1])
        return normalize(direction)


def parse_camera_json(filepath: str) -> list[Camera]:
    with open(filepath, "r") as f:
        data = json.load(f)
        sensors = data["rig"]["sensors"]
        sensors = [sensor for sensor in sensors if sensor["protocol"] == "camera.virtual"]
    
    cameraList = []
    for camera in sensors:
        props = camera["properties"]
        sensorProps = camera["nominalSensor2Rig_FLU"]
        cameraList.append(Camera(camera["name"], float(props["cx"]), float(props["cy"]), int(props["height"]), int(props["width"]), sensorProps["t"], sensorProps["roll-pitch-yaw"]))
    return cameraList

def filter_cameras(cameraList: list[Camera], camera_filter):
    out = [cam for cam in cameraList if cam.name in camera_filter]
    if len(out) == 0:
        raise Exception("No cameras found with the given filter")
    elif len(out) != len(camera_filter):
        raise Exception("Some cameras were not found")
    return out

class ImagesWithTransforms():
    def __init__(self, camera: Camera, source_video: str, timestamp_file: str, gps_left: str, gps_right: str, image_prefix):
        self.camera = camera
        self.image_prefix = image_prefix
        timestamps = read_timestamps(timestamp_file)
        frames = better_process_data(gps_left, gps_right, timestamps)
        self.images_with_transforms = []
        for frame in frames:
            transform = camera.get_transform_matrix(frame.get_translation_matrix(), frame.get_rotation_matrix())
            image_index = timestamps.index(frame.timestamp)
            self.images_with_transforms.append((image_index, transform))
        indices = [it[0] for it in self.images_with_transforms]
        save_frames(source_video, indices, image_prefix=image_prefix)

    
    def get_imagepaths_with_transforms(self):
        return [(f"{self.image_prefix}_frames_output_{it[0]}.png", ) for it in self.images_with_transforms]

def create_transform_json(all_images_transforms: list[ImagesWithTransforms], out_path="transforms.json"):
    pass

def transform_to_colmap(mat4: np.ndarray):
    """
    Takes a 4x4 matrix and converts it to a translation and quaternion
    """
    translation = mat4[3, :3]
    pass